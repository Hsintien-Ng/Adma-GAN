from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER

import os
import sys
import logging
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.distributed as dist
from model import NetG, NetD, NetD_Get_Logits_Lables
from utils import load_adj, gen_adj, contrastive_loss
import torchvision.utils as vutils

from calc_fid import calc_fid, load_patched_inception_v3

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)


class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, criterion, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param
        self.criterion = criterion

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        # weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - self.criterion(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


def get_logger(filename, verbosity=1, name=None, pretrained=False):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    log_op = "a" if pretrained else "w"
    fh = logging.FileHandler(filename, log_op)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def extract_feature_from_samples(dataloader, generator, text_encoder, inception, device):
    features = []
    for idx, data in tqdm(enumerate(dataloader, 0)):
        imags, captions, cap_lens, attr, attr_id, keys = prepare_data(data, attr_float=True)
        hidden = text_encoder.module.init_hidden(captions.size(0)) if cfg.distributed else text_encoder.init_hidden(captions.size(0))
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        imgs = imags[0].to(device)
        noise = torch.randn(imgs.shape[0], 100)
        noise = noise.to(device)
        attr_label = attr.to(device)
        attr_label = attr_label.float()
        fake = generator(noise, sent_emb, attr_label)
        fake_feat = inception(fake)[0].view(fake.shape[0], -1)
        features.append(fake_feat.to("cpu"))
    features = torch.cat(features, 0)
    return features


def train(train_dataloader, valid_dataloader,
        netG, netG_ema, netD, netD_ema, netDL2, netDL2_ema, text_encoder, 
        optimizerG, optimizerD, 
        state_epoch, batch_size_dict, device): 
    os.makedirs(os.path.join(cfg.SAVE_IMAGE_DIR, cfg.CONFIG_NAME), exist_ok=True)
    os.makedirs(os.path.join(cfg.SAVE_CKPT_DIR, cfg.CONFIG_NAME), exist_ok=True)
    
    count = 0
    if get_rank() == 0 and cfg.TRAIN.FID:
        with open(cfg.FID_feats_real_path, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
        inception = load_patched_inception_v3().to(device)
        inception.eval()
        logger.info('======Evaluating Metrics======')
        features_fake = extract_feature_from_samples(valid_dataloader, netG_ema, text_encoder, inception, device).numpy()
        fake_mean = np.mean(features_fake, 0)
        fake_cov = np.cov(features_fake, rowvar=False)
        fid = calc_fid(fake_mean, fake_cov, real_mean, real_cov)
        logger.info(f'Epoch -1: fid={fid:.5f}')
        logger.info('==============================')

    criterion_MLcls = nn.BCEWithLogitsLoss().to(device)
    accum = 0.5 ** (32 / (10 * 1000))
    loss_dict = {}
    best_fid = np.inf
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        for step, data in enumerate(train_dataloader, 0):
            
            ###### prepare image and text embeddings ######
            imags, captions, cap_lens, attr, attr_id, keys = prepare_data(data, attr_float=True)
            hidden = text_encoder.module.init_hidden(batch_size_dict['train']) if cfg.distributed else text_encoder.init_hidden(batch_size_dict['train'])
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            imgs = imags[0].to(device)
            noise = torch.randn(batch_size_dict['train'], 100)
            noise = noise.to(device)
            attr = attr.to(device)
            attr_label = attr.clone()
            attr_label = attr_label.float()
            batch_size = imgs.shape[0]


            ###### update D ######
            requires_grad(netG, False)
            requires_grad(netD, True)
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            ### update for generated images
            real_features_dict = netD(imgs)
            real_features, real_features_pooled = real_features_dict["img_feat"], real_features_dict["img_feat_pooled"]
            output_real = netDL2(real_features, sent_emb, attr_label, adc_fake=False)
            logits_sent_real = output_real["adv_logits"]
            loss_Dadv_real = torch.nn.ReLU()(1.0 - logits_sent_real).mean() 
            logits_cls_real = output_real["cls_logits"]
            attr_real_label = output_real["label"]
            loss_Dcls_real = criterion_MLcls(logits_cls_real, attr_real_label)
            
            output_mis = netDL2(real_features[:(batch_size - 1)], sent_emb[1:batch_size], attr_label[:(batch_size - 1)], adc_fake=False)
            logits_sent_mis = output_mis["adv_logits"]
            loss_Dadv_mis = torch.nn.ReLU()(1.0 + logits_sent_mis).mean()
            logits_cls_mis = output_mis["cls_logits"]
            loss_Dcls_mis = logits_cls_mis.mean() * 0.
            
            fake, label_emb = netG(noise, sent_emb, attr_label, return_emb=True)
            fake_features_dict = netD(fake.detach())
            fake_features = fake_features_dict["img_feat"]
            output_fake = netDL2(fake_features, sent_emb, attr_label.detach(), adc_fake=True)
            logits_sent_fake = output_fake["adv_logits"]
            loss_Dadv_fake = torch.nn.ReLU()(1.0 + logits_sent_fake).mean()
            logits_cls_fake = output_fake["cls_logits"]
            attr_fake_label = output_fake["label"]
            loss_Dcls_fake = criterion_MLcls(logits_cls_fake, attr_fake_label)

            loss_Dadv = (loss_Dadv_fake + loss_Dadv_mis) / 2. + loss_Dadv_real
            loss_Dcls = loss_Dcls_fake + loss_Dcls_real + loss_Dcls_mis       
            # cl
            d_sent_cl_loss = contrastive_loss(real_features_pooled, sent_emb.detach(), device=device)
            d_label_cl_loss = contrastive_loss(real_features_pooled, label_emb.detach(), device=device)
            loss_Dc = d_label_cl_loss + d_sent_cl_loss

            loss_Dall = loss_Dadv + cfg.TRAIN.weight_cls * loss_Dcls + cfg.TRAIN.weight_c * loss_Dc
            loss_Dall.backward()
            optimizerD.step()
            loss_dict["Loss_Dadv"] = loss_Dadv
            loss_dict["Loss_Dcls"] = loss_Dcls
            loss_dict["Loss_Dc"] = loss_Dc

            ###### MA-GP ######
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            interpolated = imgs.clone().requires_grad_()
            sent_inter = sent_emb.clone().requires_grad_()
            attr_inter = attr_label.clone()
            interpolated_features_dict = netD(interpolated)
            inter_features = interpolated_features_dict["img_feat"]
            inter_features_pooled = interpolated_features_dict["img_feat_pooled"]
            output = netDL2(inter_features, sent_inter, attr_inter)
            out_sent_adv = output["adv_logits"]
            out_cls = output["cls_logits"]
            grads = torch.autograd.grad(outputs=out_sent_adv,
                                    inputs=(interpolated, sent_inter),
                                    grad_outputs=torch.ones(out_sent_adv.size()).to(device),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0,grad1),dim=1)                        
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp + out_cls.mean() * 0. + inter_features.mean() * 0. + inter_features_pooled.mean() * 0.
            d_loss.backward()
            optimizerD.step()
            
            ###### update G ######
            requires_grad(netG, True)
            requires_grad(netD, False)
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            fake, label_emb = netG(noise, sent_emb, attr_label, return_emb=True)
            fake_features_dict = netD(fake)
            fake_features, fake_features_pooled = fake_features_dict["img_feat"], fake_features_dict["img_feat_pooled"]
            output = netDL2(fake_features, sent_emb, attr_label, adc_fake=True)
            # adv loss
            logits_sent_gen = output["adv_logits"]
            loss_Gadv = - logits_sent_gen.mean()
            # cls loss
            logits_cls = output["cls_logits"]
            attr_gen_label = output["label"]
            loss_Gcls = criterion_MLcls(logits_cls, attr_gen_label)
            # cl loss
            real_features_dict = netD(imgs)
            real_features_pooled = real_features_dict["img_feat_pooled"]
            img_cl_loss = contrastive_loss(fake_features_pooled, real_features_pooled, device=device)
            g_sent_cl_loss = contrastive_loss(fake_features_pooled, sent_emb, device=device)
            g_label_cl_loss = contrastive_loss(fake_features_pooled, label_emb, device=device)
            loss_Gc = img_cl_loss + g_sent_cl_loss + g_label_cl_loss
            loss_Gall = loss_Gadv - cfg.TRAIN.weight_cls * loss_Gcls + cfg.TRAIN.weight_c * loss_Gc
            loss_Gall.backward()
            optimizerG.step()
            loss_dict["Loss_Gadv"] = loss_Gadv
            loss_dict["Loss_Gcls"] = loss_Gcls
            loss_dict["Loss_Gc"] = loss_Gc
            
            netD_module = netD.module if cfg.distributed else netD
            netDL2_module = netDL2.module if cfg.distributed else netDL2
            netG_module = netG.module if cfg.distributed else netG
            accumulate(netD_ema, netD_module, accum)
            accumulate(netDL2_ema, netDL2_module, accum)
            accumulate(netG_ema, netG_module, accum)

            loss_reduced = reduce_loss_dict(loss_dict)
            loss_Dadv_val = loss_reduced["Loss_Dadv"].mean().item()
            loss_Dcls_val = loss_reduced["Loss_Dcls"].mean().item()
            loss_Dc_val = loss_reduced["Loss_Dc"].mean().item()
            loss_Gadv_val = loss_reduced["Loss_Gadv"].mean().item()
            loss_Gcls_val = loss_reduced["Loss_Gcls"].mean().item()
            loss_Gc_val = loss_reduced["Loss_Gc"].mean().item()
            
            if get_rank() == 0 and step % 10 == 0:
                logger.info('[%d/%d][%d/%d] Loss_D: %.3f Loss_Dcls: %.3f Loss_Dc: %.3f Loss_G: %.3f Loss_Gcls: %.3f Loss_Gc: %.3f'
                    % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(train_dataloader), loss_Dadv_val, loss_Dcls_val, loss_Dc_val, loss_Gadv_val, loss_Gcls_val, loss_Gc_val))
                # logger.info(1)

        if get_rank() == 0:
            with torch.no_grad():
                vis_fake = netG_ema(noise, sent_emb, attr_label)
            vutils.save_image(vis_fake.data,
                            '%s/%s/fake_samples_epoch_%03d.png' % (cfg.SAVE_IMAGE_DIR, cfg.CONFIG_NAME, epoch),
                            normalize=True)
            vutils.save_image(imgs.data,
                            '%s/%s/real_samples_epoch_%03d.png' % (cfg.SAVE_IMAGE_DIR, cfg.CONFIG_NAME, epoch),
                            normalize=True)
            if cfg.TRAIN.FID:
                logger.info('======Evaluating Metrics======')
                features_fake = extract_feature_from_samples(valid_dataloader, netG_ema, text_encoder, inception, device).numpy()
                fake_mean = np.mean(features_fake, 0)
                fake_cov = np.cov(features_fake, rowvar=False)
                fid = calc_fid(fake_mean, fake_cov, real_mean, real_cov)
                if fid < best_fid:
                    net_best = {}
                    netG_module = netG.module if cfg.distributed else netG
                    netD_module = netD.module if cfg.distributed else netD
                    netDL2_module = netDL2.module if cfg.distributed else netDL2
                    net_best["netG"] = netG_module.state_dict()
                    net_best["netD"] = netD_module.state_dict()
                    net_best["netDL2"] = netDL2_module.state_dict()
                    net_best["netG_ema"] = netG_ema.state_dict()
                    net_best["netD_ema"] = netD_ema.state_dict()
                    net_best["netDL2_ema"] = netDL2_ema.state_dict()
                    torch.save(net_best, '%s/%s/net_best.pth' % (cfg.SAVE_CKPT_DIR, cfg.CONFIG_NAME))
                    best_fid = fid
                logger.info(f'Epoch {epoch}: fid={fid:.5f}')
                logger.info('==============================')
            
            if epoch % cfg.save_per_epoch == 0:
                net_current = {}
                netG_module = netG.module if cfg.distributed else netG
                netD_module = netD.module if cfg.distributed else netD
                netDL2_module = netDL2.module if cfg.distributed else netDL2
                net_current["netG"] = netG_module.state_dict()
                net_current["netD"] = netD_module.state_dict()
                net_current["netDL2"] = netDL2_module.state_dict()
                net_current["netG_ema"] = netG_ema.state_dict()
                net_current["netD_ema"] = netD_ema.state_dict()
                net_current["netDL2_ema"] = netDL2_ema.state_dict()
                net_current["optG"] = optimizerG.state_dict()
                net_current["optD"] = optimizerD.state_dict()
                net_current["epoch"] = epoch
                torch.save(net_current, '%s/%s/net_current.pth' % (cfg.SAVE_CKPT_DIR, cfg.CONFIG_NAME))
    return count



if __name__ == "__main__":
    
    cli_conf = OmegaConf.from_cli()
    base_conf_path = cli_conf.get('--config', None)

    base_conf = OmegaConf.load(base_conf_path)
    cfg = OmegaConf.merge(base_conf, cli_conf)

    os.makedirs(os.path.join(cfg.SAVE_LOG_DIR), exist_ok=True) 
    log_path = os.path.join(cfg.SAVE_LOG_DIR, f'{cfg.CONFIG_NAME}_log.txt')
    logger = get_logger(log_path, pretrained=(cfg.pretrained_path != ''))
    logger.info('Using config:')
    logger.info(cfg)

    if cfg.seed == -1:
        seed = random.randint(0, 1e5)
    else:
        seed = cfg.seed
    
    logger.info(f"seed now is : {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.distributed = n_gpu > 1
    if cfg.distributed:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Get data loader ##################################################
    imsize = cfg.img_size
    train_batch_size = cfg.TRAIN.BATCH_SIZE
    test_batch_size = cfg.TEST.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    train_dataset = TextDataset(cfg, split="train", transform=image_transform)
    assert train_dataset
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=data_sampler(train_dataset, shuffle=True, distributed=cfg.distributed), drop_last=True)
    valid_dataset = TextDataset(cfg, split="test", transform=image_transform)
    assert valid_dataset
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False,)

    # Load preprocessed attribute text embeddings
    with open(cfg.TEXT.ATTRIBUTE_CAPTION_EMBEDDING_PATH, 'rb') as f:
        attr_cap_emb = pickle.load(f)
    init_class_emb = torch.from_numpy(attr_cap_emb['phrase_emb']) # (312, 256)
    
    A = torch.from_numpy(load_adj(cfg.NUM_CLASS, 0.4, cfg.Attributes_A_path)).float()
    adj = gen_adj(A)

    # Create model
    netG = NetG(cfg.TRAIN.NF, 100, init_class_emb, adj).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    netDL2 = NetD_Get_Logits_Lables(cfg.TRAIN.NF, cfg.NUM_CLASS).to(device)
    netG_ema = NetG(cfg.TRAIN.NF, 100, init_class_emb, adj).to(device)
    netG_ema.eval()
    accumulate(netG_ema, netG, 0)
    netD_ema = NetD(cfg.TRAIN.NF).to(device)
    netDL2_ema = NetD_Get_Logits_Lables(cfg.TRAIN.NF, cfg.NUM_CLASS).to(device)
    netD_ema.eval()
    netDL2_ema.eval()
    accumulate(netD_ema, netD, 0)
    accumulate(netDL2_ema, netDL2, 0)
    if cfg.pretrained_path != '':
        ckpt = torch.load(os.path.join(cfg.pretrained_path, "net_current.pth"))
        netG.load_state_dict(ckpt["netG"])
        netG_ema.load_state_dict(ckpt["netG_ema"])
        netD.load_state_dict(ckpt["netD"])
        netD_ema.load_state_dict(ckpt["netD_ema"])
        netDL2.load_state_dict(ckpt["netDL2"])
        netDL2_ema.load_state_dict(ckpt["netDL2_ema"])
        state_epoch = ckpt["epoch"]
    else:
        state_epoch = 0

    text_encoder = RNN_ENCODER(cfg, train_dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM).to(device)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    if cfg.distributed:
        netG = nn.parallel.DistributedDataParallel(
            netG,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=False,
        )
        netD = nn.parallel.DistributedDataParallel(
            netD,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=False,
        )
        netDL2 = nn.parallel.DistributedDataParallel(
            netDL2,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=False,
        )
        text_encoder.requires_grad_(True)
        text_encoder = nn.parallel.DistributedDataParallel(
            text_encoder,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=False,
        )
        text_encoder.requires_grad_(False)

    netD_parameters = list(netD.parameters()) + list(netDL2.parameters())
    optimizerG = torch.optim.Adam(netG.parameters(), lr=cfg.TRAIN.g_lr, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD_parameters, lr=cfg.TRAIN.d_lr, betas=(0.0, 0.9)) 
    if cfg.pretrained_path != '':
        optimizerG.load_state_dict(ckpt["optG"])
        optimizerD.load_state_dict(ckpt["optD"])

    batch_size_dict = {'train': train_batch_size, 'test': test_batch_size}

    count = train(train_dataloader, valid_dataloader,
                    netG, netG_ema, netD, netD_ema, netDL2, netDL2_ema, text_encoder, 
                    optimizerG, optimizerD, 
                    state_epoch, batch_size_dict, device)
    logger.info('Finish training !')

        