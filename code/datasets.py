from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
# from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from omegaconf import OmegaConf


def prepare_data(data, return_indices=False, attr_float=False, CUDA=True):
    imgs, captions, captions_lens, attr, attr_id, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    attr = attr[sorted_cap_indices].long() if not attr_float else attr[sorted_cap_indices].float()
    attr_id = attr_id[sorted_cap_indices].numpy().tolist()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    if return_indices:
        return [real_imgs, captions, sorted_cap_lens,
            attr, attr_id, keys, sorted_cap_indices]
    else:
        return [real_imgs, captions, sorted_cap_lens,
            attr, attr_id, keys]


def get_imgs(img_path, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
        
    ret = []
    ret.append(normalize(img))

    return ret


def cv_random_flip(img, seg):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
    return img, seg

def randomCrop(image, seg):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), seg.crop(random_region),

def get_params(img, output_size):
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

def newRandomCrop(img, seg, imsize):
    i, j, h, w = get_params(img, (imsize, imsize))
    img = transforms.functional.crop(img, i, j, h, w)
    seg = transforms.functional.crop(seg, i, j, h, w)
    return img, seg

def newRandomFlip(img, seg):
    if random.random() < 0.5:
        return (transforms.functional.hflip(img), transforms.functional.hflip(seg))
    return img, seg

def get_img_segs(img_path, seg_path, imsize, transform):
    img = Image.open(img_path).convert('RGB')
    seg = Image.open(seg_path)
    
    w, h = imsize, imsize
    rw, rh = int(w * 76 / 64), int(h * 76 / 64)
    img = img.resize((rw, rh), Image.BILINEAR)
    # seg = seg.resize((w, h), Image.NEAREST)
    seg = seg.resize((rw, rh), Image.NEAREST)
    
    img, seg = newRandomCrop(img, seg, imsize)
    img, seg = newRandomFlip(img, seg)
    
    img = transform(img)
    seg = np.array(seg)
    seg =  torch.from_numpy(seg)

    return img, seg


class TextDataset(data.Dataset):
    def __init__(self, cfg, split='train', use_num_text_per_image=1, transform=None):
        
        self.cfg = cfg
        data_dir = cfg.DATA_DIR
        self.option = cfg.DATASET_NAME # (coco, birds)
        assert self.option in ['coco', 'birds']
        
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.caption_per_image = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.use_num_text_per_image = use_num_text_per_image
        
        self.split = split
        self.data = []
        self.data_dir = data_dir
        image_name = 'coco_image' if self.option == 'coco' else 'image'
        caption_name = 'coco_caption' if self.option == 'coco' else 'caption'
        self.image_dir = os.path.join(self.data_dir, image_name)
        self.caption_dir = os.path.join(self.data_dir, caption_name)
        # if data_dir.find('birds') != -1:
        if self.option == 'birds':
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)
        
        # not all images contained in image-object pair
        if self.option == 'coco':
            with open(os.path.join(self.data_dir, f'annotations/data/{split}_anno.json'), 'rb') as f:
                anno = json.load(f)
            self.filenames_coco = []
            for an in anno:
                self.filenames_coco.append(an['file_name'][:-4])
        else:
            self.filenames_coco = None
        
        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(self.caption_dir, split)
        
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

        pair_name = cfg.PAIR_NAME # if self.option == 'birds' else 'image_objects_pair.pickle'
        self.images_attributes_pair = pickle.load(open(os.path.join(self.data_dir, pair_name), 'rb'))

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, split, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/%s2014/%s.txt' % (data_dir, split, filenames[i])
            with open(cap_path, "rb") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.caption_per_image:
                        break
                if cnt < self.caption_per_image:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        pickle_name = 'captions.pickle'
        filepath = os.path.join(data_dir, pickle_name)
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, 'train', train_names)
            test_captions = self.load_captions(data_dir, 'test', test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                # n_words = len(ixtoword) if self.option == 'birds' else 27297
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
            # filter image
            if self.filenames_coco is not None:
                captions_new = []
                filenames_new = []
                for idx_name, name in enumerate(train_names):
                    if name not in self.filenames_coco:
                        continue
                    else:
                        filenames_new.append(name)
                        for idx_num in range(self.caption_per_image):
                            captions_new.append(captions[idx_name * self.caption_per_image + idx_num])
                captions = captions_new
                filenames = filenames_new  
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
            if self.filenames_coco is not None:
                captions_new = []
                filenames_new = []
                for idx_name, name in enumerate(test_names):
                    if name not in self.filenames_coco:
                        continue
                    else:
                        filenames_new.append(name)
                        for idx_num in range(self.caption_per_image):
                            captions_new.append(captions[idx_name * self.caption_per_image + idx_num])
                captions = captions_new
                filenames = filenames_new
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_attributes(self, key):
        if self.option == 'coco':
            new_key = key + '.jpg'
        else:
            new_key = key.split('.')[1] + '.jpg'
        attributes = self.images_attributes_pair[new_key]
        new_attributes = [int(attr) for attr in attributes]
        return np.asarray(new_attributes)

    def __getitem__(self, index):
        
        # get an image
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        if self.option == 'coco':
            img_name = f'{self.image_dir}/{self.split}/{self.split}2014/{key}.jpg'
        else:
            img_name = f'{self.image_dir}/{key}.jpg'
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        
        # random select a sentence
        sent_ix_list = random.choices(range(self.caption_per_image), k=self.use_num_text_per_image)
        caps_list, cap_len_list = [], []
        for sent_ix in sent_ix_list:
            new_sent_ix = index * self.caption_per_image + sent_ix
            caps, cap_len = self.get_caption(new_sent_ix)
            caps_list.append(caps)
            cap_len_list.append(cap_len)
        if len(caps_list) == 1:
            caps = caps_list[0]
            cap_len = cap_len_list[0]
        else:
            caps = torch.tensor(caps_list)
            cap_len = torch.tensor(cap_len_list)

        # get attributes
        attributes = self.get_attributes(key)
        
        return imgs, caps, cap_len, attributes, 0, key

    def __len__(self):
        return len(self.filenames)


    

if __name__ == '__main__':
    imsize = 256
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    # cfg.TEXT.CAPTIONS_PER_IMAGE = 5
    # train_dataset = TextProDataset("/ceph-jd/pub/jupyter/lixi/notebooks/xintian/data/COCO2014", epoch=60, step_epoch=20, option='coco',
    #                         split='train',
    #                         base_size=256,
    #                         transform=image_transform,
    #                         attribute_option='all')
    
    cli_conf = OmegaConf.from_cli()
    base_conf_path = cli_conf.get('--config', None)

    base_conf = OmegaConf.load(base_conf_path)
    conf = OmegaConf.merge(base_conf, cli_conf)
    
    train_dataset = TextDataset(conf, split="train", transform=image_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    for idx, data in enumerate(train_dataloader, 0):
        imgs, captions, cap_lens, attr, attr_id, keys = data
        import pdb
        pdb.set_trace()
        break