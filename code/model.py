import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

from utils import gen_adj


#############################################################
##### Architecture of Generator #############################
#############################################################

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class affine(nn.Module):

    def __init__(self, num_features, emb_dim=256):
        super(affine, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(emb_dim, emb_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(emb_dim, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(emb_dim, emb_dim)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(emb_dim, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.fc_gamma.linear2.weight.data) # zeros
        nn.init.zeros_(self.fc_gamma.linear2.bias.data) # ones
        nn.init.zeros_(self.fc_beta.linear2.weight.data) # zeros
        nn.init.zeros_(self.fc_beta.linear2.bias.data) # zeros
        # nn.init.zeros_(self.fc_gamma.linear2.weight.data) # zeros
        # nn.init.ones_(self.fc_gamma.linear2.bias.data) # ones
        # nn.init.zeros_(self.fc_beta.linear2.weight.data) # zeros
        # nn.init.zeros_(self.fc_beta.linear2.bias.data)
        
    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch, emb_dim=256):
        super(G_Block, self).__init__()
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch, emb_dim=emb_dim)
        self.affine1 = affine(in_ch, emb_dim=emb_dim)
        self.affine2 = affine(out_ch, emb_dim=emb_dim)
        self.affine3 = affine(out_ch, emb_dim=emb_dim)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5) # zeros
        # self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None, z=None):
        return (1. - self.gamma) * self.shortcut(x) + self.gamma * self.residual(x, y, z)
        # return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None, z=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.c1(h)     
        h = self.affine2(h, z)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine3(h, z)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return self.c2(h)


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100, attr_phrase_emb=None, adj=None, graph_inp_dim=256, num_class=312):
        super(NetG, self).__init__()
        self.ngf = ngf
        assert attr_phrase_emb is not None # (num_class, 256)
        # self.register_buffer("attr_phrase_emb", attr_phrase_emb)
        self.attr_phrase_emb = nn.Parameter(attr_phrase_emb)
        assert adj is not None 
        # self.A = nn.Parameter(adj)
        self.register_buffer("A", adj)
        self.num_class = num_class
        self.gcn1 = GraphConvolution(graph_inp_dim, self.num_class)
        self.gcn2 = GraphConvolution(self.num_class, graph_inp_dim)
        self.relu = nn.LeakyReLU(0.2)
        
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8)#8x8
        self.block3 = G_Block(ngf * 8, ngf * 8)#16x16
        self.block4 = G_Block(ngf * 8, ngf * 4)#32x32
        self.block5 = G_Block(ngf * 4, ngf * 2)#64x64
        self.block6 = G_Block(ngf * 2, ngf * 1)#128x128
        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c, label, return_emb=False):
        adj = gen_adj(self.A)
        x_kernel = self.gcn1(self.attr_phrase_emb, adj)
        x_kernel = self.relu(x_kernel)
        x_kernel = self.gcn2(x_kernel, adj)
        label_emb = torch.matmul(label, x_kernel)
        
        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)
        out = self.block0(out, label_emb, label_emb)
        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out, label_emb, label_emb)
        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out, label_emb, label_emb)
        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out, c, c)
        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out, c, c)
        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out, c, c)
        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out, c, c)
        out = self.conv_img(out)
        if return_emb:
            return out, label_emb
        else:
            return out
        
        

#############################################################
##### Architecture of Generator #############################
#############################################################

class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        # self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        # if self.learned_shortcut:
        x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


class NetD_Get_Logits_Lables(nn.Module):
    def __init__(self, ndf, num_classes, emb_dim=256):
        super(NetD_Get_Logits_Lables, self).__init__()
        self.df_dim = ndf
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + self.emb_dim, ndf * 16, 3, 1, 1, bias=False)
        )
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        )
        self.num_classes = num_classes * 2
        self.final_linear = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf * 16, self.num_classes, bias=False)
        )

    def forward(self, out, emb, label, adc_fake=False):             
        # emb
        emb = emb.view(-1, self.emb_dim, 1, 1)
        emb = emb.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, emb), 1)
        joint_features = self.joint_conv(h_c_code)
        adv_logits = self.final_conv(joint_features)
        joint_feat_pooled = torch.sum(joint_features, dim=[2,3])
        for W in self.final_linear.parameters():
            W = F.normalize(W, dim=1)
        joint_feat_pooled = F.normalize(joint_feat_pooled, dim=1)
        cls_logits = self.final_linear(joint_feat_pooled)
        label = label.unsqueeze(1).repeat(1, 2, 1).permute([0, 2, 1]).contiguous()
        batch_size = label.shape[0]
        label = label.reshape(batch_size, self.num_classes)
        if adc_fake:
            label_tmp = label.clone()
            label_tmp[:, 1:] = label[:, :-1]
            label_tmp[:, 0] = 0
        else:
            label_tmp = label.clone()
        output = {}
        output["adv_logits"] = adv_logits
        output["cls_logits"] = cls_logits
        output["label"] = label_tmp
        return output


class NetD(nn.Module):
    def __init__(self, ndf, emb_dim=256):
        super(NetD, self).__init__()
        
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        self.block5 = resD(ndf * 16, ndf * 16)#4
        self.pool_fc = nn.Linear(ndf * 16, emb_dim)

    def forward(self,x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        img_feat = self.block5(out)
        img_feat_pooled = self.pool_fc(torch.sum(img_feat, [2, 3])) 
        output = {}
        output["img_feat"] = img_feat
        output["img_feat_pooled"] = img_feat_pooled
        return output