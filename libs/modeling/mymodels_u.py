from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math
# from .attention import Attention_Temporal
# from .blocks import DropPath,Mlp
from mstcn2 import Prediction_Generation
from asformer import ConvFeedForward,MultiHeadAttLayer,AttLayer,AttModule,exponential_descrease

MyAttSlidModule= AttModule

class MyEncoder():
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha, num_head):
        super(MyEncoder, self).__init__()
        # self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.PG = Prediction_Generation(num_layers=num_layers, num_f_maps=num_f_maps, dim=input_dim, num_classes=num_classes)

        # self.layers = nn.ModuleList(
        #     [MyAttModule(2 ** i, num_f_maps, num_f_maps, 'encoder', alpha) for i in # 2**i
        #      range(num_layers)])
        self.layers = nn.ModuleList([MyAttSlidModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha,num_head)
                                     for i in range(num_layers)])

        self.proj_combined = nn.Linear(num_f_maps, num_f_maps)

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        # feature = self.conv_1x1(x)
        o , feature = self.PG(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        # # for layer in self.layers:
        # feature = self.layers(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

class MyDecoder():
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha ,num_head,max_len=40000):
        super(MyDecoder, self).__init__()  # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [MyAttSlidModule( 2** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder',att_type, alpha,num_head) for i in  # 2 ** i
             range(num_layers)])
        # self.layers =MyAttModule(3, num_f_maps, num_f_maps, 'decoder',att_type, alpha)

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

class MyTransformeru(nn.Module):
    def __init__(self,num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,num_head,device):
        super(MyTransformeru, self).__init__()
        self.encoder = MyEncoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes,0.3,'sliding_att', 1, num_head=num_head)
        self.decoders = nn.ModuleList([copy.deepcopy(MyDecoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes,att_type='sliding_att',
                    alpha=exponential_descrease(s),num_head=num_head)) for s in range(num_decoders)])  # num_decoders

        self.conv_bound = nn.Conv1d(num_f_maps, 3, 1)
        self.device=device

    def forward(self, x, mask):
        mask= mask.unsqueeze(1).to(self.device)

        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        bounds = self.conv_bound(feature)

        return outputs,bounds

class MyTesttransformer(nn.Module):
    def __init__(self,num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,num_head,device):
        super(MyTransformeru, self).__init__()
        self.encoder = MyEncoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes,
                               att_type='sliding_att', alpha=1,channel_masking_rate=channel_masking_rate,num_head=num_head)
        self.decoders = nn.ModuleList([copy.deepcopy(MyDecoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes,att_type='sliding_att',
                    alpha=exponential_descrease(s),num_head=num_head)) for s in range(num_decoders)])  # num_decoders

        self.conv_bound = nn.Conv1d(num_f_maps, 3, 1)
        self.device=device
    def forward(self, x, mask):
        mask= mask.unsqueeze(1).to(self.device)

        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        bounds = self.conv_bound(feature)

        return outputs,bounds



