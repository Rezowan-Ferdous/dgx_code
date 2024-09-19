from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math

# from eval import segment_bars_with_confidence

from modules.attention import Attention_Temporal
from modules.blocks import DropPath,Mlp
from models.mstcn2 import Prediction_Generation

from train.config import Config
config= Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 8
class MySurgFormer(nn.Module):
    def __init__(self,num_classes,in_dim,embed_dim,
                 depth=12,num_heads=6,mlp_ratio= 4.0,qkv_bias=False,qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,):
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = (self.embed_dim) = embed_dim


from models.asformer import ConvFeedForward
class MyAttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, stage, alpha):
        super(MyAttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)

        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        # self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
        #                           stage=stage)  # dilation
        self.att_layer= Attention_Temporal(dim=out_channels)

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        # print("before attention ",out.shape)
        # out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out= self.att_layer(out,x.shape[0])
        # print(f'att out {out.shape}')
        out= out.permute(0,2,1)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
class Encoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        # self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.PG = Prediction_Generation(num_layers=num_layers, num_f_maps=num_f_maps, dim=input_dim, num_classes=num_classes)

        self.layers = nn.ModuleList(
            [MyAttModule(2 ** i, num_f_maps, num_f_maps, 'encoder', alpha) for i in # 2**i
             range(num_layers)])

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

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, alpha):
        super(Decoder, self).__init__()  # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [MyAttModule(2 ** i, num_f_maps, num_f_maps, 'decoder', alpha) for i in  # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature

from models.asformer import exponential_descrease

class MyAsformer(nn.Module):
    def __init__(self, num_decoders, num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate,device):
        super(MyAsformer, self).__init__()
        self.encoder = Encoder(num_layers, num_f_maps, input_dim, num_classes, channel_masking_rate,
                               att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, num_f_maps, num_classes, num_classes,
                    alpha=exponential_descrease(s))) for s in range(num_decoders)])  # num_decoders

        self.conv_bound = nn.Conv1d(num_f_maps, 3, 1)
    def forward(self, x, mask):
        mask= mask.unsqueeze(1).to(device)

        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        bounds = self.conv_bound(feature)

        return outputs,bounds


class MyActionSegmentationRefinement(nn.Module):
    def __init__(
            self,
            in_channel: int,
            n_features: int,
            n_classes: int,
            n_stages: int,
            n_layers: int,
            channel_masking_rate:float,
            n_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.2,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            n_stages_asb: Optional[int] = None,
            n_stages_brb: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(n_features)


        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(n_features)
        self.temporal_attn = Attention_Temporal(
            n_features,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.temporal_fc = nn.Linear(n_features, n_features)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(n_features)
        mlp_hidden_dim = int(n_features * mlp_ratio)
        self.mlp = Mlp(
            in_features=n_features,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.model = MySurgFormer(3, n_layers, 2, 2, n_features, in_channel, n_classes, channel_masking_rate)
        self.num_classes = n_classes

