import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[: x.size(0), :] + x
        return pos


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            # print(self.position_ids)
            position_ids = self.position_ids[:, : x.size(2)]

        # print(self.pe(position_ids).size(), x.size())

        position_embeddings = self.pe(position_ids).transpose(1, 2) + x
        return position_embeddings


import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List
import torch.nn.init as init
import copy


# class SelfAttention(nn.Module):
#     def __init__(
#         self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
#     ):
#         super().__init__()
#         self.num_heads = heads
#         head_dim = dim // heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout_rate)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = (
#             self.qkv(x)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = (
#             qkv[0],
#             qkv[1],
#             qkv[2],
#         )  # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        ))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        ))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        T, B, C = memory.shape
        intermediate = []

        for n, layer in enumerate(self.layers):

            residual = True
            output, ws = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, residual=residual)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual=True):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, ws = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt)
        tgt2, ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)

        # attn_weights [B,NUM_Q,T]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, ws

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # # # print(q.size(), k.size(), tgt2.size())
        # tgt2,ws = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        # print('1', tgt.size(), memory.size())
        # sssss
        # tgt2 = self.norm2(tgt)
        # print(self.with_pos_embed(tgt2, query_pos).size(), self.with_pos_embed(memory, pos).size())
        memory = memory.permute(2, 0, 1).contiguous()
        # print(memory.size())
        # memory_mask = self._generate_square_subsequent_mask(memory.size(0),tgt2.size(0))
        # memory_mask = memory_mask.cuda()
        # print(memory_mask.size())
        # print(tgt2.size(),memory.size())
        # attn_output_weights = torch.bmm(tgt2,memory.transpose(1, 2))
        # print(attn_output_weights.size())
        # sss
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                 key=self.with_pos_embed(memory, pos),
                                                 value=memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        tgt2 = self.norm1(tgt2)
        # # print(tgt2.size(), memory.size())
        # tgt2,attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)
        # # print(tgt2.size())
        # # sss
        tgt2 = tgt + self.dropout2(tgt2)
        # # # print('2', tgt.size())
        # tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # # print(tgt2.size())
        # # tgt = tgt + self.dropout3(tgt2)
        # # print()
        # print(attn_weights.size())
        # ssss
        return tgt2, attn_weights

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                residual=True):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, residual)

    def _generate_square_subsequent_mask(self, ls, sz):
        mask = (torch.triu(torch.ones(ls, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torchvision
# from decoder import TransformerDecoder, TransformerDecoderLayer
# from PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
import copy
import numpy as np


class FPN(nn.Module):
    def __init__(self, num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, W = y.size()
        return F.upsample(x, size=W, mode='linear') + y

    def forward(self, out_list):
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1, p2, p3, p4]


class Hierarch_TCN2(nn.Module):

    def __init__(self, args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(Hierarch_TCN2, self).__init__()
        # self.PG = Prediction_Generation(args, num_layers_PG, num_f_maps, dim, num_classes)
        self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)

        self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.first_linear = nn.Linear(num_f_maps*4, num_f_maps, 1)
        self.conv_out1 = nn.Conv1d(num_f_maps * 3, num_classes, 1)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(args, num_layers_R, num_f_maps, num_classes, num_classes, self.conv_out)) for s in
             range(num_R)])
        self.use_fpn = args.fpn
        self.use_output = args.output
        self.use_feature = args.feature
        self.use_trans = args.trans
        # self.prototpye=[]
        if args.fpn:
            self.fpn = FPN(num_f_maps)
        if args.trans:
            self.query = nn.Embedding(num_classes, num_f_maps)

            if args.positional_encoding_type == "learned":
                self.position_encoding = LearnedPositionalEncoding(
                    19971, num_f_maps
                )
            elif args.positional_encoding_type == "fixed":
                self.position_encoding = FixedPositionalEncoding(
                    num_f_maps,
                )
            else:
                self.position_encoding = None
            print('position encoding :', args.positional_encoding_type)
            decoder_layer = TransformerDecoderLayer(num_f_maps, args.head_num, args.embed_num,
                                                    0.1, 'relu', normalize_before=True)
            decoder_norm = nn.LayerNorm(num_f_maps)
            self.decoder = TransformerDecoder(decoder_layer, args.block_num, decoder_norm,
                                              return_intermediate=False)
        self.prototpye = torch.nn.Parameter(torch.zeros(1, 64, num_classes), requires_grad=True)

    def forward(self, x):
        out_list = []
        f_list = []
        x = x.permute(0, 2, 1)

        f, out1 = self.PG(x)

        f_list.append(f)
        if not self.use_fpn:
            out_list.append(out1)

        # print(out.size())

        for R in self.Rs:
            # F.softmax(out, dim=1)
            if self.use_output:
                f, out1 = R(out1)
                out_list.append(out1)
                # print(out1.size())
            else:
                f, out1 = R(f)
            # print(f.size())
            # print(out.size())

            f_list.append(f)
            if not self.use_fpn:
                out_list.append(out1)
            # outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        # print(len(out_list))
        if self.use_fpn:
            f_list = self.fpn(f_list)
            for f in f_list:
                # print(f.size())
                out_list.append(self.conv_out(f))
        # sss
        if self.use_feature:
            last_feature = f_list[-1]
            refine_out = torch.matmul(self.prototpye.transpose(1, 2), last_feature)
            out_list[-1] = 0.5 * out_list[-1] + 0.5 * refine_out

        # print(len(f_list))

        if self.use_trans:

            for i in range(len(f_list)):
                if self.position_encoding == None:
                    f_list[i] = f_list[i]
                else:
                    # print(f_list[i].size())
                    f_list[i] = self.position_encoding(f_list[i])
            # query_embed = self.query.weight.unsqueeze(1).repeat( 1, batch_size, 1)

            # first_feature = f_list[0]
            first_feature_list = []
            first_feature_list.append(f_list[0])
            first_feature = f_list[0].permute(2, 0, 1)
            # print(len(f_list))
            # sss
            for i in range(1, len(f_list)):
                middle_feature = f_list[i]

                first_feature = self.decoder(first_feature, middle_feature,
                                             memory_key_padding_mask=None, pos=None, query_pos=None)
                # print(first_feature.size(),middle_feature.size())

                # attention_w = torch.matmul(first_feature.transpose(1,2), middle_feature)
                # attention_w = F.softmax(attention_w,dim=2)
                # new_first_feature = torch.matmul(attention_w, middle_feature.transpose(1,2))
                # print(new_first_feature.transpose().size())
                # ssss
                # first_feature_list.append(new_first_feature.transpose(1,2))
                # first_feature_list.append(new_first_feature.permute(1,2,0))
                # last_feature = f_list[-1]
                # middle_feature = f_list[-2]
                # # print(pos_embd.size())

                # # x = self.conv_out(out) # (bs, c, l)
                # # out = last_feature.permute(2,0,1)
                # first_feature = f_list[0].permute(2,0,1)
                # # print(first_feature.size(), last_feature.size())
                # first_feature = self.decoder(first_feature, last_feature,
                #     memory_key_padding_mask=None, pos=None, query_pos=None)
                # f_list[0] = first_feature.permute(1,2,0)

            # f_list[0] = torch.cat(first_feature_list,dim=1)
            # f_list[0] = torch.stack(first_feature_list,dim=1).sum(dim=1)

            # print(f_list[0].size())
            # print(f_list[1].size())
            # reduced_first_feature = self.first_linear(f_list[0].transpose(1,2)).transpose(1,2)
            # reduced_first_feature=f_list[0]
            reduced_first_feature = first_feature.permute(1, 2, 0)
            out_list[0] = self.conv_out(reduced_first_feature)
            # for idx, f in enumerate(f_list):
            #     if idx == 0:
            #         out_list.append(self.conv_out1(f))
            #     else:
            #         out_list.append(self.conv_out(f))

            # out_list[-1] = pro
        return out_list, f_list, self.prototpye


class BaseCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        print(num_layers)
        super(BaseCausalTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.channel_dropout = nn.Dropout2d()
        # self.downsample = nn.Linear(num_f_maps,num_f_maps, kernel_size=3, stride=2,dilation=3)
        # self.center = torch.nn.Parameter(torch.zeros(1, 64, num_classes), requires_grad=False)
        self.num_classes = num_classes

    def forward(self, x, labels=None, mask=None, test=False):
        # x = x.permute(0,2,1) # (bs,l,c) -> (bs, c, l)

        if mask is not None:
            # print(x.size(),mask.size())
            x = x * mask

        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)

        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)

        x = self.conv_out(out)  # (bs, c, l)

        return out, x


class Prediction_Generation(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** (num_layers - 1 - i), num_f_maps, num_f_maps))
             for i in range(num_layers)]
        ))

        # self.conv_dilated_1 = nn.ModuleList((
        #     nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
        #     for i in range(num_layers)
        # ))
        self.conv_dilated_2 = nn.ModuleList((
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps))
             for i in range(num_layers)]
        ))
        # self.conv_dilated_2 = nn.ModuleList((
        #     nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
        #     for i in range(num_layers)
        # ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout()

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return f, out


class Refinement(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes, conv_out):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.conv_out = conv_out
        self.max_pool_1x1 = nn.AvgPool1d(kernel_size=7, stride=3)
        self.use_output = args.output
        self.hier = args.hier

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x
        for layer in self.layers:
            out = layer(out)
        if self.hier:
            f = self.max_pool_1x1(out)
        else:
            f = out
        out = self.conv_out(f)

        return f, out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return x + out


class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, padding=None):
        super(DilatedResidualCausalLayer, self).__init__()
        if padding == None:

            self.padding = 2 * dilation
        else:
            self.padding = padding
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)  #
        # self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0)  # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)



# utils
from matplotlib import pyplot as plt
from matplotlib import *
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from MulticoreTSNE import MulticoreTSNE as TSNE

import seaborn as sns

phase2label_dicts = {
    'cholec80': {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6},

    'm2cai16': {
        'TrocarPlacement': 0,
        'Preparation': 1,
        'CalotTriangleDissection': 2,
        'ClippingCutting': 3,
        'GallbladderDissection': 4,
        'GallbladderPackaging': 5,
        'CleaningCoagulation': 6,
        'GallbladderRetraction': 7}
}


def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]: k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] for label in labels]
    return phases


max_pool = nn.MaxPool1d(kernel_size=13, stride=5, dilation=3)

path_p = "/home/xmli/phwang/ntfs/xinpeng/code/casual_tcn/results/m2cai16/eva/resize/"


def fusion(predicted_list, labels, args):
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0 / len_layer for i in range(0, len_layer)]
    # print(weight_list)
    num = 0
    for out, w in zip(predicted_list, weight_list):
        resize_out = F.interpolate(out, size=labels.size(0), mode='nearest')
        resize_out_list.append(resize_out)
        # align_corners=True
        # print(out.size())
        resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0), size=out.size(2), mode='linear',
                                     align_corners=False)
        if out.size(2) == labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            # resize_label = max_pool(labels_list[-1].float().unsqueeze(0).unsqueeze(0))
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0), size=out.size(2), mode='nearest')
            # resize_label2 = F.interpolate(resize_label,size=labels.size(0),mode='nearest')
            # ,align_corners=True
            # print(resize_label.size(), resize_label2.size())
            # print((resize_label2 == labels).sum()/labels.size(0))
            # with open(path_p+'{}.txt'.format(num),"w") as f:
            #     for labl1, lab2 in zip(resize_label2.squeeze(), labels.squeeze()):
            #         f.writelines(str(labl1)+'\t'+str(lab2)+'\n')
            # num+=1
            labels_list.append(resize_label.squeeze().long())
            # labels_list.append(labels.squeeze().long())
        # print(resize_label.size(), out.size())
        # labels_list.append(labels.squeeze().long())
        # assert resize_out.size(2) == resize_label.size(0)
        # assert resize_label.size(2) == out.size(2)
        # print(out.size())
        # print(resize_label.size())
        # print(resize_out.size())
        # all_out_list.append(out)
        # all_out_list.append(resize_out)

        all_out_list.append(out)
        # resize_out=out
        # all_out = all_out + w*resize_out

    # sss
    return all_out, all_out_list, labels_list


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)
    # dist = 1. - similiarity
    return similiarity


def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)

    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels)
    color_map = plt.cm.tab10

    #     axprops = dict(xticks=[], yticks=[0,0.5,1], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=15)
    fig = plt.figure(figsize=(15, (num_pics + 1) * 1.5))

    interval = 1 / (num_pics + 2)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0.1, 1 - i * interval, 0.8, interval - interval / num_pics]))
    #         ax1.imshow([label], **barprops)
    titles = ['Ground Truth', 'Causal-TCN', 'Causal-TCN + PKI', 'Causal-TCN + MS-GRU']
    for i, label in enumerate(labels):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].imshow([label], **barprops)
    #         axes[i].set_title(titles[i])

    ax99 = fig.add_axes([0.1, 0.05, 0.8, interval - interval / num_pics])
    #     ax99.set_xlim(-len(confidence_score)/15, len(confidence_score) + len(confidence_score)/15)
    ax99.set_xlim(0, len(confidence_score))
    ax99.set_ylim(-0.2, 1.2)
    ax99.set_yticks([0, 0.5, 1])
    ax99.set_xticks([])

    ax99.plot(range(len(confidence_score)), confidence_score)

    if save_path is not None:
        print(save_path)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def PKI(confidence_seq, prediction_seq, transition_prior_matrix, alpha, beta,
        gamma):  # fix the predictions that do not meet priors
    initital_phase = 0
    previous_phase = 0
    alpha_count = 0
    assert len(confidence_seq) == len(prediction_seq)
    refined_seq = []
    for i, prediction in enumerate(prediction_seq):
        if prediction == initital_phase:
            alpha_count = 0
            refined_seq.append(initital_phase)
        else:
            if prediction != previous_phase or confidence_seq[i] <= beta:
                alpha_count = 0

            if confidence_seq[i] >= beta:
                alpha_count += 1

            if transition_prior_matrix[initital_phase][prediction] == 1:
                refined_seq.append(prediction)
            else:
                refined_seq.append(initital_phase)

            if alpha_count >= alpha and transition_prior_matrix[initital_phase][prediction] == 1:
                initital_phase = prediction
                alpha_count = 0

            if alpha_count >= gamma:
                initital_phase = prediction
                alpha_count = 0
        previous_phase = prediction

    assert len(refined_seq) == len(prediction_seq)
    return refined_seq