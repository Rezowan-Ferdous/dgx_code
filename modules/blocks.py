# import torch.nn as nn
# from timm.models.layers import drop_path, to_2tuple, trunc_normal_
# from modules.attention import Attention_Spatial,Attention_Temporal
# import torch.nn as nn
# from einops import rearrange
# import torch
#
# class Mlp(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         hidden_features=None,
#         out_features=None,
#         act_layer=nn.GELU,
#         drop=0.0,
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#     def extra_repr(self) -> str:
#         return "p={}".format(self.drop_prob)
#
#
# class SurgFormer_block(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.2,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention_Spatial(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#
#         ## Temporal Attention Parameters
#         self.temporal_norm1 = norm_layer(dim)
#         self.temporal_attn = Attention_Temporal(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         self.temporal_fc = nn.Linear(dim, dim)
#
#         ## drop path
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )
#         # self.mlp_cls = nn.Linear(dim, dim)
#         self.norm_cls = norm_layer(dim)
#
#     def forward(self, x, B, T, K):
#         # 如果alpha以及beta初始化为0，则xs、xt初始化为0, 在训练过程中降低了学习难度；
#         # 仿照其余模型可以使用alpha.sigmoid()以及beta.sigmoid()；
#         B, M, C = x.shape
#         assert T * K + 1 == M
#
#         # Temporal_Self_Attention
#         xt = x[:, 1:, :]
#         xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
#
#         res_temporal = self.drop_path(
#             self.temporal_attn.forward(self.temporal_norm1(xt), B)
#         )
#
#         res_temporal = rearrange(
#                 res_temporal, "(b k) t c -> b (k t) c", b=B
#             )  # 通过FC时需要将时空tokens合并，再通过残差连接连接输入特征
#         xt = self.temporal_fc(res_temporal) + x[:, 1:, :]
#
#         # Spatial_Self_Attention
#         init_cls_token = x[:, 0, :].unsqueeze(1)  # B, 1, C
#         cls_token = init_cls_token.repeat(1, T, 1)  # B, T, C
#         cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
#         xs = xt
#         xs = rearrange(xs, "b (k t) c -> (b t) k c", t=T)
#
#         xs = torch.cat((cls_token, xs), 1)  # BT, K+1, C
#         res_spatial = self.drop_path(self.attn.forward(self.norm1(xs), B))
#
#         ### Taking care of CLS token
#         cls_token = res_spatial[:, 0, :]  # BT, C 表示了在每帧单独学习的class token
#         cls_token = rearrange(cls_token, "(b t) c -> b t c", b=B, t=T)
#         cls_token = self.norm_cls(cls_token)
#         target_token = cls_token[:, -1, :].unsqueeze(1)
#         attn = (target_token @ cls_token.transpose(-1, -2))
#         attn = attn.softmax(dim=-1)
#         cls_token = (attn @ cls_token)
#
#         # cls_token = torch.mean(cls_token, 1, True)  # 通过在全局帧上平均来建立时序关联（适用于视频分类任务）
#         res_spatial = res_spatial[:, 1:, ]  # BT, xK, C
#         res_spatial = rearrange(
#             res_spatial, "(b t) k c -> b (k t) c", b=B)
#         res = res_spatial
#         x = xt
#         ## Mlp
#         x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP学习时序对应的cls_token?
#
#         return x
#
# class Block(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.2,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.norm1 = norm_layer(dim)
#         # self.attn = Attention_Spatial(
#         #     dim,
#         #     num_heads=num_heads,
#         #     qkv_bias=qkv_bias,
#         #     qk_scale=qk_scale,
#         #     attn_drop=attn_drop,
#         #     proj_drop=drop,
#         # )
#
#         ## Temporal Attention Parameters
#         self.temporal_norm1 = norm_layer(dim)
#         self.temporal_attn = Attention_Temporal(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         self.temporal_fc = nn.Linear(dim, dim)
#
#         ## drop path
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )
#         # self.mlp_cls = nn.Linear(dim, dim)
#         self.norm_cls = norm_layer(dim)
#
#     def forward(self, x, B, T, K):
#         # 如果alpha以及beta初始化为0，则xs、xt初始化为0, 在训练过程中降低了学习难度；
#         # 仿照其余模型可以使用alpha.sigmoid()以及beta.sigmoid()；
#         B, M, C = x.shape
#         assert T * K + 1 == M
#
#         # Temporal_Self_Attention
#         xt = x[:, 1:, :]
#         xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
#
#         res_temporal = self.drop_path(
#             self.temporal_attn.forward(self.temporal_norm1(xt), B)
#         )
#
#         res_temporal = rearrange(
#                 res_temporal, "(b k) t c -> b (k t) c", b=B
#             )  # 通过FC时需要将时空tokens合并，再通过残差连接连接输入特征
#         xt = self.temporal_fc(res_temporal) + x[:, 1:, :]
#
#         # Spatial_Self_Attention
#         init_cls_token = x[:, 0, :].unsqueeze(1)  # B, 1, C
#         cls_token = init_cls_token.repeat(1, T, 1)  # B, T, C
#         cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
#         xs = xt
#         xs = rearrange(xs, "b (k t) c -> (b t) k c", t=T)
#
#         # xs = torch.cat((cls_token, xs), 1)  # BT, K+1, C
#         # res_spatial = self.drop_path(self.attn.forward(self.norm1(xs), B))
#
#         ### Taking care of CLS token
#         # cls_token = res_spatial[:, 0, :]  # BT, C 表示了在每帧单独学习的class token
#         # cls_token = rearrange(cls_token, "(b t) c -> b t c", b=B, t=T)
#         # cls_token = self.norm_cls(cls_token)
#         # target_token = cls_token[:, -1, :].unsqueeze(1)
#         # attn = (target_token @ cls_token.transpose(-1, -2))
#         # attn = attn.softmax(dim=-1)
#         # cls_token = (attn @ cls_token)
#         #
#         # # cls_token = torch.mean(cls_token, 1, True)  # 通过在全局帧上平均来建立时序关联（适用于视频分类任务）
#         # res_spatial = res_spatial[:, 1:, ]  # BT, xK, C
#         # res_spatial = rearrange(
#         #     res_spatial, "(b t) k c -> b (k t) c", b=B)
#         # res = res_spatial
#         x = xt
#         ## Mlp
#         # x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP学习时序对应的cls_token?
#
#         return x
