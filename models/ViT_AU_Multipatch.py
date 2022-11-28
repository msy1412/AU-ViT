from enum import auto
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from .registry import register_model
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',AU_num_classes=21,supervise_layer=10,num_AU_patch=7):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1

        self.AU_num_classes=AU_num_classes
        self.supervise_layer = supervise_layer
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.num_AU_patch=num_AU_patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_1 = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.supervise_layer)])
        self.blocks_2=nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-self.supervise_layer)])
        self.norm = norm_layer(embed_dim)
        if num_AU_patch==7:
            self.gap_AU11 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU12 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU13 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU21 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU22 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU23 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU11 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU12 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU13 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU21 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU22 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU23 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU3  = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head_AU11= nn.Linear(embed_dim, 4)
            self.head_AU12= nn.Linear(embed_dim, 1)
            self.head_AU13= nn.Linear(embed_dim, 4)
            self.head_AU21= nn.Linear(embed_dim, 1)
            self.head_AU22= nn.Linear(embed_dim, 1)
            self.head_AU23= nn.Linear(embed_dim, 1)
            self.head_AU3= nn.Linear(embed_dim, 14)
        elif num_AU_patch==5:
            self.gap_AU1 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU21 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU22 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU23 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU1 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU21 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU22 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU23 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU3  = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head_AU1= nn.Linear(embed_dim, 5)
            self.head_AU21= nn.Linear(embed_dim, 1)
            self.head_AU22= nn.Linear(embed_dim, 1)
            self.head_AU23= nn.Linear(embed_dim, 1)
            self.head_AU3= nn.Linear(embed_dim, 14)
        elif num_AU_patch==3:
            self.gap_AU1 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU2 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU1 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU2 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.norm_AU3 = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head_AU1= nn.Linear(embed_dim, 5)
            self.head_AU2= nn.Linear(embed_dim, 2)
            self.head_AU3= nn.Linear(embed_dim, 14)
        elif num_AU_patch==1:
            self.gap_AU = nn.AdaptiveAvgPool2d(1)
            self.norm_AU = nn.LayerNorm(embed_dim, eps=1e-6)
            self.AU_head= nn.Linear(embed_dim, 21)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
    
    def AUBranch(self,AU_x):
        B, N, C = AU_x.shape
        AU_x = AU_x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2)
        # AU_x = rearrange(AU_x, 'b (h w) c -> b c h w', h=H, w=W)
        if self.num_AU_patch==7:
            AU11=AU_x[:,:, :7 , :7 ]
            AU12=AU_x[:,:, :7 ,4:10]
            AU13=AU_x[:,:, :7 ,7:  ]
            AU21=AU_x[:,:,5:12, :6 ]
            AU22=AU_x[:,:,4:10,4:10]
            AU23=AU_x[:,:,5:12,8:  ]
            AU3 =AU_x[:,:,6:  , :  ]
            AU11=self.head_AU11(self.norm_AU11(self.gap_AU11(AU11).squeeze()))
            AU12=self.head_AU12(self.norm_AU12(self.gap_AU12(AU12).squeeze()))
            AU13=self.head_AU13(self.norm_AU13(self.gap_AU13(AU13).squeeze()))
            AU21=self.head_AU21(self.norm_AU21(self.gap_AU21(AU21).squeeze()))
            AU22=self.head_AU22(self.norm_AU22(self.gap_AU22(AU22).squeeze()))
            AU23=self.head_AU23(self.norm_AU23(self.gap_AU23(AU23).squeeze()))
            AU3 =self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU1257=torch.maximum(AU11,AU13)
            AU6=torch.maximum(AU21,AU23)
            AU_all=torch.cat((AU1257[:,:2],AU12,AU1257[:,2].view(B,-1),AU6,AU1257[:,3].view(B,-1),AU22,AU3),dim=1)
        elif self.num_AU_patch==5:
            AU1 =AU_x[:,:, :7 , :  ]
            AU21=AU_x[:,:,5:12, :6 ]
            AU22=AU_x[:,:,4:10,4:10]
            AU23=AU_x[:,:,5:12,8:  ]
            AU3 =AU_x[:,:,6:  , :  ]
            AU1=self.head_AU1(self.norm_AU1(self.gap_AU1(AU1).squeeze()))
            AU21=self.head_AU21(self.norm_AU21(self.gap_AU21(AU21).squeeze()))
            AU22=self.head_AU22(self.norm_AU22(self.gap_AU22(AU22).squeeze()))
            AU23=self.head_AU23(self.norm_AU23(self.gap_AU23(AU23).squeeze()))
            AU3 =self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU6=torch.maximum(AU21,AU23)
            AU_all=torch.cat((AU1[:,:4],AU6,AU1[:,4].view(B,-1),AU22,AU3),dim=1)
        elif self.num_AU_patch==3:
            AU1=AU_x[:,:, :7 ,: ]
            AU2=AU_x[:,:,4:12,: ]
            AU3=AU_x[:,:,6:  ,: ]
            AU1=self.head_AU1(self.norm_AU1(self.gap_AU1(AU1).squeeze()))
            AU2=self.head_AU2(self.norm_AU2(self.gap_AU2(AU2).squeeze()))
            AU3=self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU_all=torch.cat((AU1[:,:4],AU2[:,0].view(B,-1),AU1[:,4].view(B,-1),AU2[:,1].view(B,-1),AU3),dim=1)
        elif self.num_AU_patch==1:
            AU_all=self.AU_head(self.norm_AU(self.gap_AU(AU_x).squeeze()))
        return AU_all

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks_1(x)
        AU_x=x[:, 1:]
        AU_output = self.AUBranch(AU_x)
        x = self.blocks_2(x)
        x = self.norm(x)

        return x[:, 0], AU_output

    def forward(self, x):
        FER_cls,AU_output = self.forward_features(x)
        FER_cls = self.head(FER_cls)

        return FER_cls,AU_output

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)