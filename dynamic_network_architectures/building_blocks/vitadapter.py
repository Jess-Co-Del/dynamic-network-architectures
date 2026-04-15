"""
ViT-Adapter with DINOv2 Backbone
=================================
Implements the ViT-Adapter architecture that injects spatial prior features
into intermediate layers of a frozen (or fine-tunable) DINOv2 backbone and
extracts multi-scale features for dense prediction tasks.

Architecture overview (matching the paper figure):
  - Spatial Prior Module: lightweight CNN stem → multi-scale spatial features
  - Injectors: cross-attention that fuses spatial priors INTO ViT features
  - Extractors: cross-attention that pulls features OUT of ViT + FFN refinement
  - DINOv2 backbone: pretrained ViT blocks (frozen or fine-tuned)

The injectors/extractors are interleaved with the ViT blocks so information
flows bidirectionally between the spatial branch and the transformer branch.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvBNReLU
from torch.nn.init import normal_


# ---------------------------------------------------------------------------
# (c) Spatial Prior Module
# ---------------------------------------------------------------------------

class _SpatialPriorModule(nn.Module):
    """
    Lightweight CNN stem that produces multi-scale spatial features from the
    input image.  Outputs F1 (stride-4), F2 (stride-8), F3 (stride-16) and
    a flattened F_sp (stride-16, projected to *hidden_dim*).

    The multi-scale features are later used by the extractors to recover
    spatial detail at different resolutions.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 768,
                 stem_channels: int = 64):
        super().__init__()
        # Progressive downsampling: stride 2 → 4 → 8 → 16
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, stem_channels, 3, stride=2, padding=1),      # /2
            ConvBNReLU(stem_channels, stem_channels, 3, stride=1, padding=1),
            ConvBNReLU(stem_channels, stem_channels, 3, stride=1, padding=1),
        )

        self.downsample1 = ConvBNReLU(stem_channels, stem_channels * 2, 3, stride=2, padding=1)      # /4  → F1
        self.downsample2 = ConvBNReLU(stem_channels * 2, stem_channels * 4, 3, stride=2, padding=1)   # /8  → F2
        #self.downsample3 = ConvBNReLU(stem_channels * 4, stem_channels * 4, 3, stride=2, padding=1)  # /32 → F3

        # Project F3 to hidden_dim so it matches the ViT token dimension
        self.proj1 = nn.Conv2d(stem_channels, hidden_dim, 1)
        self.proj2 = nn.Conv2d(stem_channels*2, hidden_dim, 1)
        self.proj3 = nn.Conv2d(stem_channels * 4, hidden_dim, 1)
        #self.proj4 = nn.Conv2d(stem_channels * 4, hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input image [B, 3, H, W]
        Returns:
            f_sp: [B, N, hidden_dim]  — flattened spatial prior tokens (stride-16)
            multi_scale: list of [F1, F2, F3] feature maps at strides 4, 8, 16
        """
        x = self.stem(x)           # /4
        f1 = self.downsample1(x)   # /8
        f2 = self.downsample2(f1)  # /16
        #f3 = self.downsample3(f2)  # /32

        # Flatten F3 → sequence of tokens for cross-attention with ViT
        f_s0 = self.proj1(x)
        #f_s3 = self.proj4(f3)                          # [B, hidden_dim, H/8, W/8]
        f_s2 = self.proj3(f2)                          # [B, hidden_dim, H/16, W/16]
        f_s1 = self.proj2(f1)                          # [B, hidden_dim, H/32, W/32]

        f_s0 = f_s0.flatten(2).transpose(1, 2)        # [B, N, hidden_dim]
        f_s2 = f_s2.flatten(2).transpose(1, 2)        # [B, N, hidden_dim]
        f_s1 = f_s1.flatten(2).transpose(1, 2)        # [B, N, hidden_dim]

        return f_s0, f_s1, f_s2  #, f_s3


# ---------------------------------------------------------------------------
# (d) Spatial Feature Injector
# ---------------------------------------------------------------------------

class SpatialFeatureInjector(nn.Module):
    """
    Injects spatial prior information into the ViT feature stream via
    cross-attention.

        Query  = F_vit^i       (ViT tokens)
        Key    = F_sp^i        (spatial prior tokens)
        Value  = F_sp^i

    The output is element-wise added back to F_vit^i (residual).
    """

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_vit = nn.LayerNorm(hidden_dim)
        self.norm_sp = nn.LayerNorm(hidden_dim)

    def forward(self, f_vit: torch.Tensor, f_sp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_vit: [B, N_vit, C] — ViT tokens (query)
            f_sp:  [B, N_sp, C]  — spatial prior tokens (key/value)
        Returns:
            f_vit_out: [B, N_vit, C] — ViT tokens with injected spatial info
        """
        residual = f_vit
        f_vit_n = self.norm_vit(f_vit)
        f_sp_n = self.norm_sp(f_sp)

        B, N_q, C = f_vit_n.shape
        N_kv = f_sp_n.shape[1]

        q = self.q_proj(f_vit_n).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(f_sp_n).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.proj_drop(self.out_proj(out))

        return residual + out


# ---------------------------------------------------------------------------
# (e) Multi-Scale Feature Extractor
# ---------------------------------------------------------------------------

class AdapterDWConv(nn.Module):
    def __init__(self, hidden_dim, sp_scales):
        super().__init__()
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True, groups=hidden_dim)
        self.sp_scales = sp_scales

    def forward(self, x):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:self.sp_scales[0]**2, :].permute(0, 2, 1).reshape(
            B, C, self.sp_scales[0], self.sp_scales[0])
        x2 = x[:, self.sp_scales[0]**2:self.sp_scales[0]**2 + self.sp_scales[1]**2, :].permute(0, 2, 1).reshape(
            B, C, self.sp_scales[1], self.sp_scales[1])
        x3 = x[:, self.sp_scales[0]**2 + self.sp_scales[1]**2:, :].permute(0, 2, 1).reshape(
            B, C, self.sp_scales[2], self.sp_scales[2])
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts features from the ViT stream back into the spatial branch via
    cross-attention + FFN.

        Key/Value = F_vit^{i+1}   (updated ViT tokens)
        Query     = F_sp^i        (spatial prior tokens)

    Followed by a 2-layer FFN with residual connection.  The updated spatial
    tokens F_sp^{i+1} are then available for the next injector.
    """

    def __init__(
        self,
        hidden_dim: int = 1024, num_heads: int = 8, dwconv_dim_ratio: float = 0.25,
        sp_scales: List[int] = [112, 56, 28], qkv_bias: bool = True,
        attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention: spatial queries, ViT keys/values
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_sp = nn.LayerNorm(hidden_dim)
        self.norm_vit = nn.LayerNorm(hidden_dim)

        # FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * dwconv_dim_ratio)),
            nn.GELU(),
            AdapterDWConv(int(hidden_dim * dwconv_dim_ratio), sp_scales),
            nn.Dropout(proj_drop),
            nn.Linear(int(hidden_dim * dwconv_dim_ratio), hidden_dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, f_sp: torch.Tensor, f_vit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_sp:  [B, N_sp, C]  — spatial tokens (query)
            f_vit: [B, N_vit, C] — ViT tokens (key/value)
        Returns:
            f_sp_out: [B, N_sp, C] — updated spatial tokens
        """
        # --- Cross-Attention ---
        residual = f_sp
        f_sp_n = self.norm_sp(f_sp)
        f_vit_n = self.norm_vit(f_vit)

        B, N_q, C = f_sp_n.shape
        N_kv = f_vit_n.shape[1]

        q = self.q_proj(f_sp_n).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(f_vit_n).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.proj_drop(self.out_proj(out))
        f_sp = residual + out

        # --- FFN ---
        f_sp = f_sp + self.ffn(self.norm_ffn(f_sp))

        return f_sp


# ---------------------------------------------------------------------------
# DINOv2 backbone wrapper
# ---------------------------------------------------------------------------

class DINOv2Backbone(nn.Module):
    """
    Wraps a DINOv2 model to expose its intermediate blocks for interleaving
    with injectors/extractors.

    Supports grouping the N transformer blocks into `num_interactions` groups
    so that each group is followed by an extractor/injector pair.
    """

    def __init__(self, model_name: str = "dinov2_vitb14",
                 pretrained: bool = True, freeze: bool = False):
        super().__init__()

        # Load DINOv2 via torch.hub
        if pretrained:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name)
        else:
            # For offline / custom init you'd replace this
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name,
                                         pretrained=False)

        if freeze:
            for p in self.dinov2.parameters():
                p.requires_grad = False

        self.hidden_dim = self.dinov2.hidden_dim
        self.patch_size = self.dinov2.patch_size
        self.num_blocks = len(self.dinov2.blocks)

    def patch_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Run the DINOv2 patch embedding + position embedding.
        Returns:
            tokens: [B, 1+N, C]  (CLS + patch tokens, with pos embed added)
            H_patches, W_patches: spatial grid dimensions
        """
        B, C, H, W = x.shape
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size

        # DINOv2 patch embed
        x = self.dinov2.patch_embed(x)  # [B, N, C]

        # Prepend CLS token
        cls_token = self.dinov2.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+N, C]

        # Add position embeddings (interpolated if needed)
        x = x + self._interpolate_pos_embed(x, H_patches, W_patches)

        return x, H_patches, W_patches

    def _interpolate_pos_embed(
        self, x: torch.Tensor,
        H_patches: int, W_patches: int) -> torch.Tensor:
        """Interpolate DINOv2 position embeddings to match input resolution."""
        pos_embed = self.dinov2.pos_embed  # [1, 1+N_train, C]
        N_train = pos_embed.shape[1] - 1
        N_cur = H_patches * W_patches

        if N_cur == N_train:
            return pos_embed

        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]

        dim = patch_pos.shape[-1]
        h0 = int(math.sqrt(N_train))
        w0 = h0  # DINOv2 trains on square images

        patch_pos = patch_pos.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(H_patches, W_patches),
                                  mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)


# ---------------------------------------------------------------------------
# Full ViT-Adapter with DINOv2
# ---------------------------------------------------------------------------

from dynamic_network_architectures.building_blocks.ms_deform_attn import MSDeformAttn
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/drop.py

from torch import nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):

            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        with_cp=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(query, feat):

            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
            )
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        for idx, blk in enumerate(blocks):
            x = blk(x, H_toks, W_toks)
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H_c,
                    W=W_c,
                )
        return x, c


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            n_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks):
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        cls, x = (
            x[
                :,
                :1,
            ],
            x[
                :,
                1:,
            ],
        )
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H_c,
                    W=W_c,
                )
        return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(in_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * in_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * in_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * in_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

class ViTAdapterDINOv2(nn.Module):
    """
    Full ViT-Adapter built on top of DINOv2.

    Architecture:
        Image
          ├─→ DINOv2 patch embed ──→ [Block group 1] ──→ ... ──→ [Block group K]
          │         ↑ inject            ↑ inject                       │ extract
          └─→ Spatial Prior Module ─→ Injector 1 ─→ Extractor 1 ─→ ... ─→ Extractor K
                                                                         │
                                                            Multi-scale feature maps
                                                         (for detection / segmentation)
    # export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
    Args:
        model_name: DINOv2 hub name (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
        pretrained: whether to load pretrained DINOv2 weights
        freeze_backbone: freeze the DINOv2 backbone weights
        num_interactions: how many injector/extractor pairs (splits blocks into groups)
        num_heads: attention heads in injectors/extractors
        mlp_ratio: FFN expansion ratio in extractors
        out_channels: channel dim of each output feature map (for FPN-style heads)
        out_indices: which extractor outputs to return (0-indexed). By default all.
    """

    def __init__(
        self,
        backbone: nn.Module,
        image_size: int,
        in_channels: int = 3,
        add_vit_feature: bool = True,
        num_interactions: int = 4,
        num_heads: int = 12,
        use_cls: bool = True,
        out_channels: int = 256,
        norm_layer: nn.Module = None,
        out_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        # --- Backbone ---
        self.backbone = backbone
        hidden_dim = self.backbone.hidden_dim
        self.patch_size = self.backbone.patch_size
        self.num_interactions = num_interactions
        interaction_indexes = self.backbone.layer_indices
        self.add_vit_feature = add_vit_feature
        self.image_size = image_size
        self.H_toks, self.W_toks = image_size // self.patch_size, image_size // self.patch_size
        self.use_cls = use_cls
        n_points = 4
        use_extra_extractor = True
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Split backbone blocks into groups
        self.blocks = nn.ModuleList(
            self.backbone.get_block_groups()
        )

        # --- Spatial Prior Module ---
        self.sp_embed = nn.Parameter(torch.zeros(3, hidden_dim))
        normal_(self.sp_embed)
        self.spm = SpatialPriorModule(
            in_channels=in_channels, hidden_dim=hidden_dim
        )
        
        # Record spatial sizes for use to reshape later
        block_fn = InteractionBlockWithCls if use_cls else InteractionBlock
        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    init_values=0.00,
                    drop_path=0.0,
                    norm_layer=self.norm_layer,
                    with_cffn=True,
                    cffn_ratio=0.25,
                    deform_ratio=1.0,
                    extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                    with_cp=False,
                )
                for i in range(len(interaction_indexes))
            ]
        )

        # --- Output projections: reshape spatial tokens → 2D feature maps ---
        # Produce multi-scale outputs at strides 4, 8, 16, 32 (like an FPN)
        self.out_indices = out_indices or list(range(num_interactions))
        
        # uPSAMPLE LAST MULTI-scale
        self.up = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(hidden_dim)
        self.norm2 = nn.SyncBatchNorm(hidden_dim)
        self.norm3 = nn.SyncBatchNorm(hidden_dim)
        self.norm4 = nn.SyncBatchNorm(hidden_dim)
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.sp_embed[0]
        c3 = c3 + self.sp_embed[1]
        c4 = c4 + self.sp_embed[2]
        return c2, c3, c4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: input image [B, 3, H, W]  (H, W must be divisible by patch_size)
        Returns:
            List of multi-scale feature maps:
              [scale_4: B×C×H/4×W/4,
               scale_8: B×C×H/8×W/8,
               scale_16: B×C×H/16×W/16,
               scale_32: B×C×H/32×W/32]
        """
        B, _, H, W = x.shape
        # 1) Package DINOv2 applies preprocessing
        if callable(self.backbone._preprocess):
            x = self.backbone._preprocess(x)
        H_toks, W_toks = self.H_toks, self.W_toks
        H_c, W_c = H // 16, W // 16
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # 1) DINOv2 patch + position embedding + 1st blocks
        x = self.backbone.backbone.embeddings(x)  # [B, 1+N, D]

        # print("H_toks, W_toks =", H_toks, W_toks)
        bs, n, dim = x.shape

        # Interaction
        cls, x = (x[:, :1, ], x[:, 1:, ])
        outs = list()
        for i, layer in enumerate(self.interactions):
            if self.use_cls:
                x, c, cls = layer(
                    x,
                    c,
                    cls,
                    self.blocks[i],
                    deform_inputs1,
                    deform_inputs2,
                    H_c,
                    W_c,
                    H_toks,
                    W_toks,
                )
            else:
                x, c = layer(
                    x,
                    c,
                    self.blocks[i],
                    deform_inputs1,
                    deform_inputs2,
                    H_c,
                    W_c,
                    H_toks,
                    W_toks,
                )
            outs.append(
                x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous() \
                    if i < self.num_interactions-1 else \
                self.backbone.backbone.layernorm(x).transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous()
                )
        # # Run first ViT blocks group
        # f_vit = self.block_groups[0](f_vit)
        # patch_tokens= f_vit[:, 1:, :]  # (B, seq_len - 1, D)

        # # 2) Prepare Spatial Pooling pyramid
        # D = f_vit.size(-1)
        # f_s0_base, f_s1, f_s2 = self.spatial_prior(x)

        # # Add learnable level embeddings and concatenate into one sequence
        # f_s1 = f_s1 + self.sp_embed[0]
        # f_s2 = f_s2 + self.sp_embed[1]
        # f_s0 = f_s0_base + self.sp_embed[2]
        # #print([f.shape for f in [f_s0, f_s1, f_s2]])
        # f_sp = torch.cat([f_s0, f_s1, f_s2], dim=1)   # (B, N_8+N_16+N_32, D)

        # # 3) Interleaved injection / ViT blocks / extraction
        # outs = list()
        # outs.append(
        #         f_vit[:, 1:, :].permute(0, 2, 1).reshape(B, -1, self.H_toks, self.W_toks)
        # )
        # for i in range(1, self.num_interactions):

        #     # Extract features from ViT back to spatial branch
        #     #print(f_sp.shape)
        #     f_sp_i = self.extractors[i](f_sp, patch_tokens)  # skip CLS for extraction
        #     f_sp = f_sp + f_sp_i

        #     # Run ViT block group
        #     f_vit = self.block_groups[i](f_vit)

        #     # Inject spatial features into ViT tokens
        #     # We only inject into the patch tokens (skip CLS at index 0)
        #     cls_token = f_vit[:, :1, :]
        #     patch_tokens= f_vit[:, 1:, :]  # (B, seq_len - 1, D)
        #     patch_tokens = self.injectors[i](patch_tokens, f_sp)

        #     f_vit = torch.cat([cls_token, patch_tokens], dim=1)
        #     #f_vit = f_vit + f_vit_i

        #     outs.append(
        #         f_vit[:, 1:, :].permute(0, 2, 1).reshape(B, -1, self.H_toks, self.W_toks) \
        #             if i < self.num_interactions-1 else \
        #         self.backbone.backbone.layernorm(f_vit)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, self.H_toks, self.W_toks)
        #     )

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            # print(c1.shape, c2.shape, c3.shape, c4.shape, x1.shape, x2.shape, x3.shape, x4.shape, H_c, H_toks)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]



# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

# from dynamic_network_architectures.architectures.dinosegmentor import DINOv2FeatureExtractor
# backbone = DINOv2FeatureExtractor(
#         layer_indices=[6, 13, 19, 23],
#         adapter="all",
#         )

def vit_adapter_dinov2_small(**kwargs) -> ViTAdapterDINOv2:
    """ViT-Adapter with DINOv2-S/14 backbone (384-dim, 6 heads)."""
    defaults = dict(backbone=backbone, num_heads=6)
    defaults.update(kwargs)
    return ViTAdapterDINOv2(**defaults)


def vit_adapter_dinov2_base(**kwargs) -> ViTAdapterDINOv2:
    """ViT-Adapter with DINOv2-B/14 backbone (768-dim, 12 heads)."""
    defaults = dict(model_name="dinov2_vitb14", num_heads=12)
    defaults.update(kwargs)
    return ViTAdapterDINOv2(**defaults)


def vit_adapter_dinov2_large(**kwargs) -> ViTAdapterDINOv2:
    """ViT-Adapter with DINOv2-L/14 backbone (1024-dim, 16 heads)."""
    defaults = dict(backbone=backbone, num_heads=16)
    defaults.update(kwargs)
    return ViTAdapterDINOv2(**defaults)


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build model (use small for quick testing)
    print("Building ViT-Adapter with DINOv2-S/14 ...")
    model = vit_adapter_dinov2_large(
        num_interactions=4,
        out_channels=256,
        image_size=224
    ).to(device)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total / 1e6:.1f}M")
    print(f"Trainable params: {trainable / 1e6:.1f}M")

    # Dummy forward pass (DINOv2 patch_size=14, use 224×224)
    dummy = torch.randn(2, 3, 224, 224).to(device)  # 224 = 14 * 16
    print(f"\nInput shape: {dummy.shape}")

    with torch.no_grad():
        outputs = model(dummy)

    print("\nMulti-scale output shapes:")
    for i, feat in enumerate(outputs):
        stride = [4, 8, 16, 32][i]
        print(f"  stride-{stride:2d}: {feat.shape}")

    spm = SpatialPriorModule(in_channels=3, hidden_dim=256).to(device)
    with torch.no_grad():
        multiscales = spm(dummy)

    print("\nSpatial prior Module output shapes:")
    print(f"  SPM-{[feat.shape for feat in multiscales]}")
    print("\nDone!")
