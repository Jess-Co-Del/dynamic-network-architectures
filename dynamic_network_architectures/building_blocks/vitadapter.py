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
from simple_conv_blocks import ConvBNReLU
from torch.nn.init import normal_


# ---------------------------------------------------------------------------
# (c) Spatial Prior Module
# ---------------------------------------------------------------------------

class SpatialPriorModule(nn.Module):
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                     # /2
        )
        self.downsample1 = ConvBNReLU(stem_channels, stem_channels * 2, 3, stride=2, padding=1)      # /8  → F1
        self.downsample2 = ConvBNReLU(stem_channels * 2, stem_channels * 4, 3, stride=2, padding=1)   # /16  → F2
        self.downsample3 = ConvBNReLU(stem_channels * 4, stem_channels * 4, 3, stride=2, padding=1)  # /32 → F3

        # Project F3 to hidden_dim so it matches the ViT token dimension
        self.proj1 = nn.Conv2d(stem_channels, hidden_dim, 1)
        self.proj2 = nn.Conv2d(stem_channels*2, hidden_dim, 1)
        self.proj3 = nn.Conv2d(stem_channels * 4, hidden_dim, 1)
        self.proj4 = nn.Conv2d(stem_channels * 4, hidden_dim, 1)

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
        f3 = self.downsample3(f2)  # /32

        # Flatten F3 → sequence of tokens for cross-attention with ViT
        f_s0 = self.proj1(x)
        f_s3 = self.proj4(f3)                          # [B, hidden_dim, H/8, W/8]
        f_s2 = self.proj3(f2)                          # [B, hidden_dim, H/16, W/16]
        f_s1 = self.proj2(f1)                          # [B, hidden_dim, H/32, W/32]

        f_s3 = f_s3.flatten(2).transpose(1, 2)        # [B, N, hidden_dim]
        f_s2 = f_s2.flatten(2).transpose(1, 2)        # [B, N, hidden_dim]
        f_s1 = f_s1.flatten(2).transpose(1, 2)        # [B, N, hidden_dim]

        return f_s0, f_s1, f_s2, f_s3


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
        hidden_dim: int = 768, num_heads: int = 8,
        mlp_ratio: float = 4.0, qkv_bias: bool = True,
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, hidden_dim),
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
        in_channels: int = 3,
        add_vit_feature: bool = True,
        num_interactions: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 256,
        out_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        # --- Backbone ---
        self.backbone = backbone
        hidden_dim = self.backbone.hidden_dim
        self.patch_size = self.backbone.patch_size
        self.num_interactions = num_interactions
        self.add_vit_feature = add_vit_feature
        # Split backbone blocks into groups
        self.block_groups = nn.ModuleList(
            self.backbone.get_block_groups(num_interactions)
        )

        # --- Spatial Prior Module ---
        self.sp_embed = nn.Parameter(torch.zeros(3, hidden_dim))
        normal_(self.sp_embed)
        self.spatial_prior = SpatialPriorModule(
            in_channels=in_channels, hidden_dim=hidden_dim, stem_channels=out_channels
        )

        # --- Injectors & Extractors (one per interaction) ---
        self.injectors = nn.ModuleList([
            SpatialFeatureInjector(hidden_dim, num_heads)
            for _ in range(num_interactions)
        ])
        self.extractors = nn.ModuleList([
            MultiScaleFeatureExtractor(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_interactions)
        ])

        # --- Output projections: reshape spatial tokens → 2D feature maps ---
        # Produce multi-scale outputs at strides 4, 8, 16, 32 (like an FPN)
        self.out_indices = out_indices or list(range(num_interactions))
        
        # uPSAMPLE LAST MULTI-scale
        self.upsample_sp0 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(hidden_dim)
        self.norm2 = nn.SyncBatchNorm(hidden_dim)
        self.norm3 = nn.SyncBatchNorm(hidden_dim)
        self.norm4 = nn.SyncBatchNorm(hidden_dim)
        # We reshape the final f_sp back to 2D at stride-16, then use
        # convolutions to produce 4 scales (stride 4, 8, 16, 32).
        # self.output_proj = nn.ModuleDict({
        #     "scale_4": nn.Sequential(
        #         nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=4),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #     ),
        #     "scale_8": nn.Sequential(
        #         nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=2, stride=2),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #     ),
        #     "scale_16": nn.Sequential(
        #         nn.Conv2d(hidden_dim, out_channels, 1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #     ),
        #     "scale_32": nn.Sequential(
        #         nn.Conv2d(hidden_dim, out_channels, kernel_size=2, stride=2),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #     ),
        # })

        # Lateral connections to fuse CNN multi-scale features (F1, F2, F3)
        # with the reshaped spatial prior tokens
        # self.lateral_f1 = nn.Conv2d(64, out_channels, 1)    # stride-4
        # self.lateral_f2 = nn.Conv2d(128, out_channels, 1)   # stride-8
        # self.lateral_f3 = nn.Conv2d(256, out_channels, 1)   # stride-16

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
        H_toks, W_toks = H // self.patch_size, W // self.patch_size

        # 1) DINOv2 patch + position embedding
        f_vit = self.backbone.backbone.embeddings(x)  # [B, 1+N, D]
        D = f_vit.size(-1)
        f_s0, f_s1, f_s2, f_s3 = self.spatial_prior(x)
 
        # Record spatial sizes for use in reshape later (BUG 1 FIX: derived
        # from the actual feature map size, not from patch_size constants)
        H4, W4  = H // 4,  W // 4    # stride-4
        H8, W8  = H // 8,  W // 8    # stride-8
        H16, W16 = H // 16, W // 16   # stride-16
        H32, W32 = H // 32, W // 32   # stride-32
 
        # Add learnable level embeddings and concatenate into one sequence
        f_s1 = f_s1 + self.sp_embed[0]
        f_s2 = f_s2 + self.sp_embed[1]
        f_s3 = f_s3 + self.sp_embed[2]
        f_sp = torch.cat([f_s1, f_s2, f_s3], dim=1)   # (B, N_8+N_16+N_32, D)
 

        # 3) Interleaved injection / ViT blocks / extraction
        outs = list()
        for i in range(self.num_interactions):
            # Inject spatial features into ViT tokens
            # We only inject into the patch tokens (skip CLS at index 0)
            cls_token = f_vit[:, :1, :]
            patch_tokens= f_vit[:, 1:, :]  # (B, seq_len - 1, D)

            patch_tokens = self.injectors[i](patch_tokens, f_sp)

            f_vit = torch.cat([cls_token, patch_tokens], dim=1)

            # Run ViT block group
            f_vit = self.block_groups[i](f_vit)

            # Extract features from ViT back to spatial branch
            f_sp = self.extractors[i](f_sp, f_vit[:, 1:, :])  # skip CLS for extraction
            outs.append(f_vit[:, 1:, :].transpose(1, 2).view(B, D, H_toks, W_toks).contiguous())

        # 4) Reshape f_sp back to 2D spatial feature maps
        extracted_sp2 = f_sp[:, 0: f_s1.size(1), :]
        extracted_sp3 = f_sp[:, f_s1.size(1): f_s1.size(1) + f_s2.size(1), :]
        extracted_sp4 = f_sp[:, f_s1.size(1) + f_s2.size(1):, :]

        sp_8 = extracted_sp2.transpose(1, 2).view(
            B, D, H8, W8).contiguous()
        sp_16 = extracted_sp3.transpose(1, 2).view(
            B, D, H16,W16).contiguous()
        sp_32 = extracted_sp4.transpose(1, 2).view(
            B, D, H32, W32).contiguous()  # [B, C, H/ps, W/ps]
        sp_4 = self.upsample_sp0(sp_8) + f_s0

        # 5) Produce multi-scale outputs + fuse with CNN lateral features  (stride-14)
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            sp_4 += F.interpolate(x1, size=(H4, W4), mode='bilinear', align_corners=False)
            sp_8 = sp_8 + F.interpolate(x2, size=(H8, W8), mode='bilinear', align_corners=False)
            sp_16 = sp_16 + F.interpolate(x3, size=(H16, W16), mode='bilinear', align_corners=False)
            sp_32 = sp_32 + F.interpolate(x4, size=(H32, W32), mode='bilinear', align_corners=False)

        # Final Norm
        f1 = self.norm1(sp_4)
        f2 = self.norm2(sp_8)
        f3 = self.norm3(sp_16)
        f4 = self.norm4(sp_32)
        return [f1, f2, f3, f4]

        # out_s4 = self.output_proj["scale_4"](f_sp_2d) + self.lateral_f1(f1)
        # out_s8 = self.output_proj["scale_8"](f_sp_2d) + self.lateral_f2(f2)
        # out_s16 = self.output_proj["scale_16"](f_sp_2d) + self.lateral_f3(f3)
        # out_s32 = self.output_proj["scale_32"](f_sp_2d)

        # return [out_s4, out_s8, out_s16, out_s32]


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

from dynamic_network_architectures.architectures.dinosegmentor import DINOv2FeatureExtractor

backbone = DINOv2FeatureExtractor(
        layer_indices=[6, 13, 19, 23],
        adapter="all",
        )

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
