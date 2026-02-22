"""
Architecture 1 — Linear Probe / Simple MLP Head
================================================
The simplest possible segmentation head on top of I-JEPA.
Freezes the backbone entirely and trains only a 1×1 conv + bilinear upsample.

Use this for:
  - Benchmarking how linearly separable I-JEPA features are
  - Low-data regimes (few labeled images available)
  - Fast prototyping / sanity checking the pipeline

Limitations:
  - Bilinear upsample produces blurry boundaries
  - Single scale (14×14) — no high-frequency spatial detail
  - Cannot do instance or panoptic segmentation

Dependencies:
    pip install torch torchvision transformers

Reference:
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023.
    arXiv:2301.08243
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Dict, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class IJEPALinearSegHead(nn.Module):
    """
    I-JEPA backbone + linear segmentation head.

    Forward pass output shape: [B, num_classes, image_size, image_size]

    Args:
        input_channels:   Number of input channels.
        num_classes:   Number of semantic classes (e.g. 150 for ADE20K).
        image_size:    Input image spatial resolution (assumed square).
        patch_size:    ViT patch size (16 for vitg16).
        freeze_backbone: If True, no gradients flow into I-JEPA encoder.
        model_id:      HuggingFace model identifier.
    """

    def __init__(
        self,
        input_channels:   int = 1,
        num_classes:      int  = 150,
        image_size:       int  = 224,
        patch_size:       int  = 16,
        freeze_backbone:  bool = True,
        model_id:         str  = "facebook/ijepa_vitg16_22k",
        deep_supervision: bool = False
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size  = image_size // patch_size   # 14 for 224/16

        # ── Backbone (ViT-G, hidden_size = 1408) ─────────────────────────
        self.backbone = AutoModel.from_pretrained(model_id)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 1408 for ViT-G

        # ── Segmentation head ─────────────────────────────────────────────
        # 1×1 conv: projects [B, 1408, 14, 14] → [B, num_classes, 14, 14]
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 4, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: [B, 3, image_size, image_size]

        Returns:
            logits: [B, num_classes, image_size, image_size]
        """
        B = pixel_values.shape[0]
        pixel_values = torch.tile(pixel_values, (1,3,1,1)) # [B, 3, H, W], H=W=224
        # ── 1. Run I-JEPA encoder ─────────────────────────────────────────
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: [B, 196, 1408]  (197 = 196 patches)

        # ── 2. Reshape to spatial grid ───────────────────
        patch_tokens = outputs.last_hidden_state  # [:, 1:, :]   # [B, 196, 1408]
        x = (patch_tokens
             .permute(0, 2, 1)                                   # [B, 1408, 196]
             .reshape(B, -1, self.grid_size, self.grid_size)) # [B, 1408, 14, 14]

        # ── 3. Apply segmentation head ────────────────────────────────────
        x = self.seg_head(x)                           # [B, num_classes, 14, 14]

        # ── 4. Bilinear upsample to input resolution ──────────────────────
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,                   
        )                                            # [B, num_classes, 224, 224]

        return x

    @torch.no_grad()
    def predict(self, pixel_values: Tensor) -> Tensor:
        """Returns per-pixel class indices. Shape: [B, H, W]."""
        return self.forward(pixel_values).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    Standard cross-entropy loss for semantic segmentation.
    Ignores pixels labeled with ignore_index (common in ADE20K = 255).
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B, num_classes, H, W]
            targets: [B, H, W]  integer class labels

        Returns:
            scalar loss
        """
        return self.ce(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Quick usage demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = IJEPALinearSegHead(num_classes=150, image_size=224).to(device)
    criterion = SegmentationLoss(ignore_index=255)

    # Dummy batch
    images  = torch.randn(2, 3, 224, 224, device=device)
    targets = torch.randint(0, 150, (2, 224, 224), device=device)

    # Forward
    logits = model(images)                     # [2, 150, 224, 224]
    loss   = criterion(logits, targets)

    print(f"Output shape : {logits.shape}")    # torch.Size([2, 150, 224, 224])
    print(f"Loss         : {loss.item():.4f}")

    # Count trainable vs frozen params
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")


"""
arch1_3d_linear_head.py — 3D Linear Probe / Simple MLP Segmentation Head
=========================================================================
Extends Architecture 1 to volumetric inputs [B, C, D, H, W].
Freezes the inflated 3D I-JEPA encoder and trains only a 3D 1×1 conv
head + trilinear upsample to produce voxel-wise segmentation logits.

Typical use case:
  - Medical image segmentation (CT, MRI) with limited labels
  - Benchmarking 3D I-JEPA feature quality
  - Fast baseline: only the head is trained

Input:   [B, C, D, H, W]   e.g. [2, 1, 32, 224, 224] for CT volumes
Output:  [B, num_classes, D, H, W]

Dependencies:
    pip install torch transformers
    (ijepa_3d_backbone.py must be in the same directory)
"""

from .jepavitbase import IJEPAViT3D


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class IJEPALinearSegHead3D(nn.Module):
    """
    3D I-JEPA backbone + linear volumetric segmentation head.

    The head consists of a small 1×1×1 conv bottleneck that maps the
    1408-dim I-JEPA features to num_classes, followed by trilinear
    upsampling back to the original input resolution.

    Args:
        num_classes:     Number of segmentation classes.
        input_channels:  Input volume channels (1 for CT, 3 for RGB video).
        patch_d:         Depth patch size for the 3D patch embedding.
        patch_hw:        Height/width patch size (16, matching I-JEPA).
        freeze_backbone: Freeze inflated ViT weights.
        model_id:        HuggingFace I-JEPA 2D checkpoint to inflate.
    """

    def __init__(
        self,
        num_classes:     int  = 14,      # e.g. 14 organs for CT segmentation
        input_channels:  int  = 1,
        patch_d:         int  = 2,
        patch_hw:        int  = 16,
        freeze_backbone: bool = True,
        model_id:        str  = "facebook/ijepa_vitg16_22k",
        deep_supervision: bool = False
    ) -> None:
        super().__init__()

        hidden_size = 1408   # ViT-G constant

        # ── 3D Backbone (inflated from 2D I-JEPA) ────────────────────────
        self.backbone = IJEPAViT3D(
            input_channels = input_channels,
            patch_d     = patch_d,
            patch_h     = patch_hw,
            patch_w     = patch_hw,
            freeze      = freeze_backbone,
            model_id    = model_id,
        )

        # ── 3D Segmentation Head ──────────────────────────────────────────
        # 1×1×1 conv: projects volumetric feature map → class logits
        # Analogous to the 2D version but with Conv3d
        self.seg_head = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size // 4, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, D, H, W]  — input volume

        Returns:
            logits: [B, num_classes, D, H, W]
        """
        B, C, D, H, W = x.shape

        # ── 1. 3D Backbone → volumetric token sequence ───────────────────
        out       = self.backbone(x)
        tokens    = out['last_hidden_state']   # [B, N, 1408]
        grid_dhw  = out['grid_dhw']            # (gd, gh, gw)

        # ── 2. Reshape tokens → spatial feature volume ───────────────────
        # [B, N, 1408] → [B, 1408, gd, gh, gw]
        feat_vol = self.backbone.tokens_to_spatial(tokens, grid_dhw)

        # ── 3. Segmentation head ──────────────────────────────────────────
        logits_low = self.seg_head(feat_vol)   # [B, num_classes, gd, gh, gw]

        # ── 4. Trilinear upsample to original volume resolution ───────────
        logits = F.interpolate(
            logits_low,
            size=(D, H, W),
            mode='trilinear',
            align_corners=False,
        )   # [B, num_classes, D, H, W]

        return logits

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Returns per-voxel class indices. Shape: [B, D, H, W]."""
        return self.forward(x).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class VolumetricSegLoss(nn.Module):
    """
    Combined cross-entropy + Dice loss for volumetric segmentation.

    Cross-entropy handles per-voxel classification; Dice handles
    class imbalance (especially important in medical imaging where
    foreground structures are small relative to background).

    Args:
        num_classes:   Number of segmentation classes.
        ignore_index:  Voxel label to ignore (e.g. 255 or -1).
        dice_weight:   Weight of Dice term relative to CE.
        smooth:        Smoothing constant in Dice denominator.
    """

    def __init__(
        self,
        num_classes:  int   = 14,
        ignore_index: int   = -1,
        dice_weight:  float = 1.0,
        smooth:       float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.dice_weight  = dice_weight
        self.smooth       = smooth
        self.ce           = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _dice_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Multi-class Dice loss averaged over classes and batch.

        Args:
            logits:  [B, C, D, H, W]
            targets: [B, D, H, W]  integer labels

        Returns:
            scalar Dice loss
        """
        probs = logits.softmax(dim=1)   # [B, C, D, H, W]
        # One-hot encode targets: [B, C, D, H, W]
        target_oh = F.one_hot(
            targets.clamp(0),   # clamp ignore_index to 0 temporarily
            self.num_classes
        ).permute(0, 4, 1, 2, 3).float()   # [B, C, D, H, W]

        # Mask out ignored voxels
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs    = probs    * mask
            target_oh = target_oh * mask

        # Flatten spatial dims: [B, C, N_voxels]
        probs_flat  = probs.flatten(2)
        target_flat = target_oh.flatten(2)

        intersection = (probs_flat * target_flat).sum(-1)          # [B, C]
        union        = probs_flat.sum(-1) + target_flat.sum(-1)    # [B, C]

        dice_per_class = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return dice_per_class.mean()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B, num_classes, D, H, W]
            targets: [B, D, H, W]  integer voxel labels

        Returns:
            scalar loss
        """
        ce_loss   = self.ce(logits, targets)
        dice_loss = self._dice_loss(logits, targets)
        return ce_loss + self.dice_weight * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Quick usage demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Typical medical imaging setup:
    # - 1-channel CT volume
    # - 32 axial slices, 224×224 spatial
    # - 14 organ classes
    model     = IJEPALinearSegHead3D(num_classes=14, input_channels=1).to(device)
    criterion = VolumetricSegLoss(num_classes=14)

    volumes = torch.randn(2, 1, 32, 224, 224, device=device)
    targets = torch.randint(0, 14, (2, 32, 224, 224), device=device)

    logits = model(volumes)
    loss   = criterion(logits, targets)

    print(f"Input shape  : {volumes.shape}")   # [2, 1, 32, 224, 224]
    print(f"Output shape : {logits.shape}")    # [2, 14, 32, 224, 224]
    print(f"Loss         : {loss.item():.4f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}")