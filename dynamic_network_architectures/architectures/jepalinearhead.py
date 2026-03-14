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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
        self.input_channels = input_channels

        # ── Built-in normalization (ImageNet stats) ──────────────────────
        # Expects raw [0, 1] input; applies ImageNet mean/std.
        # Buffers follow .to(device) / .half() automatically.
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

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

    def _normalize(self, pixel_values: Tensor) -> Tensor:
        """
        Tile grayscale → 3-ch if needed, then apply ImageNet normalization.

        Args:
            pixel_values: [B, C, H, W] with values in [0, 1].

        Returns:
            Normalized [B, 3, H, W] tensor.
        """
        if self.input_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        # Normalize to range [0,1] if needed
        if pixel_values.max() > 1. or  pixel_values.max() < 0.:
            pixel_values = (pixel_values - pixel_values.min())
            pixel_values /= pixel_values.max()

        return (pixel_values - self.pixel_mean) / self.pixel_std

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: [B, C, image_size, image_size]  values in [0, 1]

        Returns:
            logits: [B, num_classes, image_size, image_size]
        """
        B = pixel_values.shape[0]

        # ── 0. Normalize ──────────────────────────────────────────────────
        pixel_values = self._normalize(pixel_values)  # [B, 3, H, W]

        # ── 1. Run I-JEPA encoder ─────────────────────────────────────────
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: [B, 196, 1408]  (196 patches for 224/16)

        # ── 2. Reshape to spatial grid ───────────────────
        patch_tokens = outputs.last_hidden_state  # [B, 196, 1408]
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

    # Dummy batch — values in [0, 1] (normalization is built in)
    images  = torch.rand(2, 3, 224, 224, device=device)
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
