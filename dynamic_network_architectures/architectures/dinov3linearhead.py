"""
Architecture 1 — Linear Probe / Simple MLP Head (DINOv3 Backbone)
==================================================================
Freezes the DINOv3 backbone entirely and trains only a 1×1 conv +
bilinear upsample segmentation head.

DINOv3 specifics:
  - Patch size 16 → 14×14 grid for 224×224 input
  - Has [CLS] token at position 0 + register tokens (typically 4)
    → patch tokens start at index (1 + num_register_tokens)
  - Uses Rotary Position Embeddings (RoPE) with jittering
  - hidden_size: 384 (S), 768 (B), 1024 (L), etc.
  - Model IDs: "facebook/dinov3-vits16-pretrain-lvd1689m",
               "facebook/dinov3-vitb16-pretrain-lvd1689m",
               "facebook/dinov3-vitl16-pretrain-lvd1689m", etc.
  - Requires gated access on HuggingFace (accept license first)
  - Requires transformers installed from main branch (as of 2025)

Dependencies:
    pip install torch torchvision
    pip install git+https://github.com/huggingface/transformers.git

Reference:
    Siméoni et al. "DINOv3." arXiv:2508.10104, 2025.
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Dict, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class DINOv3LinearSegHead(nn.Module):
    """
    DINOv3 backbone + linear segmentation head.

    Forward pass output shape: [B, num_classes, image_size, image_size]

    Key difference from DINOv2: DINOv3 outputs include register tokens
    between the [CLS] token and the patch tokens, so we must skip
    (1 + num_register_tokens) prefix tokens when extracting patch features.

    Args:
        input_channels:   Number of input channels (1 for grayscale, 3 for RGB).
        num_classes:      Number of semantic classes (e.g. 150 for ADE20K).
        image_size:       Input image spatial resolution (assumed square).
        freeze_backbone:  If True, no gradients flow into DINOv3 encoder.
        model_id:         HuggingFace model identifier for DINOv3.
        deep_supervision: Unused, kept for API compatibility.
    """

    def __init__(
        self,
        input_channels:   int  = 1,
        num_classes:      int  = 150,
        image_size:       int  = 224,
        freeze_backbone:  bool = True,
        model_id:         str  = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.input_channels = input_channels

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(model_id)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size            # 16
        self.grid_size = image_size // self.patch_size               # 14
        self.num_register_tokens = getattr(
            self.backbone.config, "num_register_tokens", 4
        )
        # Total prefix tokens to skip: 1 (CLS) + num_register_tokens
        self.num_prefix_tokens = 1 + self.num_register_tokens

        # ── Segmentation head ────────────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 4, num_classes, kernel_size=1),
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _preprocess(self, pixel_values: Tensor) -> Tensor:
        """
        Tile grayscale → 3-ch if needed, then apply ImageNet normalization.

        Args:
            pixel_values: [B, C, H, W] with values in [0, 1].

        Returns:
            Normalized [B, 3, H, W] tensor.
        """
        # Tile grayscale → 3-ch if needed
        if self.input_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        
        if pixel_values.max() > 1. or  pixel_values.max() < 0.:
            pixel_values = (pixel_values - pixel_values.min())
            pixel_values /= pixel_values.max()

        return (pixel_values - self.pixel_mean) / self.pixel_std

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: [B, C, image_size, image_size]

        Returns:
            logits: [B, num_classes, image_size, image_size]
        """
        B = pixel_values.shape[0]

        # 0. Preprocess
        pixel_values = self._preprocess(pixel_values)
        #print(pixel_values.max(), pixel_values.min())

        # 1. Run DINOv3 encoder ─────────────────────────────────────────
        outputs = self.backbone(pixel_values=pixel_values)
        # last_hidden_state: [B, 1 + num_reg + num_patches, hidden_size]

        # 2. Skip CLS + register tokens, reshape to spatial grid ────────
        patch_tokens = outputs.last_hidden_state[:, self.num_prefix_tokens:, :]
        # [B, num_patches, hidden_size]

        x = (patch_tokens
             .permute(0, 2, 1)                                    # [B, D, N]
             .reshape(B, -1, self.grid_size, self.grid_size))  # [B, D, g, g]

        # 3. Apply segmentation head ────────────────────────────────────
        x = self.seg_head(x)  # [B, num_classes, g, g]

        # 4. Bilinear upsample to input resolution ──────────────────────
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # [B, num_classes, H, W]

        return x

    @torch.no_grad()
    def predict(self, pixel_values: Tensor) -> Tensor:
        """Returns per-pixel class indices. Shape: [B, H, W]."""
        return self.forward(pixel_values).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3D Variant
# ─────────────────────────────────────────────────────────────────────────────

class DINOv3LinearSegHead3D(nn.Module):
    """
    3D DINOv3 backbone + linear volumetric segmentation head.

    Processes volumetric inputs [B, C, D, H, W] slice-by-slice through
    the 2D DINOv3 encoder, then assembles and upsamples to produce
    voxel-wise segmentation logits.

    Args:
        num_classes:     Number of segmentation classes.
        input_channels:  Input volume channels (1 for CT, 3 for RGB video).
        freeze_backbone: Freeze DINOv3 weights.
        model_id:        HuggingFace DINOv3 checkpoint.
        image_size:      H/W resolution of each slice.
        deep_supervision: Unused, kept for API compatibility.
    """

    def __init__(
        self,
        num_classes:     int  = 14,
        input_channels:  int  = 1,
        freeze_backbone: bool = True,
        model_id:        str  = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size:      int  = 224,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.image_size = image_size

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(model_id)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size            # 16
        self.grid_size = image_size // self.patch_size               # 14
        self.num_register_tokens = getattr(
            self.backbone.config, "num_register_tokens", 4
        )
        self.num_prefix_tokens = 1 + self.num_register_tokens

        # ── 3D Segmentation Head ─────────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size // 4, num_classes, kernel_size=1),
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _preprocess(self, pixel_values: Tensor) -> Tensor:
        """
        Tile grayscale → 3-ch if needed, then apply ImageNet normalization.

        Args:
            pixel_values: [B, C, H, W] with values in [0, 1].

        Returns:
            Normalized [B, 3, H, W] tensor.
        """
        # Tile grayscale → 3-ch if needed
        if self.input_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        
        if pixel_values.max() > 1. or  pixel_values.max() < 0.:
            pixel_values = (pixel_values - pixel_values.min())
            pixel_values /= pixel_values.max()
            
        return (pixel_values - self.pixel_mean) / self.pixel_std

    def _encode_slices(self, x: Tensor) -> Tensor:
        """
        Encode a volume slice-by-slice through the 2D backbone.

        Args:
            x: [B, C, D, H, W]

        Returns:
            feat_vol: [B, hidden_size, D, grid_h, grid_w]
        """
        B, C, D, H, W = x.shape
        g = self.grid_size

        # Reshape to process all slices in one batch
        slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        slices = self._preprocess(slices)

        #print(slices.max(), slices.min())
        outputs = self.backbone(pixel_values=slices)
        # Skip CLS + register tokens
        patch_tokens = outputs.last_hidden_state[:, self.num_prefix_tokens:, :]

        hidden = patch_tokens.shape[-1]
        feat = (patch_tokens
                .permute(0, 2, 1)
                .reshape(B * D, hidden, g, g))

        # Reassemble depth dimension
        feat_vol = feat.reshape(B, D, hidden, g, g).permute(0, 2, 1, 3, 4)
        return feat_vol  # [B, hidden, D, g, g]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, D, H, W]

        Returns:
            logits: [B, num_classes, D, H, W]
        """
        B, C, D, H, W = x.shape

        feat_vol = self._encode_slices(x)
        logits_low = self.seg_head(feat_vol)

        logits = F.interpolate(
            logits_low,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False,
        )

        return logits

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Returns per-voxel class indices. Shape: [B, D, H, W]."""
        return self.forward(x).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss (shared)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """Standard cross-entropy for 2D semantic segmentation."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.ce(logits, targets)


class VolumetricSegLoss(nn.Module):
    """Combined CE + Dice loss for 3D segmentation."""

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
        probs = logits.softmax(dim=1)
        target_oh = F.one_hot(
            targets.clamp(0), self.num_classes
        ).permute(0, 4, 1, 2, 3).float()

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs     = probs * mask
            target_oh = target_oh * mask

        probs_flat  = probs.flatten(2)
        target_flat = target_oh.flatten(2)

        intersection = (probs_flat * target_flat).sum(-1)
        union        = probs_flat.sum(-1) + target_flat.sum(-1)

        dice_per_class = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return dice_per_class.mean()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.ce(logits, targets) + self.dice_weight * self._dice_loss(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Quick usage demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2D ---
    print("=" * 60)
    print("DINOv3 2D Linear Seg Head")
    print("=" * 60)
    model     = DINOv3LinearSegHead(num_classes=150, image_size=224).to(device)
    criterion = SegmentationLoss(ignore_index=255)

    images  = torch.randn(2, 3, 224, 224, device=device)
    targets = torch.randint(0, 150, (2, 224, 224), device=device)

    logits = model(images)
    loss   = criterion(logits, targets)

    print(f"Output shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    # --- 3D ---
    print()
    print("=" * 60)
    print("DINOv3 3D Linear Seg Head")
    print("=" * 60)
    model3d   = DINOv3LinearSegHead3D(num_classes=14, input_channels=1).to(device)
    criterion3d = VolumetricSegLoss(num_classes=14)

    volumes = torch.randn(2, 1, 32, 224, 224, device=device)
    targets3d = torch.randint(0, 14, (2, 32, 224, 224), device=device)

    logits3d = model3d(volumes)
    loss3d   = criterion3d(logits3d, targets3d)

    print(f"Input shape  : {volumes.shape}")
    print(f"Output shape : {logits3d.shape}")
    print(f"Loss         : {loss3d.item():.4f}")

    trainable3d = sum(p.numel() for p in model3d.parameters() if p.requires_grad)
    frozen3d    = sum(p.numel() for p in model3d.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable3d:,}")
    print(f"Frozen params    : {frozen3d:,}")
