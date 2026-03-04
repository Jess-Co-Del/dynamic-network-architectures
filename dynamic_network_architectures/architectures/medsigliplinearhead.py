"""
Architecture 1 — Linear Probe / Simple MLP Head (MedSigLIP Backbone)
=====================================================================
Freezes the MedSigLIP vision encoder entirely and trains only a 1×1
conv + bilinear upsample segmentation head.

MedSigLIP specifics:
  - Based on SigLIP-400M architecture (ViT with 24 layers)
  - Patch size 14, hidden_size = 1152
  - Default image resolution 448×448 → 32×32 patch grid
  - NO [CLS] token — SigLIP uses sigmoid loss, not softmax,
    so there is no classification token prepended.
  - last_hidden_state: [B, num_patches, hidden_size] (all patch tokens)
  - Loads via SiglipVisionModel from HuggingFace

Dependencies:
    pip install torch torchvision transformers

Reference:
    Sellergren et al. "MedGemma Technical Report."
    arXiv:2507.05201, 2025.
    MedSigLIP model: https://huggingface.co/google/medsiglip-448
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

class MedSigLIPLinearSegHead(nn.Module):
    """
    MedSigLIP vision encoder + linear segmentation head.

    Forward pass output shape: [B, num_classes, image_size, image_size]

    Key difference from DINO-family models: SigLIP does NOT use a [CLS]
    token, so last_hidden_state contains only patch tokens.

    Args:
        input_channels:   Number of input channels (1 for grayscale, 3 for RGB).
        num_classes:      Number of semantic classes.
        image_size:       Input image spatial resolution (448 for MedSigLIP).
        freeze_backbone:  If True, no gradients flow into MedSigLIP encoder.
        model_id:         HuggingFace model identifier.
        deep_supervision: Unused, kept for API compatibility.
    """

    def __init__(
        self,
        input_channels:   int  = 1,
        num_classes:      int  = 150,
        image_size:       int  = 448,
        freeze_backbone:  bool = True,
        model_id:         str  = "google/medsiglip-448",
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.input_channels = input_channels

        # ── Backbone (vision encoder only) ────────────────────────────────
        # MedSigLIP is a SiglipModel (vision + text); we only need vision.
        full_model = AutoModel.from_pretrained(model_id)
        self.backbone = full_model.vision_model
        del full_model  # free the text encoder

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Read config from the vision sub-config
        vision_config = self.backbone.config
        hidden_size = vision_config.hidden_size             # 1152
        self.patch_size = vision_config.patch_size           # 14
        self.grid_size = image_size // self.patch_size       # 32 for 448/14

        # ── Segmentation head ────────────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 4, kernel_size=1),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 4, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: [B, C, image_size, image_size]

        Returns:
            logits: [B, num_classes, image_size, image_size]
        """
        B = pixel_values.shape[0]

        # Tile grayscale → 3-ch if needed
        if self.input_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        # ── 1. Run MedSigLIP vision encoder ───────────────────────────────
        outputs = self.backbone(pixel_values=pixel_values)
        # SigLIP has NO CLS token:
        # last_hidden_state: [B, num_patches, hidden_size]
        #   num_patches = grid_size² = 1024 for 448/14

        patch_tokens = outputs.last_hidden_state  # [B, N, D]

        # ── 2. Reshape to spatial grid ────────────────────────────────────
        x = (patch_tokens
             .permute(0, 2, 1)                                    # [B, D, N]
             .reshape(B, -1, self.grid_size, self.grid_size))  # [B, D, g, g]

        # ── 3. Apply segmentation head ────────────────────────────────────
        x = self.seg_head(x)  # [B, num_classes, g, g]

        # ── 4. Bilinear upsample to input resolution ──────────────────────
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

class MedSigLIPLinearSegHead3D(nn.Module):
    """
    3D MedSigLIP backbone + linear volumetric segmentation head.

    Processes volumetric inputs [B, C, D, H, W] slice-by-slice through
    the 2D MedSigLIP vision encoder, then assembles and upsamples to
    produce voxel-wise segmentation logits.

    Particularly well-suited for medical imaging since MedSigLIP was
    pre-trained on chest X-rays, CT slices, MRI slices, dermatology,
    ophthalmology, and histopathology images.

    Args:
        num_classes:     Number of segmentation classes.
        input_channels:  Input volume channels (1 for CT/MRI).
        freeze_backbone: Freeze MedSigLIP weights.
        model_id:        HuggingFace MedSigLIP checkpoint.
        image_size:      H/W resolution of each slice (448 default).
        deep_supervision: Unused, kept for API compatibility.
    """

    def __init__(
        self,
        num_classes:     int  = 14,
        input_channels:  int  = 1,
        freeze_backbone: bool = True,
        model_id:        str  = "google/medsiglip-448",
        image_size:      int  = 448,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.image_size = image_size

        # ── Backbone (vision encoder only) ────────────────────────────────
        full_model = AutoModel.from_pretrained(model_id)
        self.backbone = full_model.vision_model
        del full_model

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        vision_config = self.backbone.config
        hidden_size = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.grid_size = image_size // self.patch_size

        # ── 3D Segmentation Head ─────────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size // 4, num_classes, kernel_size=1),
        )

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

        if self.input_channels == 1:
            slices = slices.repeat(1, 3, 1, 1)

        outputs = self.backbone(pixel_values=slices)
        # SigLIP: no CLS token, all tokens are patches
        patch_tokens = outputs.last_hidden_state  # [B*D, N, hidden]

        hidden = patch_tokens.shape[-1]
        feat = (patch_tokens
                .permute(0, 2, 1)
                .reshape(B * D, hidden, g, g))

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
    print("MedSigLIP 2D Linear Seg Head")
    print("=" * 60)
    model     = MedSigLIPLinearSegHead(
        num_classes=14, image_size=448
    ).to(device)
    criterion = SegmentationLoss(ignore_index=255)

    images  = torch.randn(2, 3, 448, 448, device=device)
    targets = torch.randint(0, 14, (2, 448, 448), device=device)

    logits = model(images)
    loss   = criterion(logits, targets)

    print(f"Output shape : {logits.shape}")    # [2, 14, 448, 448]
    print(f"Loss         : {loss.item():.4f}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    # --- 3D ---
    print()
    print("=" * 60)
    print("MedSigLIP 3D Linear Seg Head")
    print("=" * 60)
    model3d   = MedSigLIPLinearSegHead3D(
        num_classes=14, input_channels=1, image_size=448
    ).to(device)
    criterion3d = VolumetricSegLoss(num_classes=14)

    volumes = torch.randn(2, 1, 16, 448, 448, device=device)
    targets3d = torch.randint(0, 14, (2, 16, 448, 448), device=device)

    logits3d = model3d(volumes)
    loss3d   = criterion3d(logits3d, targets3d)

    print(f"Input shape  : {volumes.shape}")   # [2, 1, 16, 448, 448]
    print(f"Output shape : {logits3d.shape}")  # [2, 14, 16, 448, 448]
    print(f"Loss         : {loss3d.item():.4f}")

    trainable3d = sum(p.numel() for p in model3d.parameters() if p.requires_grad)
    frozen3d    = sum(p.numel() for p in model3d.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable3d:,}")
    print(f"Frozen params    : {frozen3d:,}")
