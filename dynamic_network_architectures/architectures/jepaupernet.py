"""
Multi-Scale UPerNet-style FPN Decoder
=======================================================
Taps intermediate ViT layers to simulate a feature pyramid (FPN),
then fuses them top-down through a UPerNet-style head.

This is the best choice for fine-grained semantic segmentation with
crisp object boundaries — it explicitly recovers spatial detail that
is lost in the deepest ViT layers.

Use this for:
  - High-resolution semantic segmentation (ADE20K, Cityscapes)
  - When boundary quality matters more than instance awareness
  - Larger datasets where the multi-scale head can be fully trained

Advantages over Architecture 1:
  - Multi-scale features simulate an FPN without a CNN backbone
  - Progressive upsampling → much sharper boundaries
  - Intermediate layer features capture both fine texture and semantics

Limitations:
  - More parameters than the linear head
  - No query-based reasoning — cannot do instance segmentation
  - All ViT intermediate features are 14×14 (no true spatial pyramid
    from the backbone itself — we simulate it via progressive upsample)

Dependencies:
    pip install torch torchvision transformers

References:
    Xiao et al. "Unified Perceptual Parsing for Scene Understanding."
    ECCV 2018. (UPerNet)  arXiv:1807.10221

    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023. arXiv:2301.08243
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────────────────
# FPN Neck
# ─────────────────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck.

    Takes 4 feature maps of identical resolution (14×14 from ViT)
    and produces 4 output maps at progressively doubled resolutions:
        P4: 14×14  (stride-16, deepest / most semantic)
        P3: 28×28  (stride-8)
        P2: 56×56  (stride-4)
        P1: 112×112 (stride-2, finest)

    Top-down fusion: information flows from deep (P4) → shallow (P1),
    so semantic understanding enriches fine-grained spatial features.

    Args:
        in_channels:  Channel count of each input feature map (1408 for ViT-G).
        out_channels: Channel count for all output FPN levels.
    """

    def __init__(self, in_channels: int = 1408, out_channels: int = 256) -> None:
        super().__init__()

        # Lateral 1×1 convs: channel reduction for each tapped ViT layer
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_channels),
            )
            for _ in range(4)   # one per tapped layer
        ])

        # Output 3×3 convs: smooth after top-down addition
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: list of 4 tensors, each [B, in_channels, 14, 14]
                      ordered from shallowest (index 0) to deepest (index 3)

        Returns:
            list of 4 tensors at resolutions: [14, 28, 56, 112] × out_channels
        """
        # Step 1 — lateral projections (still all 14×14)
        laterals = [self.lateral_convs[i](f) for i, f in enumerate(features)]

        # Step 2 — top-down fusion with progressive upsampling
        # Start from deepest (index 3) and propagate up
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample the deeper feature to twice its current size
            target_scale = 2 ** (len(laterals) - 1 - i)
            upsampled = F.interpolate(
                laterals[i + 1],
                scale_factor=2,
                mode='bilinear',
                align_corners=False,
            )
            laterals[i] = laterals[i] + F.interpolate(
                upsampled,
                size=laterals[i].shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Step 3 — apply output convs + progressive upsample
        # Each level doubles the spatial resolution compared to the previous
        outs = []
        for i, lat in enumerate(laterals):
            scale_factor = 2 ** i          # level 0 → 1×, 1 → 2×, 2 → 4×, 3 → 8×
            upsampled = F.interpolate(
                lat,
                scale_factor=float(scale_factor),
                mode='bilinear',
                align_corners=False,
            )
            outs.append(self.output_convs[i](upsampled))

        # outs:  [14×14,  28×28,  56×56,  112×112]
        return outs


# ─────────────────────────────────────────────────────────────────────────────
# UPerNet Fusion Head
# ─────────────────────────────────────────────────────────────────────────────

class UPerHead(nn.Module):
    """
    UPerNet-style fusion head.

    Fuses all FPN levels via:
      1. PPM (Pooling Pyramid Module) on the deepest feature
      2. Concatenation of all levels
      3. 3×3 conv fusion
      4. Final 1×1 segmentation classifier

    Args:
        in_channels:  Channel count of each FPN level.
        num_classes:  Number of semantic segmentation classes.
        pool_scales:  Scales for the Pooling Pyramid Module.
    """

    def __init__(
        self,
        in_channels:  int        = 256,
        num_classes:  int        = 150,
        pool_scales:  Tuple[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.num_fpn_levels = 4

        # ── PPM on the deepest feature (captures global context) ──────────
        self.ppm_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])

        # PPM output: original + 4 pooled → (1 + len(pool_scales)) × in_channels
        ppm_out_channels = in_channels + len(pool_scales) * in_channels

        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # ── FPN level fusion ──────────────────────────────────────────────
        # After upsampling all levels to the same resolution and concatenating:
        # 4 levels × in_channels channels
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(self.num_fpn_levels * in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # ── Final classifier ──────────────────────────────────────────────
        self.cls_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, fpn_features: List[Tensor]) -> Tensor:
        """
        Args:
            fpn_features: list of 4 tensors [B, C, H_i, W_i]
                          resolutions: [14, 28, 56, 112]

        Returns:
            logits: [B, num_classes, 112, 112]
                    (caller is responsible for final upsampling to input res)
        """
        # ── PPM on deepest feature (14×14) ────────────────────────────────
        deepest = fpn_features[-1]   # [B, 256, 14, 14]  (index 0 is shallowest)
        # Wait — our ordering: index 0 = 14×14 (most semantic from deepest ViT layer)
        #                       index 3 = 112×112 (from shallowest ViT layer)
        # Use index 0 for PPM (most semantic)
        semantic = fpn_features[0]   # [B, 256, 14, 14]

        ppm_outs = [semantic]
        for ppm_conv in self.ppm_convs:
            pooled = ppm_conv(semantic)               # [B, 256, scale, scale]
            ppm_outs.append(
                F.interpolate(pooled, size=semantic.shape[-2:],
                              mode='bilinear', align_corners=False)
            )
        ppm_fused = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))   # [B, 256, 14, 14]

        # Replace deepest level with PPM-enriched version
        fpn_list = [ppm_fused] + list(fpn_features[1:])

        # ── Upsample all levels to the finest resolution (112×112) ────────
        target_h, target_w = fpn_features[-1].shape[-2:]  # 112×112
        aligned = [
            F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
            for f in fpn_list
        ]

        # ── Concatenate + fuse ────────────────────────────────────────────
        fused = self.fpn_bottleneck(torch.cat(aligned, dim=1))  # [B, 256, 112, 112]

        return self.cls_seg(fused)   # [B, num_classes, 112, 112]


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────

class IJEPAUPerNetSeg(nn.Module):
    """
    I-JEPA ViT-G/16 backbone + Multi-scale FPN neck + UPerNet head.

    Taps 4 intermediate ViT-G layers (every ~10 layers of 40 total)
    to simulate a feature pyramid, then fuses them with a UPerNet head.

    Output: [B, num_classes, image_size, image_size]

    Args:
        num_classes:     Semantic classes.
        out_layer_indices: Which ViT layers to tap. ViT-G has 40 layers.
                           Default: (9, 19, 29, 39) — taps every 10th.
        fpn_channels:    FPN output channel width.
        freeze_backbone: Freeze I-JEPA weights.
        image_size:      Input/output spatial resolution.
        patch_size:      ViT patch size.
        model_id:        HuggingFace checkpoint.
    """

    # ViT-G layer tap indices (0-indexed; 40 total layers)
    DEFAULT_TAP_LAYERS: Tuple[int, ...] = (9, 19, 29, 39)

    def __init__(
        self,
        input_channels:    int = 1,
        num_classes:       int            = 150,
        out_layer_indices: Tuple[int, ...]= DEFAULT_TAP_LAYERS,
        fpn_channels:      int            = 256,
        freeze_backbone:   bool           = True,
        image_size:        int            = 224,
        patch_size:        int            = 16,
        model_id:          str            = "facebook/ijepa_vitg16_22k",
        deep_supervision: bool = False
    ) -> None:
        super().__init__()

        self.image_size        = image_size
        self.grid_size         = image_size // patch_size   # 14
        self.out_layer_indices = out_layer_indices

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(
            model_id,
            output_hidden_states=True,   # needed to access intermediate layers
        )
        hidden_size = self.backbone.config.hidden_size  # 1408

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── FPN neck + UPerNet head ────────────────────────────────────────
        self.fpn  = FPNNeck(in_channels=hidden_size, out_channels=fpn_channels)
        self.head = UPerHead(in_channels=fpn_channels, num_classes=num_classes)

    def _tokens_to_spatial(self, tokens: Tensor) -> Tensor:
        """
        Reshape flat patch token sequence to spatial feature map.

        Args:
            tokens: [B, 197, D]   (197 = 1 CLS + 196 patches)

        Returns:
            spatial: [B, D, 14, 14]
        """
        B = tokens.shape[0]
        return (tokens  # [:, 1:, :]                                   # drop CLS → [B, 196, D]
                .permute(0, 2, 1)                                  # [B, D, 196]
                .reshape(B, -1, self.grid_size, self.grid_size))   # [B, D, 14, 14]

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]

        Returns:
            logits: [B, num_classes, H, W]
        """
        pixel_values = torch.tile(pixel_values, (1,3,1,1))  # [B, 3, H, W], H=W=224

        # ── 1. Run backbone, collect intermediate hidden states ───────────
        outputs = self.backbone(pixel_values=pixel_values)
        # hidden_states: tuple of (num_layers+1) tensors, each [B, 197, 1408]

        # ── 2. Extract and reshape tapped layers ──────────────────────────
        # Order: shallowest tap first (most spatial detail in early layers)
        multi_scale = [
            self._tokens_to_spatial(outputs.hidden_states[i])
            for i in self.out_layer_indices
        ]
        # multi_scale: list of 4 tensors, each [B, 1408, 14, 14]

        # ── 3. FPN neck: project + top-down fusion + upsample ─────────────
        fpn_features = self.fpn(multi_scale)
        # fpn_features: [B,256,14], [B,256,28], [B,256,56], [B,256,112]

        # ── 4. UPerNet head: PPM + concat + classify ──────────────────────
        logits_low = self.head(fpn_features)    # [B, num_classes, 112, 112]

        # ── 5. Final upsample to input resolution ─────────────────────────
        logits = F.interpolate(
            logits_low,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )                                       # [B, num_classes, 224, 224]

        return logits

    @torch.no_grad()
    def predict(self, pixel_values: Tensor) -> Tensor:
        """Returns per-pixel class indices. Shape: [B, H, W]."""
        return self.forward(pixel_values).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class UPerNetLoss(nn.Module):
    """
    Standard cross-entropy with optional auxiliary loss
    (common practice when training UPerNet-style heads).

    Args:
        ignore_index:  Pixel label to ignore (e.g. 255 in ADE20K).
        aux_weight:    Weight for auxiliary loss if used.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [B, num_classes, H, W]
            targets: [B, H, W]

        Returns:
            scalar loss
        """
        return self.ce(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Quick usage demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = IJEPAUPerNetSeg(num_classes=150).to(device)
    criterion = UPerNetLoss(ignore_index=255)

    images  = torch.randn(2, 3, 224, 224, device=device)
    targets = torch.randint(0, 150, (2, 224, 224), device=device)

    logits = model(images)
    loss   = criterion(logits, targets)

    print(f"Output shape : {logits.shape}")    # [2, 150, 224, 224]
    print(f"Loss         : {loss.item():.4f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}")