"""
UNETR Decoder for Frozen ViT Backbones (2D Slice-Wise Liver Tumor Segmentation)
=================================================================================
Adapts the UNETR architecture (Hatamizadeh et al., WACV 2022) to work with
same-resolution patch tokens from frozen ViT backbones (e.g. DINOv2, I-JEPA,
SigLIP) that lack hierarchical spatial downsampling.

Key design differences vs. original UNETR:
  - The ViT produces tokens at a SINGLE resolution (no spatial shrinkage).
  - We tap intermediate transformer layers {L/4, L/2, 3L/4, L} to recover a
    pseudo-hierarchy, then upsample each to a 2× finer resolution than the
    previous stage — giving a proper decoder pyramid.
  - Each skip connection goes through a "reshape → DeConv" block before being
    merged with the bottom-up decoder stream.
  - The final head upsamples to the original image resolution.

Architecture overview (for ViT-L/16, image 512×512):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Frozen ViT backbone (e.g. DINOv2-L/14 or I-JEPA ViT-L/16)     │
  │  Patch tokens: [B, N, D]  where N = (H/P)*(W/P), D = embed_dim  │
  │  Tapped layers: z3, z6, z9, z12 (indices vary by depth)         │
  └─────────────────────────────────────────────────────────────────┘
            │ z3        │ z6        │ z9        │ z12 (bottleneck)
            ▼           ▼           ▼           ▼
       Reshape+Proj  Reshape+Proj  Reshape+Proj  Reshape+Proj
       (H/16,W/16)  (H/16,W/16)  (H/16,W/16)  (H/16,W/16)
            │           │           │           │
            │    DeConv2x    DeConv2x    DeConv2x
            │           │           │           │
            └──cat──────┘  └──cat──────┘  └──cat──────┘
               │               │               │
            DeConv2x        DeConv2x        DeConv2x
               │               │               │
               └───────────────┴───────────────┘
                                 │
                          Final DeConv head
                                 │
                          [B, num_classes, H, W]

Usage
-----
    backbone  = DINOv2(..., return_all_layers=True)   # returns list of tensors
    decoder   = UNETRDecoder(
                    embed_dim=1024, patch_size=16,
                    image_size=512, num_classes=3,
                    feature_size=64, num_layers=24,
                )
    tokens_per_layer = backbone(x)          # list of [B, N, D], len=num_layers
    logits  = decoder(tokens_per_layer, image_hw=(512, 512))
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_conv_blocks import ConvBNReLU


class DeConvBlock(nn.Module):
    """
    Transposed-convolution upsampling block used in UNETR.
    Doubles spatial resolution then refines with two conv layers.
    """
    def __init__(self, in_ch: int, out_ch: int, n_conv: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        n_convs = []
        for _ in range(n_conv):
            n_convs.append(ConvBNReLU(out_ch, out_ch))
        self.refine = nn.Sequential(*n_convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.deconv(x))


class MergeBlock(nn.Module):
    """
    Merges a skip-connection feature map with the bottom-up decoder stream.
    Both tensors must have the same spatial size before calling this module.
    """
    def __init__(self, skip_ch: int, up_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(skip_ch + up_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        x = torch.cat([skip, up], dim=1)
        return self.conv(x)


class TokenProject(nn.Module):
    """
    Projects the channel dimension with a 1×1 conv.

    Args:
        embed_dim : ViT embedding dimension (D).
        out_ch    : Target channel count after projection.
    """
    def __init__(self, embed_dim: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]  (class token already removed upstream)
        Returns:
            [B, out_ch, H', W']
        """
        return self.proj(x)  # [B, out_ch, H', W']


# ---------------------------------------------------------------------------
# Main UNETR Decoder
# ---------------------------------------------------------------------------

class UNETRDecoder(nn.Module):
    """
    UNETR-style decoder for same-resolution ViT features.

    Args:
        embed_dim   : ViT hidden dimension (e.g. 768 for ViT-B, 1024 for ViT-L).
        patch_size  : ViT patch size in pixels (e.g. 14 for DINOv2, 16 for I-JEPA).
        image_size  : Assumed square input size in pixels (e.g. 512).
                      Also accepts (H, W) tuple for non-square inputs.
        num_classes : Number of output segmentation classes.
        feature_size: Base number of decoder channels (F). Stages use
                      [8F, 4F, 2F, F] channels top-down. Default 64.
        num_layers  : Total depth of the ViT (e.g. 12 for ViT-B, 24 for ViT-L).
                      Used to auto-select the four tap indices.
        tap_indices : Optional explicit list of four layer indices (0-based) to
                      tap as skip connections, ordered shallowest to deepest.
                      If None, defaults to [L//4, L//2, 3*L//4, L-1].
        dropout     : Dropout probability applied before the final 1×1 head.

    Inputs (forward):
        layer_tokens : List of [B, N, D] tensors — one per ViT layer.
                       Provide ALL layers; the decoder selects the tap layers.
                       Class token must be removed before passing in.
        image_hw     : (H, W) of the original input image. Used to upsample
                       the final output to full resolution.

    Output:
        logits : [B, num_classes, H, W] — raw (unactivated) segmentation logits.

    Example
    -------
        decoder = UNETRDecoder(
            embed_dim=1024, patch_size=14, image_size=518,
            num_classes=3, feature_size=64, num_layers=24,
        )
        # tokens_all: list of 24 tensors, each [B, 1369, 1024]  (518/14=37, 37²=1369)
        logits = decoder(tokens_all, image_hw=(518, 518))
        # logits: [B, 3, 518, 518]
    """

    def __init__(
        self,
        hidden_dim: int,
        input_channels: int,
        image_size,
        num_classes: int,
        feature_size: int = 64,
        num_layers: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── spatial grid at patch resolution ──────────────────────────────
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        # ── channel counts for the four decoder stages ─────────────────────
        F = feature_size
        # Stage channels (coarsest → finest): 8F, 4F, 2F, F
        stage_chs = [F, 2 * F, 4 * F, 8 * F]
        self.stage_chs = stage_chs
        # Each tapped layer: [B, stage_ch, grid_h, grid_w]
        self.skip_projs = nn.ModuleList([
            TokenProject(hidden_dim, ch)
            for ch in stage_chs
        ])
        # ── decoder stages (bottom-up) ────────────────────────────────────
        # Stage 0: bottleneck (deepest tap) → upsample 2×
        self.deconv0 = DeConvBlock(stage_chs[3], stage_chs[3])
        # Stage 1: merge with skip[2], then upsample 2×
        self.skip_upsample1 = DeConvBlock(stage_chs[2], stage_chs[2], n_conv=1)
        self.merge1  = MergeBlock(stage_chs[2], stage_chs[3], stage_chs[2])
        self.deconv1 = DeConvBlock(stage_chs[2], stage_chs[2], n_conv=2)
        # Stage 2: merge with skip[1], then upsample 2×
        self.skip_upsample2 = nn.Sequential(
            DeConvBlock(stage_chs[1], stage_chs[1], n_conv=1),
            DeConvBlock(stage_chs[1], stage_chs[1], n_conv=1)
        )
        self.merge2  = MergeBlock(stage_chs[1], stage_chs[2], stage_chs[1])
        self.deconv2 = DeConvBlock(stage_chs[1], stage_chs[1], n_conv=2)
        # Stage 3: merge with skip[0], then upsample 2×
        self.skip_upsample3 = nn.Sequential(
            DeConvBlock(stage_chs[0], stage_chs[0], n_conv=1),
            DeConvBlock(stage_chs[0], stage_chs[0], n_conv=1),
            DeConvBlock(stage_chs[0], stage_chs[0], n_conv=1),
        )
        self.merge3  = MergeBlock(stage_chs[0], stage_chs[1], stage_chs[0])
        self.deconv3 = DeConvBlock(stage_chs[0], stage_chs[0], n_conv=2)

        # Stage 4: Convolve original input 2x, merge with inputs, Convolve 2x
        self.inputconv  = nn.Sequential(
            ConvBNReLU(input_channels, stage_chs[0]),
            ConvBNReLU(stage_chs[0], stage_chs[0])
        )
        self.mergefinal  = MergeBlock(stage_chs[0], stage_chs[0], stage_chs[0])

        # ── final head ────────────────────────────────────────────────────
        # After 4× deconv2× stages the spatial size is grid_hw * 16.
        # We then bilinearly upsample to the exact image_size.
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(stage_chs[0], num_classes, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        layer_tokens: List[torch.Tensor],
        inputs: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            layer_tokens : list of  4 tensors [B, hidden_dim, grid_h, grid_w], one per ViT layer.
            image_hw     : target output (H, W). Defaults to self.image_size.
        Returns:
            logits [B, num_classes, H, W]
        """
        input_skip = self.inputconv(inputs)

        if image_hw is None:
            image_hw = self.image_size

        # ── select the four tapped layers ─────────────────────────────────
        z0, z1, z2, z3 = layer_tokens

        # ── project tokens → spatial feature maps ─────────────────────────
        s0 = self.skip_projs[0](z0)   # [B, F, gh, gw]
        s1 = self.skip_projs[1](z1)   # [B, 2F, gh, gw]
        s2 = self.skip_projs[2](z2)   # [B, 4F, gh, gw]
        s3 = self.skip_projs[3](z3)   # [B, 8F, gh, gw]

        # ── bottom-up decoder ─────────────────────────────────────────────
        # All skips start at the same (gh, gw); we progressively upsample.
        # The UNETR design: start from bottleneck, upsample, merge shallower skip.

        x = self.deconv0(s3)        # [B,  F, gh*2, gw*2]

        # Align skip s2 to x's spatial size before merging
        s2_up = self.skip_upsample1(s2)
        x = self.merge1(s2_up, x)   # [B, 2F, gh*2, gw*2]
        x = self.deconv1(x)         # [B, 2F, gh*4, gw*4]

        s1_up = self.skip_upsample2(s1)
        x = self.merge2(s1_up, x)   # [B, 4F, gh*4, gw*4]
        x = self.deconv2(x)         # [B, 4F, gh*8, gw*8]

        s0_up = self.skip_upsample3(s0)
        x = self.merge3(s0_up, x)   # [B, 8F, gh*8, gw*8]
        x = self.deconv3(x)         # [B, 8F, gh*16, gw*16]

        # ── Align to exact image resolution ────────────────────────────
        x = F.interpolate(x, size=image_hw, mode="bilinear", align_corners=False)
        x = self.mergefinal(input_skip, x)

        # ── head ──────────────────────────────────────────────────────────
        x = self.dropout(x)
        logits = self.head(x)       # [B, num_classes, H, W]
        return logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Returns the number of trainable decoder parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Integration shim: wraps a backbone that returns all layer tokens
# ---------------------------------------------------------------------------

class UNETRSegmenter(nn.Module):
    """
    Thin wrapper combining a frozen ViT backbone with the UNETRDecoder.

    Expects `backbone(x)` to return a list of [B, N, D] tensors (one per
    layer), with the class token already stripped.  If your backbone returns
    the class token as the first token, set `strip_cls_token=True`.

    Args:
        backbone        : Frozen ViT with `return_all_layers=True`.
        decoder         : UNETRDecoder instance.
        strip_cls_token : Set True if backbone prepends a [CLS] token to N.
    """
    def __init__(
        self,
        backbone: nn.Module,
        decoder: UNETRDecoder,
        strip_cls_token: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.strip_cls_token = strip_cls_token

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W] — normalised CT slices (single-channel or RGB).
        Returns:
            logits [B, num_classes, H, W]
        """
        image_hw = (x.shape[2], x.shape[3])

        with torch.no_grad():
            layer_tokens = self.backbone(x)  # list of [B, N(+1), D]

        return self.decoder(layer_tokens, x, image_hw=image_hw)


# ---------------------------------------------------------------------------
# Quick sanity check (runs as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("UNETRDecoder — sanity check")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- DINOv2-L / 14 config ---
    BATCH      = 2
    IMAGE_SIZE = 224          # 37 * 14
    PATCH_SIZE = 14
    EMBED_DIM  = 1024
    NUM_LAYERS = 4
    NUM_CLS    = 3            # background, liver, tumour
    GRID_H = GRID_W = IMAGE_SIZE // PATCH_SIZE   # = 37
    N_TOKENS    = GRID_H * GRID_W                # = 1369

    decoder = UNETRDecoder(
        hidden_dim=EMBED_DIM,
        input_channels=3,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLS,
        feature_size=64,
        num_layers=NUM_LAYERS,
    ).to(device)

    print(f"Tap stage_chs      : {decoder.stage_chs}")
    print(f"Decoder params   : {decoder.count_parameters():,}\n")

    # Simulate backbone output: list of NUM_LAYERS tensors [B, N, D]
    dummy_tokens = [
        torch.randn(BATCH, EMBED_DIM, GRID_H, GRID_W, device=device)
        for _ in range(NUM_LAYERS)
    ]
    dummy_input = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Forward pass
    t0 = time.time()
    with torch.no_grad():
        logits = decoder(dummy_tokens, dummy_input, image_hw=(IMAGE_SIZE, IMAGE_SIZE))
    elapsed = time.time() - t0

    print(f"Input  tokens : {NUM_LAYERS} × [{BATCH}, {N_TOKENS}, {EMBED_DIM}]")
    print(f"Output logits : {list(logits.shape)}")
    print(f"Forward time  : {elapsed*1000:.1f} ms  (no backbone, decoder only)")

    expected = (BATCH, NUM_CLS, IMAGE_SIZE, IMAGE_SIZE)
    assert logits.shape == expected, f"Shape mismatch: {logits.shape} != {expected}"
    print("\n✓ All assertions passed.")

