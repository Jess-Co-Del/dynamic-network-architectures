"""
UPerNet Decoder for Frozen ViT Backbones (2D Slice-Wise Liver Tumor Segmentation)
===================================================================================
Adapts UPerNet (Xiao et al., ECCV 2018) to work with same-resolution patch tokens
from frozen ViT backbones (DINOv2, I-JEPA, SigLIP, etc.) that produce a flat
sequence of tokens at constant spatial resolution across all layers.

Background on UPerNet
---------------------
The original UPerNet couples two components:
  1. Feature Pyramid Network (FPN): merges multi-scale CNN feature maps
     top-down with lateral connections, producing {P2, P3, P4, P5}.
  2. Pyramid Pooling Module (PPM): applied to the deepest feature map only;
     pools at multiple scales and concatenates to capture global context.

The outputs of PPM and all FPN levels are fused, then upsampled to a fixed
resolution and classified by a single conv head.

Adaptation for same-resolution ViT features
--------------------------------------------
A frozen ViT has NO spatial hierarchy — every layer produces tokens at the
same grid (H/P × W/P). We recover a pseudo-hierarchy the same way as in
DINOv2's linear probing paper and mmsegmentation's BEiT/DINOv2 configs:

  - Tap four intermediate layers at depths {L/4, L/2, 3L/4, L}.
  - Project each to the same channel width (feature_size) via 1×1 conv.
  - Reshape each from [B, N, D] → [B, C, H', W'].
  - Feed these four same-resolution maps into the FPN.

Because all four maps share the same spatial size, the FPN lateral path still
works — but there is no 2× downsampling between levels, so no upsampling is
needed in the top-down path. The FPN here is essentially a multi-level feature
fusion with learned weights, not a spatial pyramid. The PPM on the deepest tap
still provides meaningful global context pooling.

Architecture diagram (ViT-L/14, 518×518 image → grid 37×37)
-------------------------------------------------------------

  Frozen ViT backbone
  ┌──────────────────────────────────────────────────────────┐
  │  z_L/4   z_L/2   z_3L/4   z_L                           │
  │  [B,N,D] [B,N,D] [B,N,D] [B,N,D]                        │
  └──────────────────────────────────────────────────────────┘
       │        │        │        │
   Proj+Reshape (1×1 conv, BN, ReLU → C channels, H'×W' each)
       │        │        │        │
      c0       c1       c2       c3   ← all [B, C, H', W'] = [B,C,37,37]
                                 │
                           ┌─────┴──────┐
                           │    PPM     │  Pool at {1,2,3,6} → upsample → cat → conv
                           └─────┬──────┘
                                 │ ppm_out [B, C, H', W']
                                 │
  FPN top-down pass (no upsampling needed — all same resolution):
       c3+ppm_out → lateral3 → p3
       p3 + c2    → lateral2 → p2
       p2 + c1    → lateral1 → p1
       p1 + c0    → lateral0 → p0

  FPN outputs {p0, p1, p2, p3} all at [B, C, H', W']
  Upsample all to p0 size (same — no-op) then concatenate:
       fused: [B, 4C, H', W']

  Fusion head conv: [B, 4C, H', W'] → [B, C, H', W']
  Bilinear upsample to [B, C, H, W]
  Segmentation head (1×1 conv): [B, num_classes, H, W]

References
----------
- UPerNet: Xiao et al., "Unified Perceptual Parsing", ECCV 2018.
- PPM: Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017.
- DINOv2 linear / UPerNet configs: Oquab et al. 2023 + mmseg DINOv2 configs.
- BEiT + UPerNet: mmsegmentation configs/beit.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_conv_blocks import ConvBNReLU


# ---------------------------------------------------------------------------
# Pyramid Pooling Module (PPM)
# ---------------------------------------------------------------------------

class PyramidPoolingModule(nn.Module):
    """
    PPM from PSPNet (Zhao et al., CVPR 2017), used unchanged in UPerNet.
 
    Pools the feature map at multiple scales, upsamples each back to the
    original size, concatenates with the original features, and projects
    back to `out_ch` channels.
 
    Args:
        in_ch     : Input channel count.
        out_ch    : Output channel count (usually same as in_ch).
        pool_sizes: Spatial sizes of the adaptive average pooling kernels.
                    Default {1, 2, 3, 6} as in the original PSPNet/UPerNet.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        hidden = in_ch // len(pool_sizes)  # bottleneck channels per scale
 
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])
 
        # Fuse: original + all pooled branches
        fuse_in = in_ch + hidden * len(pool_sizes)
        self.fuse = ConvBNReLU(fuse_in, out_ch, kernel_size=3)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        branches = [x]
        for stage in self.stages:
            pooled = stage(x)
            # Upsample back to input spatial size
            pooled = F.interpolate(pooled, size=(H, W),
                                   mode="bilinear", align_corners=False)
            branches.append(pooled)
        return self.fuse(torch.cat(branches, dim=1))


# ---------------------------------------------------------------------------
# FPN for same-resolution feature maps
# ---------------------------------------------------------------------------

class SameResolutionFPN(nn.Module):
    """
    Feature Pyramid Network adapted for same-resolution inputs.

    Standard FPN assumes feature maps at different spatial scales and uses
    2× nearest-neighbour upsampling in the top-down path. When all inputs
    share the same resolution (as with flat ViT tokens), no upsampling is
    needed. The top-down path reduces to element-wise addition of lateral
    projections — essentially learned multi-level feature fusion.

    Args:
        in_ch  : Channel count of all input feature maps (uniform after projection).
        out_ch : Channel count of every FPN output level.
        num_levels: Number of tap levels (default 4).
    """
    def __init__(self, in_ch: int, out_ch: int, num_levels: int = 4):
        super().__init__()
        # Lateral 1×1 convs: compress each level to out_ch
        self.laterals = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0)
            for _ in range(num_levels)
        ])
        # Output 3×3 convs: smooth the summed features
        self.outputs = nn.ModuleList([
            ConvBNReLU(out_ch, out_ch, kernel_size=3)
            for _ in range(num_levels)
        ])
        self.num_levels = num_levels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: list of [B, in_ch, H, W] — ordered shallowest to deepest.
                      The deepest entry (features[-1]) acts as the FPN top.
        Returns:
            List of [B, out_ch, H, W], same ordering (shallowest first).
        """
        assert len(features) == self.num_levels

        # Lateral projections
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down fusion (index num_levels-1 is the deepest/top)
        # Because all resolutions are identical, addition is trivially aligned.
        for i in range(self.num_levels - 1, 0, -1):
            # Upsample top-down feature to match lower level
            # (no-op when same resolution, but kept for correctness if ever
            # the inputs come from a true spatial pyramid)
            td = F.interpolate(laterals[i],
                               size=laterals[i - 1].shape[2:],
                               mode="nearest")
            laterals[i - 1] = laterals[i - 1] + td

        # Output smoothing
        return [out_conv(lat) for out_conv, lat in zip(self.outputs, laterals)]


# ---------------------------------------------------------------------------
# Main UPerNet Decoder
# ---------------------------------------------------------------------------

class UPerNetDecoder(nn.Module):
    """
    UPerNet decoder for same-resolution ViT patch tokens.

    Args:
        hidden_dim   : ViT hidden dimension (e.g. 768 ViT-B, 1024 ViT-L).
        patch_size  : ViT patch size in pixels (e.g. 14 DINOv2, 16 I-JEPA).
        image_size  : Input image size in pixels. Accepts int (square) or (H,W).
        num_classes : Number of segmentation output classes.
        feature_size: Uniform channel width C used throughout the decoder.
                      Default 256 (same as original UPerNet).
        pool_sizes  : PPM pooling scales. Default (1, 2, 3, 6).
        num_layers  : Total ViT depth. Used to auto-select tap indices.
        dropout     : Dropout probability before the segmentation head.

    Forward inputs:
        layer_tokens : List of [B, N, D] tensors, one per ViT layer.
                       Class token must be removed before passing in.
        image_hw     : (H, W) to upsample the final output to.
                       Defaults to self.image_size.

    Forward output:
        logits : [B, num_classes, H, W]

    Example
    -------
        decoder = UPerNetDecoder(
            hidden_dim=1024, patch_size=14, image_size=518,
            num_classes=3, feature_size=256, num_layers=24,
        )
        logits = decoder(all_layer_tokens, image_hw=(518, 518))
    """

    def __init__(
        self,
        hidden_dim: int,
        #input_channels: int,
        image_size,
        num_classes: int,
        feature_size: int = 256,
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── spatial grid ──────────────────────────────────────────────────
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        C = feature_size

        # ── token → spatial projections ───────────────────────────────────
        # All four taps projected to the same C channels before FPN.
        self.token_projs = nn.ModuleList([
            ConvBNReLU(hidden_dim, C) for _ in range(num_layers)
        ])

        # ── PPM on the deepest tap ─────────────────────────────────────────
        self.ppm = PyramidPoolingModule(C, C, pool_sizes=pool_sizes)

        # ── FPN ───────────────────────────────────────────────────────────
        # Input to the FPN: [c0, c1, c2, ppm(c3)] — all [B, C, H', W']
        self.fpn = SameResolutionFPN(in_ch=C, out_ch=C, num_levels=num_layers)

        # ── Fusion head ───────────────────────────────────────────────────
        # Concatenate all 4 FPN outputs → fuse to C channels
        self.fusion_head = nn.Sequential(
            ConvBNReLU(4 * C, C, kernel_size=3),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

        # ── Segmentation head ─────────────────────────────────────────────
        self.seg_head = nn.Conv2d(C, num_classes, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
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
            layer_tokens : list of [B, N, D], one per ViT layer.
            image_hw     : target output (H, W).
        Returns:
            logits [B, num_classes, H, W]
        """
        if image_hw is None:
            image_hw = self.image_size

        # ── token → spatial feature maps ──────────────────────────────────
        # c0..c3: each [B, C, grid_h, grid_w]
        c0, c1, c2, c3 = [proj(t) for proj, t in
                           zip(self.token_projs, layer_tokens)]

        # ── PPM on deepest features ────────────────────────────────────────
        c3_ppm = self.ppm(c3)           # [B, C, grid_h, grid_w]

        # ── FPN ───────────────────────────────────────────────────────────
        # Feed [shallowest, ..., ppm(deepest)] — deepest is the FPN "top"
        fpn_outs = self.fpn([c0, c1, c2, c3_ppm])
        # fpn_outs: list of 4 × [B, C, grid_h, grid_w]

        # ── Upsample all FPN levels to the same size then fuse ────────────
        # In the standard UPerNet the coarsest level is upsampled to P2 size.
        # Here all levels are already the same size (grid_hw), so the
        # interpolation is a guaranteed no-op but kept for robustness.
        target_hw = fpn_outs[0].shape[2:]
        aligned = [
            F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
            if p.shape[2:] != target_hw else p
            for p in fpn_outs
        ]

        fused = self.fusion_head(torch.cat(aligned, dim=1))   # [B, C, grid_h, grid_w]

        # ── Upsample to image resolution ──────────────────────────────────
        fused = F.interpolate(fused, size=image_hw,
                              mode="bilinear", align_corners=False)

        return self.seg_head(fused)     # [B, num_classes, H, W]

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Returns trainable parameter counts per sub-module."""
        def count(m): return sum(p.numel() for p in m.parameters()
                                 if p.requires_grad)
        return {
            "token_projs" : count(self.token_projs),
            "ppm"         : count(self.ppm),
            "fpn"         : count(self.fpn),
            "fusion_head" : count(self.fusion_head),
            "seg_head"    : count(self.seg_head),
            "total"       : self.count_parameters(),
        }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def upernet_decoder_for_dinov2(
    model_size: str = "large",
    num_classes: int = 3,
    feature_size: int = 256,
    image_size: int = 518,
    pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
) -> UPerNetDecoder:
    """UPerNetDecoder pre-configured for DINOv2 (patch_size=14)."""
    configs = {
        "small" : dict(embed_dim=384,  num_layers=12),
        "base"  : dict(embed_dim=768,  num_layers=12),
        "large" : dict(embed_dim=1024, num_layers=24),
        "giant" : dict(embed_dim=1536, num_layers=40),
    }
    cfg = configs[model_size]
    return UPerNetDecoder(
        hidden_dim=cfg["embed_dim"],
        image_size=image_size, num_classes=num_classes,
        feature_size=feature_size, pool_sizes=pool_sizes,
        num_layers=cfg["num_layers"],
    )


def upernet_decoder_for_ijepa(
    num_classes: int = 3,
    feature_size: int = 256,
    image_size: int = 224,
    pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
) -> UPerNetDecoder:
    """UPerNetDecoder pre-configured for I-JEPA ViT-H/16."""
    return UPerNetDecoder(
        hidden_dim=1280,
        image_size=image_size, num_classes=num_classes,
        feature_size=feature_size, pool_sizes=pool_sizes,
        num_layers=32,
    )


# ---------------------------------------------------------------------------
# Integration wrapper
# ---------------------------------------------------------------------------

class UPerNetSegmenter(nn.Module):
    """
    Thin wrapper combining a frozen ViT backbone with UPerNetDecoder.

    Backbone must return all intermediate layer tokens as a list of
    [B, N, D] tensors. Set strip_cls_token=True if the backbone prepends
    a [CLS] token to N.
    """
    def __init__(self, backbone: nn.Module, decoder: UPerNetDecoder):
        super().__init__()
        self.backbone = backbone
        self.decoder  = decoder
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_hw = (x.shape[2], x.shape[3])
        with torch.no_grad():
            layer_tokens = self.backbone(x)
        return self.decoder(layer_tokens, image_hw=image_hw)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("UPerNetDecoder — sanity check")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── DINOv2-L / 14 ─────────────────────────────────────────────────────
    BATCH      = 2
    IMAGE_SIZE = 224
    PATCH_SIZE = 14
    EMBED_DIM  = 1024
    NUM_LAYERS = 24
    NUM_CLS    = 3
    GRID_H = GRID_W = IMAGE_SIZE // PATCH_SIZE   # = 37
    N_TOKENS   = GRID_W * GRID_H               # 1369

    decoder = upernet_decoder_for_dinov2(
        model_size="large", num_classes=NUM_CLS,
        feature_size=256, image_size=IMAGE_SIZE,
    ).to(device)

    bd = decoder.parameter_breakdown()
    print(f"\nParameter breakdown:")
    for k, v in bd.items():
        print(f"  {k:<16}: {v:>10,}")

    dummy = [torch.randn(BATCH,EMBED_DIM, GRID_H, GRID_W, device=device)
             for _ in range(NUM_LAYERS)]
    dummy_input = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)


    t0 = time.time()
    with torch.no_grad():
        logits = decoder(dummy, image_hw=(IMAGE_SIZE, IMAGE_SIZE))
    print(f"\nForward time : {(time.time()-t0)*1000:.1f} ms  (decoder only)")
    print(f"Output shape : {list(logits.shape)}")

    assert logits.shape == (BATCH, NUM_CLS, IMAGE_SIZE, IMAGE_SIZE)
    print("✓ DINOv2-L config OK.\n")

    # ── I-JEPA ViT-H/16 ───────────────────────────────────────────────────
    print("--- I-JEPA ViT-H/16 ---")
    decoder_h = upernet_decoder_for_ijepa(num_classes=3, image_size=512).to(device)
    N2 = (512 // 16) ** 2
    dummy_h = [torch.randn(BATCH, 1280, 512, 512, device=device) for _ in range(NUM_LAYERS)]
    with torch.no_grad():
        out_h = decoder_h(dummy_h, image_hw=(512, 512))
    print(f"Params       : {decoder_h.count_parameters():,}")
    print(f"Output shape : {list(out_h.shape)}")
    assert out_h.shape == (BATCH, 3, 512, 512)
    print("✓ I-JEPA config OK.")