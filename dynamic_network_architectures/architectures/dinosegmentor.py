"""
DINOv2 Semantic Segmentation Decoders
======================================

A collection of nn.Module implementations that extract intermediate feature maps
from a frozen (or fine-tunable) DINOv2 backbone and decode them into per-pixel
semantic segmentation predictions.

Architecture context
--------------------
DINOv2 is a Vision Transformer (ViT). Unlike CNNs, all transformer layers
operate at the **same spatial resolution** — there is no built-in feature
pyramid with decreasing spatial resolution. For a ViT with patch_size=14 and
input 518×518, every hidden layer produces patch tokens of shape:

    (batch, num_patches, hidden_dim)     e.g. (B, 1369, 384) for ViT-S/14

where num_patches = (H // patch_size) * (W // patch_size) = 37 * 37 = 1369.

The [CLS] token is prepended (index 0), so the full sequence length is
num_patches + 1. For models *with registers*, additional register tokens
are also prepended after CLS.

Key design decision: which layers to tap
-----------------------------------------
Because all layers share the same resolution, we cannot directly mimic FPN-style
multi-scale fusion as done with CNNs (ResNet stages at 1/4, 1/8, 1/16, 1/32).
Instead, we treat layers at different *depths* as providing different levels of
semantic abstraction (shallow = low-level / texture, deep = high-level /
semantic), all at the same spatial grid.

The default layer indices used here follow established practice:
    ViT-S (12 layers): layers [2, 5, 8, 11]   (evenly spaced, 0-indexed)
    ViT-B (12 layers): layers [2, 5, 8, 11]
    ViT-L (24 layers): layers [4, 11, 17, 23]
    ViT-g (40 layers): layers [9, 19, 29, 39]

Dependencies
------------
    pip install torch transformers

Usage
-----
    from dinov2_segmentation_decoders import (
        DINOv2FeatureExtractor,
        LinearDecoder,
        MultiScaleConcatDecoder,
        FPNLikeDecoder,
        UPerNetLikeDecoder,
        ProgressiveUpsampleDecoder,
    )

    # Instantiate feature extractor (wraps HuggingFace DINOv2)
    extractor = DINOv2FeatureExtractor(
        model_name="facebook/dinov2-small",
        layer_indices=[2, 5, 8, 11],
        freeze_backbone=True,
    )

    # Pick a decoder
    decoder = LinearDecoder(
        hidden_dim=extractor.hidden_dim,
        num_classes=21,
    )

    # Full segmentation model
    model = DINOv2Segmenter(extractor, decoder, image_size=518)
    logits = model(pixel_values)  # (B, num_classes, H, W)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# =============================================================================
# 1. FEATURE EXTRACTOR — wraps HuggingFace DINOv2 and returns intermediate maps
# =============================================================================

class DINOv2FeatureExtractor(nn.Module):
    """
    Wraps a HuggingFace ``Dinov2Model`` and returns patch-token feature maps
    from selected intermediate transformer layers.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"facebook/dinov2-small"``,
        ``"facebook/dinov2-base"``, ``"facebook/dinov2-large"``,
        ``"facebook/dinov2-giant"``.
    layer_indices : list[int]
        0-based indices of the transformer layers whose hidden states to return.
    freeze_backbone : bool
        If True, all backbone parameters are frozen (no gradient).
    """

    def __init__(
        self,
        input_channels:   int  = 1,
        model_name: str = "facebook/dinov2-large",
        layer_indices: List[int] = [1, 10, 20, 23],  # Total 25 blocks
        adapter: str = 'last',
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.patch_size: int = self.backbone.config.patch_size
        self.hidden_dim: int = self.backbone.config.hidden_size
        self.num_layers: int = self.backbone.config.num_hidden_layers

        # ── Adapter strategy config ──────────────────────────────────────
        self.adapter = adapter
        if adapter == 'last':
            self.layer_indices = [-1]
        else:
            self.layer_indices = sorted(layer_indices)
            for idx in self.layer_indices:
                if idx < 0 or idx >= self.num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {self.num_layers - 1}]"
                    )

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
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

        # Normalize to range [0,1] if needed
        if pixel_values.max() > 1. or  pixel_values.max() < 0.:
            pixel_values = (pixel_values - pixel_values.min())
            pixel_values /= pixel_values.max()

        return (pixel_values - self.pixel_mean) / self.pixel_std

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int, int]:
        """
        Parameters
        ----------
        pixel_values : Tensor of shape (B, 3, H, W)

        Returns
        -------
        features : list[Tensor]
            Each tensor has shape (B, hidden_dim, h, w) where
            h = H // patch_size, w = W // patch_size.
        h : int
            Spatial height of the patch grid.
        w : int
            Spatial width of the patch grid.
        """
        B, C, H, W = pixel_values.shape
        h = H // self.patch_size
        w = W // self.patch_size
        # ── 0. Run DINOv2 preprocessing-───────────────────────────────────
        pixel_values = self._preprocess(pixel_values)
        # ── 1. Run DINOv2 encoding   -───────────────────────────────────
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # index 0 = embedding output, index i = output of layer i
        # Each tensor shape: (B, seq_len, hidden_dim)
        # seq_len = 1 (CLS) + num_patches [+ num_register_tokens]

        hidden_states = outputs.hidden_states  # tuple of (B, seq_len, D)

        features = []

        for idx in self.layer_indices:
            # +1 because hidden_states[0] is the embedding layer output
            hs = hidden_states[idx]  # (B, seq_len, D)

            # Remove CLS token (always at position 0)
            patch_tokens = hs[:, 1:, :]  # (B, seq_len - 1, D)

            # If the model has register tokens, they come right after CLS
            # and before the actual patch tokens. We need to remove them.
            num_register_tokens = getattr(
                self.backbone.config,
                "num_register_tokens", 0
            )
            if num_register_tokens > 0:
                patch_tokens = patch_tokens[:, num_register_tokens:, :]

            # Truncate to exactly h * w tokens (safety for padding edge cases)
            #patch_tokens = patch_tokens[:, : h * w, :]

            # Reshape to spatial grid: (B, h*w, D) -> (B, D, h, w)
            feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, h, w)
            features.append(feat)

        return features, h, w


# =============================================================================
# 2. DECODER STRATEGIES
# =============================================================================

# ---- Shared utility blocks ----

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU (a ubiquitous building block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module from PSPNet (Zhao et al., 2017).
    Captures multi-scale context via adaptive average pooling at several bin sizes.
    """

    def __init__(self, in_dim: int, reduction_dim: int, bins: Tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for bin_size in bins:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    ConvBNReLU(in_dim, reduction_dim, kernel_size=1, padding=0),
                )
            )
        # After concat: in_dim + len(bins) * reduction_dim
        self.bottleneck = ConvBNReLU(
            in_dim + len(bins) * reduction_dim, in_dim, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pyramids = [x]
        for stage in self.stages:
            pooled = stage(x)
            pyramids.append(
                F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            )
        out = torch.cat(pyramids, dim=1)
        return self.bottleneck(out)


# --------------------------------------------------------------------------
# DECODER V1: Linear Decoder (single-layer baseline)
# --------------------------------------------------------------------------

class LinearDecoder(nn.Module):
    """
    **Strategy: single linear projection on the last selected layer's features.**

    This is the simplest possible decoder. It takes the output from a single
    transformer layer (by default the last one in the selected list), applies
    batch normalization, then a 1×1 convolution to map to num_classes.

    Why use this
    ------------
    * Baseline / sanity check — if a linear head already works well, the
      DINOv2 features are strong enough and a complex decoder adds
      unnecessary parameters.
    * Fastest to train and least memory.
    * Established in the DINOv2 paper and repository (BNHead).

    Advantages
    ----------
    + Minimal learnable parameters (just BN + 1×1 conv).
    + Avoids overfitting on small datasets.
    + Clean ablation baseline.

    Disadvantages
    -------------
    - Ignores multi-layer feature richness; only uses the final layer.
    - No multi-scale reasoning.
    - Output resolution limited to patch-grid resolution (e.g. 37×37 for 518px
      input with patch_size=14). Needs bilinear upsample to full resolution.

    Parameters
    ----------
    hidden_dim : int
        Channel dimension of the backbone features.
    num_classes : int
        Number of semantic segmentation classes.
    use_layer_idx : int
        Which feature in the list to use (default -1 = last).
    """

    def __init__(self, hidden_dim: int, num_classes: int, use_layer_idx: int = -1):
        super().__init__()
        self.use_layer_idx = use_layer_idx
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor]
            Each of shape (B, hidden_dim, h, w).

        Returns
        -------
        logits : Tensor of shape (B, num_classes, h, w)
        """
        x = features[self.use_layer_idx]
        x = self.bn(x)
        return self.classifier(x)


# --------------------------------------------------------------------------
# DECODER V2: Multi-Scale Concatenation Decoder
# --------------------------------------------------------------------------

class MultiScaleConcatDecoder(nn.Module):
    """
    **Strategy: concatenate features from multiple layers, then decode.**

    This mirrors the *resize_concat* approach used in the official DINOv2
    segmentation notebook. Features from N selected layers are (optionally)
    resized to match the spatial size of the largest, concatenated along the
    channel dimension, and passed through a small convolutional head.

    Why use this
    ------------
    * Combines information from multiple depths (shallow texture + deep
      semantic) without complex fusion logic.
    * Well-validated: the official DINOv2 repo ships pre-trained BNHead
      weights using exactly this approach.

    Advantages
    ----------
    + Simple and proven effective with DINOv2 features.
    + Multi-layer fusion captures both low- and high-level cues.
    + Still relatively few parameters (BN + 1-2 conv layers).

    Disadvantages
    -------------
    - Channel dimension grows linearly with number of layers (4 layers ×
      384 = 1536 channels for ViT-S). Can be memory-heavy for large models.
    - All layers share the same spatial resolution, so "multi-scale" here
      refers only to semantic depth, not spatial scale.
    - No explicit interaction/refinement between feature levels.

    Parameters
    ----------
    hidden_dim : int
        Channel dimension of each backbone feature map.
    num_layers : int
        Number of feature maps that will be concatenated.
    num_classes : int
        Number of segmentation classes.
    intermediate_dim : int or None
        If provided, a bottleneck conv reduces channels before classification.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        intermediate_dim: Optional[int] = None,
    ):
        super().__init__()
        concat_dim = hidden_dim * num_layers
        self.bn = nn.BatchNorm2d(concat_dim)

        if intermediate_dim is not None:
            self.head = nn.Sequential(
                nn.Conv2d(concat_dim, intermediate_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(intermediate_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(intermediate_dim, num_classes, kernel_size=1),
            )
        else:
            self.head = nn.Conv2d(concat_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w)

        Returns
        -------
        logits : (B, num_classes, h, w)
        """
        # All features already share the same spatial resolution.
        x = torch.cat(features, dim=1)  # (B, hidden_dim * num_layers, h, w)
        x = self.bn(x)
        return self.head(x)


# --------------------------------------------------------------------------
# DECODER V3: FPN-Like Decoder
# --------------------------------------------------------------------------

class FPNLikeDecoder(nn.Module):
    """
    **Strategy: Feature Pyramid Network adapted for ViT (same-resolution).**

    A classic FPN assumes multi-scale encoder outputs. Since all DINOv2 layers
    output at the same resolution, we first *project* each layer to a common
    channel width (``fpn_dim``), then perform top-down lateral additions (from
    the deepest/most-semantic layer to the shallowest). Finally, all levels are
    concatenated and passed through a classification head.

    This creates a *learned interaction* between features at different depths,
    which plain concatenation does not provide.

    Why use this
    ------------
    * Well-understood architecture from object detection / segmentation.
    * Top-down pathway lets deep semantic information flow to shallow layers.
    * Widely adopted for ViT-based segmentation (e.g. ViTDet, Swin + FPN).

    Advantages
    ----------
    + Explicit top-down refinement of shallow features with deep semantics.
    + Controlled parameter budget (fpn_dim is typically 256).
    + Each level can also be supervised independently (auxiliary losses).

    Disadvantages
    -------------
    - More complex than concat; more hyperparameters (fpn_dim).
    - Top-down pathway is sequential — deeper errors propagate downward.
    - Without true multi-scale spatial features, the FPN adds depth-wise
      interaction but no genuine spatial-scale reasoning.

    Parameters
    ----------
    hidden_dim : int
        Channel dimension of backbone features.
    num_layers : int
        Number of intermediate feature maps.
    fpn_dim : int
        Internal channel width for the FPN.
    num_classes : int
        Number of segmentation classes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        fpn_dim: int = 256,
        num_classes: int = 21,
    ):
        super().__init__()
        # Lateral 1×1 projections (one per layer)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, fpn_dim, kernel_size=1) for _ in range(num_layers)]
        )
        # Smoothing 3×3 convs after addition
        self.smooth_convs = nn.ModuleList(
            [ConvBNReLU(fpn_dim, fpn_dim, kernel_size=3, padding=1) for _ in range(num_layers)]
        )
        # Final fusion: concat all levels → classify
        self.fusion = nn.Sequential(
            ConvBNReLU(fpn_dim * num_layers, fpn_dim, kernel_size=3, padding=1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w).
                   Ordered shallow → deep.

        Returns
        -------
        logits : (B, num_classes, h, w)
        """
        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway: from deepest to shallowest
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + laterals[i + 1]

        # Smoothing
        smoothed = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]

        # Concatenate all levels and classify
        out = torch.cat(smoothed, dim=1)
        return self.fusion(out)


# --------------------------------------------------------------------------
# DECODER V4: UPerNet-Like Decoder
# --------------------------------------------------------------------------

class UPerNetLikeDecoder(nn.Module):
    """
    **Strategy: Unified Perceptual Parsing Network adapted for ViT.**

    UPerNet (Xiao et al., 2018) combines an FPN with a Pyramid Pooling Module
    (PPM) on the deepest feature map, capturing global context before fusing
    multi-level features. This is the decoder used by BEiT, Swin, and many
    other ViT-based segmentation models in mmsegmentation.

    Pipeline:
    1. Apply PPM to the deepest feature map → rich global context.
    2. FPN top-down pathway merges PPM-enhanced deep features with shallower
       lateral features.
    3. All levels are upsampled to the finest resolution, concatenated, and
       passed through a classification head.

    Why use this
    ------------
    * State-of-the-art decoder for ViT backbones in semantic segmentation.
    * PPM captures scene-level context (important for large objects / stuff).
    * The combination of PPM + FPN is strictly more expressive than either
      alone.

    Advantages
    ----------
    + Multi-scale context via PPM.
    + Top-down feature refinement via FPN pathway.
    + Best overall accuracy on most benchmarks with ViT backbones.
    + Well-proven architecture with extensive ablation studies.

    Disadvantages
    -------------
    - Heaviest decoder in this collection (most parameters and compute).
    - More hyperparameters to tune (PPM bins, FPN dim, etc.).
    - May overfit on small datasets — consider freezing backbone.

    Parameters
    ----------
    hidden_dim : int
        Backbone feature channel width.
    num_layers : int
        Number of feature maps from the extractor.
    fpn_dim : int
        Internal FPN channel width.
    num_classes : int
        Number of segmentation classes.
    ppm_bins : tuple[int, ...]
        Bin sizes for the Pyramid Pooling Module.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        fpn_dim: int = 256,
        num_classes: int = 21,
        ppm_bins: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        # PPM on the deepest feature
        self.ppm = PyramidPoolingModule(hidden_dim, hidden_dim // 4, bins=ppm_bins)

        # Lateral 1×1 convolutions
        self.lateral_convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim  # PPM output still has hidden_dim channels
            self.lateral_convs.append(
                nn.Conv2d(in_ch, fpn_dim, kernel_size=1, bias=False)
            )

        # FPN smoothing convolutions
        self.fpn_convs = nn.ModuleList(
            [ConvBNReLU(fpn_dim, fpn_dim) for _ in range(num_layers)]
        )

        # Final fusion head
        self.fusion = nn.Sequential(
            ConvBNReLU(fpn_dim * num_layers, fpn_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w).
                   Ordered shallow → deep.

        Returns
        -------
        logits : (B, num_classes, h, w)
        """
        # Apply PPM to the deepest feature
        features = list(features)  # make mutable copy
        features[-1] = self.ppm(features[-1])

        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Same resolution, so no upsample needed — just add
            laterals[i] = laterals[i] + laterals[i + 1]

        # Smoothing
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Concatenate and classify
        out = torch.cat(fpn_outs, dim=1)
        return self.fusion(out)


# --------------------------------------------------------------------------
# DECODER V5: Progressive Upsample Decoder
# --------------------------------------------------------------------------

class ProgressiveUpsampleDecoder(nn.Module):
    """
    **Strategy: iteratively fuse features and upsample to full resolution.**

    This decoder introduces *artificial spatial hierarchy* from the
    same-resolution ViT features. Starting from the deepest features, it
    progressively upsamples by 2× at each stage while fusing in shallower
    features via skip connections. This mimics the U-Net decoding path but
    with learned upsampling.

    Pipeline (for 4 layers at patch-grid resolution h×w):
        Stage 4: deep features → conv block                    → h × w
        Stage 3: upsample 2× + concat(layer 3 feat) → conv    → 2h × 2w
        Stage 2: upsample 2× + concat(layer 2 feat) → conv    → 4h × 4w
        Stage 1: upsample 2× + concat(layer 1 feat) → conv    → 8h × 8w
        Final:   1×1 classification head                       → 8h × 8w

    The result is a *higher-resolution output* than the other decoders without
    requiring a separate bilinear upsample to the original image size (though
    a final upsample may still be needed for exact pixel alignment).

    Why use this
    ------------
    * Generates genuinely higher-resolution segmentation maps.
    * Skip connections preserve spatial detail from shallower layers.
    * Gradual upsampling avoids checkerboard artifacts.

    Advantages
    ----------
    + Output resolution much closer to input resolution.
    + U-Net–style skip connections are well-studied and effective.
    + Each stage can be individually supervised for deep supervision.

    Disadvantages
    -------------
    - Requires bilinear interpolation of ViT features to create the
      artificial multi-scale hierarchy — may introduce artifacts.
    - More parameters than linear or concat decoders.
    - Assumes the number of selected layers matches the number of
      progressive stages.

    Parameters
    ----------
    hidden_dim : int
        Backbone feature channel width.
    num_layers : int
        Number of intermediate feature maps (= number of progressive stages).
    decoder_dim : int
        Internal channel width within each decoder stage.
    num_classes : int
        Number of segmentation classes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        decoder_dim: int = 256,
        num_classes: int = 21,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Project each backbone feature to decoder_dim
        self.input_projs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, decoder_dim, kernel_size=1) for _ in range(num_layers)]
        )

        # Decoder stages (processed deepest → shallowest)
        # After the first stage, each receives concat(upsampled_prev, skip) = 2 * decoder_dim
        self.stages = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # Deepest: no skip concat, just refine
                self.stages.append(
                    nn.Sequential(
                        ConvBNReLU(decoder_dim, decoder_dim),
                        ConvBNReLU(decoder_dim, decoder_dim),
                    )
                )
            else:
                # Fuse upsampled previous + current skip
                self.stages.append(
                    nn.Sequential(
                        ConvBNReLU(decoder_dim * 2, decoder_dim),
                        ConvBNReLU(decoder_dim, decoder_dim),
                    )
                )

        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w).
                   Ordered shallow → deep.

        Returns
        -------
        logits : (B, num_classes, H_out, W_out)
            where H_out = h * 2^(num_layers - 1), W_out = w * 2^(num_layers - 1)
        """
        # Project all features to decoder_dim
        projected = [proj(f) for proj, f in zip(self.input_projs, features)]

        # Process deepest first (last in list)
        x = self.stages[0](projected[-1])  # (B, decoder_dim, h, w)

        # Progressive upsample + fuse
        for i in range(1, self.num_layers):
            # Upsample by 2×
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Get the corresponding skip connection (going from deep to shallow)
            skip_idx = self.num_layers - 1 - i
            skip = projected[skip_idx]

            # Resize skip to match upsampled x
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

            # Concat and process
            x = torch.cat([x, skip], dim=1)
            x = self.stages[i](x)

        return self.classifier(x)


# --------------------------------------------------------------------------
# DECODER V6: Multi-Head Attention Fusion Decoder
# --------------------------------------------------------------------------

class AttentionFusionDecoder(nn.Module):
    """
    **Strategy: learnable cross-attention between feature levels, then decode.**

    Instead of simple concatenation or top-down addition, this decoder uses
    a lightweight multi-head self-attention mechanism to let features from
    different layers attend to each other. Each layer's feature map is
    projected to a shared dimension, and attention is computed across the
    *layer dimension* (not spatial), allowing the model to learn which
    combination of layers is most informative for each spatial location.

    Pipeline:
    1. Project each layer's features to ``fusion_dim``.
    2. At each spatial position, stack the N layer features into a sequence
       of length N.
    3. Apply multi-head self-attention across the layer dimension.
    4. Aggregate (mean pool across layers) → single feature per position.
    5. 1×1 classification head.

    Why use this
    ------------
    * Data-driven fusion: the attention weights are learned, so the model
      can dynamically weight layers differently depending on image content.
    * Motivated by research showing that optimal layer combination varies
      per image region (e.g., textures benefit from shallow features, object
      boundaries from deep features).

    Advantages
    ----------
    + Adaptive, content-dependent layer fusion.
    + Lightweight if fusion_dim and num_heads are small.
    + Can reveal (via attention weights) which layers matter where.

    Disadvantages
    -------------
    - Attention over N=4 layers is cheap, but scales O(N²) if many layers.
    - Adds architectural complexity; harder to debug than concat.
    - May not improve much over concat when backbone features are already
      very strong (DINOv2 features are highly correlated across nearby layers).

    Parameters
    ----------
    hidden_dim : int
        Backbone feature channel width.
    num_layers : int
        Number of feature maps.
    fusion_dim : int
        Internal dimension for the attention mechanism.
    num_heads : int
        Number of attention heads.
    num_classes : int
        Segmentation classes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        fusion_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 21,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.fusion_dim = fusion_dim

        # Per-layer projection
        self.projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, fusion_dim, kernel_size=1) for _ in range(num_layers)]
        )

        # Cross-layer attention (applied per spatial position)
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)

        # Classification head
        self.classifier = nn.Sequential(
            ConvBNReLU(fusion_dim, fusion_dim),
            nn.Conv2d(fusion_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w)

        Returns
        -------
        logits : (B, num_classes, h, w)
        """
        B, _, h, w = features[0].shape

        # Project each layer: (B, fusion_dim, h, w) → flatten spatial
        projected = []
        for proj, feat in zip(self.projections, features):
            p = proj(feat)  # (B, fusion_dim, h, w)
            projected.append(p)

        # Stack layers at each spatial position
        # Reshape: (B, N, fusion_dim, h, w) → (B * h * w, N, fusion_dim)
        stacked = torch.stack(projected, dim=1)  # (B, N, D, h, w)
        stacked = stacked.permute(0, 3, 4, 1, 2).reshape(B * h * w, self.num_layers, self.fusion_dim)

        # Self-attention across layers
        attn_out, _ = self.attn(stacked, stacked, stacked)  # (B*h*w, N, D)
        attn_out = self.norm(attn_out + stacked)  # residual + norm

        # Aggregate across layers: mean pool
        fused = attn_out.mean(dim=1)  # (B*h*w, D)

        # Reshape back to spatial
        fused = fused.reshape(B, h, w, self.fusion_dim).permute(0, 3, 1, 2)  # (B, D, h, w)

        return self.classifier(fused)


# =============================================================================
# 3. FULL SEGMENTATION MODEL (wrapper)
# =============================================================================

class DINOv2Segmenter(nn.Module):
    """
    End-to-end DINOv2 segmentation model that combines the feature extractor
    and any decoder from above.

    Parameters
    ----------
    extractor : DINOv2FeatureExtractor
        Produces a list of intermediate feature maps from DINOv2.
    decoder : nn.Module
        Any decoder that accepts ``List[Tensor]`` and returns ``(B, C, h, w)``.
    image_size : int or tuple[int, int]
        Expected input image size (used to upsample logits to original resolution).
        If None, no final upsampling is done.
    """

    def __init__(
        self,
        extractor: DINOv2FeatureExtractor,
        decoder: nn.Module,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.extractor = extractor
        self.decoder = decoder
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : Tensor (B, 3, H, W)

        Returns
        -------
        logits : Tensor (B, num_classes, H, W)
            Upsampled to the original image size if ``image_size`` was set.
        """
        features, h, w = self.extractor(pixel_values)
        logits = self.decoder(features)  # (B, num_classes, h', w')

        # Upsample logits to the original image resolution
        target_size = self.image_size or pixel_values.shape[2:]
        if logits.shape[2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


# =============================================================================
# 4. FACTORY / CONVENIENCE CONSTRUCTORS
# =============================================================================

def build_segmenter(
    input_channels:   int  = 1,
    model_name: str = "facebook/dinov2-small",
    decoder_type: str = "concat",
    num_classes: int = 21,
    layer_indices: Optional[List[int]] = None,
    freeze_backbone: bool = True,
    image_size: int = 518,
    deep_supervision: bool = False,
    **decoder_kwargs,
) -> DINOv2Segmenter:
    """
    Factory function to build a complete DINOv2 segmentation model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    decoder_type : str
        One of: ``"linear"``, ``"concat"``, ``"fpn"``, ``"upernet"``,
        ``"progressive"``, ``"attention"``.
    num_classes : int
        Number of segmentation classes.
    layer_indices : list[int] or None
        Transformer layer indices to extract. Defaults to evenly-spaced 4 layers.
    freeze_backbone : bool
        Whether to freeze the DINOv2 backbone.
    image_size : int
        Input image size (assumes square images).
    **decoder_kwargs
        Additional keyword arguments forwarded to the decoder constructor.
    """
    # Infer default layer indices based on model depth
    _depth_defaults = {
        12: [2, 5, 8, 11],
        24: [4, 11, 17, 23],
        40: [9, 19, 29, 39],
    }

    extractor = DINOv2FeatureExtractor(
        input_channels=input_channels,
        model_name=model_name,
        layer_indices=layer_indices or [2, 5, 8, 11],  # safe default for 12-layer
        freeze_backbone=freeze_backbone,
        adapter=decoder_type
    )

    # Override layer indices after extractor creation if needed
    if layer_indices is None:
        depth = extractor.num_layers
        if depth in _depth_defaults:
            extractor.layer_indices = _depth_defaults[depth]

    hidden_dim = extractor.hidden_dim
    num_layers = len(extractor.layer_indices)

    decoder_map = {
        "linear": lambda: LinearDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "concat": lambda: MultiScaleConcatDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "fpn": lambda: FPNLikeDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "upernet": lambda: UPerNetLikeDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "progressive": lambda: ProgressiveUpsampleDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "attention": lambda: AttentionFusionDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
    }

    if decoder_type not in decoder_map:
        raise ValueError(
            f"Unknown decoder_type '{decoder_type}'. "
            f"Choose from: {list(decoder_map.keys())}"
        )

    decoder = decoder_map[decoder_type]()

    return DINOv2Segmenter(
        extractor=extractor,
        decoder=decoder,
        image_size=image_size,
    )


# =============================================================================
# 5. QUICK SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """
    Run a shape-check smoke test with random tensors (does NOT download the
    actual DINOv2 weights — uses random initialization for speed).
    """
    print("=" * 70)
    print("DINOv2 Segmentation Decoders — Smoke Test (random weights)")
    print("=" * 70)

    # Create a small DINOv2 config for fast testing
    backbone = DINOv2FeatureExtractor(adapter='concat')

    # Mock the extractor
    dummy = torch.randn(1, 3, 224, 224)
    B, C, H, W = dummy.shape
    h, w = H // 14, W // 14  # 37, 37
    num_classes = 2

    # Simulate features
    features = backbone(dummy)

    decoders = {
        "LinearDecoder": LinearDecoder(384, num_classes),
        "MultiScaleConcatDecoder": MultiScaleConcatDecoder(384, 4, num_classes, intermediate_dim=256),
        "FPNLikeDecoder": FPNLikeDecoder(384, 4, fpn_dim=256, num_classes=num_classes),
        "UPerNetLikeDecoder": UPerNetLikeDecoder(384, 4, fpn_dim=256, num_classes=num_classes),
        "ProgressiveUpsampleDecoder": ProgressiveUpsampleDecoder(384, 4, decoder_dim=128, num_classes=num_classes),
        "AttentionFusionDecoder": AttentionFusionDecoder(384, 4, fusion_dim=128, num_heads=4, num_classes=num_classes),
    }

    for name, decoder in decoders.items():
        decoder.eval()
        with torch.no_grad():
            out = decoder(features)
        print(f"  {name:35s}  input: 4×(B,384,{h},{w})  →  output: {tuple(out.shape)}")

    print("\nAll smoke tests passed.")
