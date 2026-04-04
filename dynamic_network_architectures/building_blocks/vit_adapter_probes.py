# =============================================================================
# 2. DECODER STRATEGIES
# =============================================================================
from typing import List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_conv_blocks import ConvBNReLU


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
      MedSigLip features are strong enough and a complex decoder adds
      unnecessary parameters.
    * Fastest to train and least memory.
    * Established in the MedSigLip paper and repository (BNHead).

    Advantages
    ----------
    + Minimal learnable parameters (just BN + 1×1 conv).
    + Avoids overfitting on small datasets.
    + Clean ablation baseline.

    Disadvantages
    -------------
    - Ignores multi-layer feature richness; only uses the final layer.
    - No multi-scale reasoning.
    - Output resolution limited to patch-grid resolution (e.g. 32x32 for 448
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
        #self.bn = nn.BatchNorm2d(hidden_dim)
        #self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

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
        # x = self.bn(x)
        return x  # self.classifier(x)


# --------------------------------------------------------------------------
# DECODER V2: Multi-Scale Concatenation Decoder
# --------------------------------------------------------------------------

class MultiScaleConcatDecoder(nn.Module):
    """
    **Strategy: concatenate features from multiple layers, then decode.**

    This mirrors the *resize_concat* approach used in the official MedSigLip
    segmentation notebook. Features from N selected layers are (optionally)
    resized to match the spatial size of the largest, concatenated along the
    channel dimension, and passed through a small convolutional head.

    Why use this
    ------------
    * Combines information from multiple depths (shallow texture + deep
      semantic) without complex fusion logic.
    * Well-validated: the official MedSigLip repo ships pre-trained BNHead
      weights using exactly this approach.

    Advantages
    ----------
    + Simple and proven effective with MedSigLip features.
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

        # if intermediate_dim is not None:
        #     self.head = nn.Sequential(
        #         nn.Conv2d(concat_dim, intermediate_dim, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(intermediate_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(intermediate_dim, hidden_dim, kernel_size=1),
        #     )
        # else:
        #     self.head = nn.Conv2d(concat_dim, hidden_dim, kernel_size=1)

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
        # x = self.bn(x)
        return x # self.head(x)


class _Reassemble(nn.Module):
    """
    One reassemble block for a single tap.
    Converts (B, 1+L, D) → (B, out_channels, H_out, W_out) where
    H_out, W_out = h * scale, w * scale.
    scale > 1 → upsample (deconv), scale < 1 → downsample (conv stride).
    """
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        scale: float,                          # 0.5, 1, 2, 4
        readout: Literal["ignore", "add", "none"] = "none",
    ):
        super().__init__()
        self.readout = readout
 
        # Channel projection
        self.proj = nn.Conv2d(embed_dim, out_channels, 1, bias=False)
 
        # Spatial resampler
        if scale == 4:
            self.resample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=4)
        elif scale == 2:
            self.resample = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        elif scale == 1:
            self.resample = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        elif scale == 0.5:
            self.resample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported scale: {scale}")
 
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.readout == "ignore":
            patch_tokens = tokens[:, 1:, :]  # drop CLS
        elif self.readout == "add":
            patch_tokens = tokens[:, 1:, :] + tokens[:, 0:1, :]
        else:
            patch_tokens = tokens
 
        x = self.proj(patch_tokens)           # (B, out_ch, h, w)
        x = self.resample(x)                  # (B, out_ch, H_out, W_out)
        return x


class MultiScalePyramidDecoder(nn.Module):
    """
    **Strategy: concatenate features from multiple layers, then decode.**

    This mirrors the *resize_concat* approach used in the official MedSigLip
    segmentation notebook. Features from N selected layers are (optionally)
    resized to match the spatial size of the largest, concatenated along the
    channel dimension, and passed through a small convolutional head.

    Why use this
    ------------
    * Combines information from multiple depths (shallow texture + deep
      semantic) without complex fusion logic.
    * Well-validated: the official MedSigLip repo ships pre-trained BNHead
      weights using exactly this approach.

    Advantages
    ----------
    + Simple and proven effective with MedSigLip features.
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
        image_size: int,
        # readout: Literal["ignore", "add", "none"] = "none",
        # has_cls: bool = False,
        ppm_bins: Tuple[int, ...] = (1, 2, 3, 6),
        fpn_dim: int = 256,
    ):
        super().__init__()
        concat_dim = fpn_dim * num_layers
        self.image_size = image_size

        self.ppm = PyramidPoolingModule(hidden_dim, hidden_dim // 4, bins=ppm_bins)

        # Lateral 1×1 convolutions
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, fpn_dim, kernel_size=1, bias=False) for _ in range(num_layers)]
        )

        # FPN smoothing convolutions
        # self.smooth_convs = nn.ModuleList(
        #     [ConvBNReLU(fpn_dim, fpn_dim) for _ in range(num_layers)]
        # )

        # Final fusion head
        self.fusion = nn.Sequential(
            ConvBNReLU(concat_dim, hidden_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3),
        )
        self.pup = nn.ModuleList(
            [nn.Sequential(
            ConvBNReLU(hidden_dim, hidden_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        ) for _ in range(3)]
        )

    def forward(
        self, 
        features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w)

        Returns
        -------
        logits : (B, num_classes, h, w)
        """
        # All features already share the same spatial resolution.
        #maps = [block(vit_stage) for block, vit_stage in zip(self.reassemble, features)]
        features = list(features)  # make mutable copy
        features[-1] = self.ppm(features[-1])

        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Same resolution, so no upsample needed — just add
            laterals[i] = laterals[i] + laterals[i + 1]

        # Concatenate and classify
        out = torch.cat(laterals, dim=1)
        out = self.fusion(out)
        for pup, patch_scale in zip(self.pup, [8, 4, 2]):
            out = F.interpolate(
                    out, size=(self.image_size//patch_scale),
                    mode='bilinear', align_corners=False
            )
            out = pup(out)
        return out
 

# --------------------------------------------------------------------------
# DECODER V3: FPN-Like Decoder
# --------------------------------------------------------------------------

class FPNLikeDecoder(nn.Module):
    """
    **Strategy: Feature Pyramid Network adapted for ViT (same-resolution).**

    A classic FPN assumes multi-scale encoder outputs. Since all MedSigLip layers
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
            nn.Conv2d(fpn_dim, hidden_dim, kernel_size=1),
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
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, fpn_dim, kernel_size=1, bias=False) for _ in range(num_layers)]
        )

        # FPN smoothing convolutions
        self.fpn_convs = nn.ModuleList(
            [ConvBNReLU(fpn_dim, fpn_dim) for _ in range(num_layers)]
        )

        # Final fusion head
        self.fusion = nn.Sequential(
            ConvBNReLU(fpn_dim * num_layers, fpn_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, hidden_dim, kernel_size=1),
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
# DECODER V5: UPerNet-Interpolation PUP Decoder
# --------------------------------------------------------------------------

class UPerNetInterpPUPAdapter(nn.Module):
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
        skip_fusion: str = 'add',  # [add | concat]
        ppm_bins: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.skip_fusion = skip_fusion

        # PPM on the deepest feature
        self.ppm = PyramidPoolingModule(hidden_dim, hidden_dim // 4, bins=ppm_bins)

        # Lateral 1×1 convolutions
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, fpn_dim, kernel_size=1, bias=False) for _ in range(num_layers)]
        )

        # FPN smoothing convolutions
        self.smooth_convs = nn.ModuleList(
            [ConvBNReLU(
                fpn_dim*2 if skip_fusion == 'concat' else fpn_dim,
                hidden_dim, kernel_size=3)
            for _ in range(num_layers)]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w).
                   Ordered shallow → deep.

        Returns
        -------
        multiscale_features :  list[Tensor], (B, num_classes, scale_h, scale_h) h -> h * 2, h * 4, h * 8
        """
        # Apply PPM to the deepest feature
        features = list(features)  # make mutable copy
        features[-1] = self.ppm(features[-1])

        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i, factor in zip(range(len(laterals) - 2, -1, -1), [2, 4, 8]):
            if self.skip_fusion == 'add':
                laterals[i] = F.interpolate(
                    laterals[i], scale_factor=factor,
                    mode='bilinear', align_corners=True
                ) + \
                F.interpolate(
                    laterals[i + 1], scale_factor=2,
                    mode='bilinear', align_corners=True)
            else:
                    # Same resolution, concat
                laterals[i] = torch.cat([
                    F.interpolate(
                        laterals[i], scale_factor=factor,
                        mode='bilinear', align_corners=True
                    ),
                    F.interpolate(
                        laterals[i + 1], scale_factor=2,
                        mode='bilinear', align_corners=True)
                    ],
                    dim=1
                )

        # Smoothing
        smoothed = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]

        return smoothed


# --------------------------------------------------------------------------
# DECODER V6: UPerNet-ConvPUP Decoder
# --------------------------------------------------------------------------

class UPerNetConvPUPAdapter(nn.Module):
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
        skip_fusion: str = 'add',  # [add | concat]
        ppm_bins: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.skip_fusion = skip_fusion

        # PPM on the deepest feature
        self.ppm = PyramidPoolingModule(hidden_dim, hidden_dim // 4, bins=ppm_bins)

        # Lateral 1×1 convolutions
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, fpn_dim, kernel_size=1, bias=False) for _ in range(num_layers)]
        )

        self.upsampling_up = nn.ModuleList(
            [nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2) for _ in range(num_layers)]
        )
        self.upsampling_skips =  nn.ModuleList(
            [nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2*(i+1)) for i in range(num_layers)]
        )

        # FPN smoothing convolutions
        self.smooth_convs = nn.ModuleList(
            [ConvBNReLU(
                fpn_dim*2 if skip_fusion == 'concat' else fpn_dim,
                hidden_dim, kernel_size=3)
            for _ in range(num_layers)]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list[Tensor], each (B, hidden_dim, h, w).
                   Ordered shallow → deep.

        Returns
        -------
        multiscale_features :  list[Tensor], (B, num_classes, scale_h, scale_h) h -> h * 2, h * 4, h * 8
        """
        # Apply PPM to the deepest feature
        features = list(features)  # make mutable copy
        features[-1] = self.ppm(features[-1])

        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i, factor in zip(range(len(laterals) - 2, -1, -1), range(0, len(laterals))):
            if self.skip_fusion == 'add':
                laterals[i] = self.upsampling_up[factor](laterals[i]) + \
                    self.upsampling_skips[factor](laterals[i+1])
            else:
                laterals[i] = torch.cat([
                    self.upsampling_up[factor](laterals[i]),
                    self.upsampling_skips[factor](laterals[i+1])
                    ],
                    dim=1
                )

        # Smoothing
        smoothed = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]

        return smoothed


# --------------------------------------------------------------------------
# DECODER VN: Progressive Upsample Decoder
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

        #self.classifier = nn.Conv2d(decoder_dim, hidden_dim, kernel_size=1)

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

        return x  # self.classifier(x)


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
      very strong (MedSigLip features are highly correlated across nearby layers).

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
            ConvBNReLU(fusion_dim, hidden_dim, kernel_size=1),
            #nn.Conv2d(fusion_dim, num_classes, kernel_size=1),
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
