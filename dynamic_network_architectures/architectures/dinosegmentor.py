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
input 224x224, every hidden layer produces patch tokens of shape:

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
    model = DINOv2Segmenter(extractor, decoder, image_size=224)
    logits = model(pixel_values)  # (B, num_classes, H, W)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from dynamic_network_architectures.building_blocks.vit_adapter_probes import *
from dynamic_network_architectures.building_blocks.mask2formerdecoder import Mask2Former


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
        if pixel_values.shape[1] == 1:
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
        adapter: nn.Module,
        decoder: nn.Module,
        linear_probe: bool = True,
        image_size: Optional[int] = None,
        hidden_dim: int = 256,
        num_classes: int = 1,
    ):
        super().__init__()
        self.extractor = extractor
        self.adapter = adapter
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        print(f'DINOv2Segmenter class: Encoder {extractor.__class__}, decoder {decoder.__class__}')

        if linear_probe:
            self.linear_probe = linear_probe
            self.decoder = nn.Sequential(
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
            )
        else:
            self.decoder = decoder

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
        logits, _, _ = self.extractor(pixel_values)

        if self.adapter:
            logits = self.adapter(logits)  # (B, num_classes, h', w')

            if self.linear_probe:
                # Upsample logits to the original image resolution
                target_size = self.image_size or pixel_values.shape[2:]
                if logits.shape[2:] != target_size:
                    logits = F.interpolate(
                        logits,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )

        return self.decoder(logits)


# =============================================================================
# 4. FACTORY / CONVENIENCE CONSTRUCTORS
# =============================================================================

def build_segmenter(
    input_channels:   int  = 1,
    model_name: str = "facebook/dinov2-large",
    decoder_type: str = "none",
    adapter_type: str = "concat",
    num_classes: int = 1,
    layer_indices: Optional[List[int]] = None,
    freeze_backbone: bool = True,
    image_size: int = 224,
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
        layer_indices=layer_indices or [4, 11, 17, 23],  # safe default for 12-layer
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

    adapter_map = {
        "linear": lambda: LinearDecoder(
            hidden_dim=hidden_dim,
            num_classes=hidden_dim,
            **decoder_kwargs,
        ),
        "concat": lambda: MultiScaleConcatDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=hidden_dim,
            **decoder_kwargs,
        ),
        "fpn": lambda: FPNLikeDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=hidden_dim,
            **decoder_kwargs,
        ),
        "upernet": lambda: UPerNetLikeDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=hidden_dim,
            **decoder_kwargs,
        ),
        "progressive": lambda: ProgressiveUpsampleDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=hidden_dim,
            decoder_dim=hidden_dim,
            **decoder_kwargs,
        ),
        "attention": lambda: AttentionFusionDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=hidden_dim,
            **decoder_kwargs,
        ),
        "none": None
    }

    decoder_map = {
        "mask2former": lambda: Mask2Former(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            patch_size=extractor.patch_size,
            image_size=image_size,
            **decoder_kwargs
        ),
        "none": None
    }

    if adapter_type not in adapter_map:
        raise ValueError(
            f"Unknown decoder_type '{adapter_map}'. "
            f"Choose from: {list(adapter_map.keys())}"
        )
    elif adapter_type == 'none':
        adapter = None
    else:
        adapter = adapter_map[adapter_type]()

    if decoder_type not in decoder_map:
        raise ValueError(
            f"Unknown decoder_type '{decoder_type}'. "
            f"Choose from: {list(decoder_map.keys())}"
        )
    elif decoder_type == 'none':
        decoder = None
    else:
        decoder = decoder_map[decoder_type]()

    return DINOv2Segmenter(
        extractor=extractor,
        adapter=adapter,
        decoder=decoder,
        linear_probe=False,
        image_size=image_size,
        hidden_dim=hidden_dim if decoder_type == 'concat' else hidden_dim,
        num_classes=num_classes
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

    def count_trainable_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    IMAGE_SHAPE = (1, 3, 224, 224)
    NUM_CLASSES = 2
    HIDDEN_DIM = 1024  # ViT-L
    NUM_LAYERS = 4

    # Build extractor once (frozen backbone)
    backbone = DINOv2FeatureExtractor(
        model_name="facebook/dinov2-large",
        layer_indices=[2, 5, 8, 11],
        adapter="concat",
        freeze_backbone=True,
    )
    backbone.eval()

    dummy = torch.randn(*IMAGE_SHAPE)
    with torch.no_grad():
        features, h, w = backbone(dummy)

    print(f"\nInput shape : {IMAGE_SHAPE}")
    print(f"Patch grid  : {h}×{w}  (patch_size=14, input=224)")
    print(f"Num features: {len(features)}  each {tuple(features[0].shape)}")
    print(f"Backbone trainable params: {count_trainable_params(backbone):,}  (frozen → 0)\n")
    print(f"{'Decoder':<35} {'Trainable Params':>18}  {'Output Shape'}")
    print("-" * 70)

    decoders = {
        "LinearDecoder": LinearDecoder(
            hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES
        ),
        "MultiScaleConcatDecoder": MultiScaleConcatDecoder(
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES, intermediate_dim=256
        ),
        "FPNLikeDecoder": FPNLikeDecoder(
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            fpn_dim=256, num_classes=NUM_CLASSES
        ),
        "UPerNetLikeDecoder": UPerNetLikeDecoder(
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            fpn_dim=256, num_classes=NUM_CLASSES
        ),
        "ProgressiveUpsampleDecoder": ProgressiveUpsampleDecoder(
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            decoder_dim=256, num_classes=NUM_CLASSES
        ),
        "AttentionFusionDecoder": AttentionFusionDecoder(
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            fusion_dim=256, num_heads=4, num_classes=NUM_CLASSES
        ),
    }

    for name, decoder in decoders.items():
        decoder.eval()
        with torch.no_grad():
            out = decoder(features)
        n_params = count_trainable_params(decoder)
        print(f"  {name:<33} {n_params:>18,}  {tuple(out.shape)}")

    print("\nAll smoke tests passed ✓")
