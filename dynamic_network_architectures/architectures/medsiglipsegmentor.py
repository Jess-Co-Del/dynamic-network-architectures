"""
MedSigLip Semantic Segmentation Decoders
======================================

A collection of nn.Module implementations that extract intermediate feature maps
from a frozen (or fine-tunable) MedSigLip backbone and decode them into per-pixel
semantic segmentation predictions.

Architecture context
--------------------
MedSigLip is a Vision Transformer (ViT). Unlike CNNs, all transformer layers
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
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from dynamic_network_architectures.building_blocks.vit_adapter_probes import *
from dynamic_network_architectures.building_blocks.mask2formerdecoder import Mask2Former, Mask2FormerDecoderHF
from dynamic_network_architectures.building_blocks.unetr_decoder import UNETRDecoder, UNETRFPNDecoder
from dynamic_network_architectures.building_blocks.segformer_decoder import SegFormerDecoder
from dynamic_network_architectures.building_blocks.vitadapter import ViTAdapter


# =============================================================================
# 1. FEATURE EXTRACTOR — wraps HuggingFace MedSigLip and returns intermediate maps
# =============================================================================

class MedSigLipFeatureExtractor(nn.Module):
    """
    Wraps a HuggingFace ``MedSigLip2Model`` and returns patch-token feature maps
    from selected intermediate transformer layers.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``""google/medsiglip-448","``
    layer_indices : list[int]
        0-based indices of the transformer layers whose hidden states to return.
    freeze_backbone : bool
        If True, all backbone parameters are frozen (no gradient).
    """

    def __init__(
        self,
        model_name: str = "google/medsiglip-448",
        layer_indices: List[int] = [1, 10, 19, 26],
        freeze_backbone: bool = True,
        image_size:      int  = 448,
        adapter: str = 'last',
    ):
        super().__init__()
        # ── 1 Backbone ─────────────────────────────────────────────────────
        full_model = AutoModel.from_pretrained(model_name)
        self.backbone = full_model.vision_model
        del full_model  # free memory of the text encoder

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.layer_indices = sorted(layer_indices)
        vision_config = self.backbone.config
        self.hidden_dim = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.grid_size = image_size // self.patch_size
        self.num_layers = vision_config.num_hidden_layers

        # ── 2. Adapter strategy config ────────────────────────────────────
        self.adapter = adapter
        if adapter == 'last':
            self.layer_indices = [self.num_layers - 1]
        else:
            self.layer_indices = sorted(layer_indices)
            for idx in self.layer_indices:
                if idx < 0 or idx >= self.num_layers+1:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {self.num_layers}]"
                    )

        # ── Built-in normalization (SigLIP: map [0,1] → [-1,1]) ─────────
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
        )

    def _preprocess(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Tile grayscale → 3-ch, then normalize [0,1] → [-1,1]."""
        x = pixel_values.clone().float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Per-sample rescaling to [0, 1] if not already in range
        # Shape: (B, C, H, W) — reduce over C, H, W independently per sample
        vmin = x.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)   # (B,1,1,1)
        vmax = x.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)   # (B,1,1,1)

        out_of_range = ((vmin < 0) | (vmax > 1)).squeeze()
        if out_of_range.any():
            denom = (vmax - vmin).clamp(min=1e-8)
            x = (x - vmin) / denom
 
        # MedSigLip normalisation: [0,1] → [-1,1]
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int, int]:
        """
        Parameters
        ----------
        pixel_values : torch.Tensor
            Shape (B, 3, 448, 448), values normalised to (-1, 1).
            Use SiglipProcessor or:  pixel_values = img / 127.5 - 1.0
 
        Returns
        -------
        List[torch.Tensor]
            One (B, D, h, w) tensor per extracted block, shallow -> deep.
            For default medsiglip-448: shape is (B, 1152, 32, 32) per map.
        """
        B, C, H, W = pixel_values.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # ── 0. Run MedSigLip preprocessing-─────────────────────────────────
        pixel_values = self._preprocess(pixel_values)
        # ── 1. Run MedSigLip encoding   -───────────────────────────────────
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        
        features = []

        for idx in self.layer_indices:
            # +1 because hidden_states[0] is the embedding layer output
            hs = hidden_states[idx]  # (B, seq_len, D)

            assert hs.shape[1] == h * w, (
            f"Expected seq_len={h * w} (grid {h}x{w}) but got {hs.shape[1]}. "
            "Check image_size / patch_size in model config."
        )
        
            # Reshape to spatial grid: (B, h*w, D) -> (B, D, h, w)
            feat = hs.permute(0, 2, 1).reshape(B, -1, h, w)
            features.append(feat)

        return features

    # Expose sub-sequences of blocks so the adapter can interleave
    # Prepared for interleaved layer input injection
    def get_block_groups(self) -> List[nn.Sequential]:
        """
        Split transformer blocks into `num_groups` roughly equal groups.
        Dont forget to pass by:
            - first inputs go through self.backbone.embeddings(x)
            - after these blocks outputs fo through self.backbone.layernorm(f_vit)
        """
        blocks = list(self.backbone.encoder.layers)
        num_groups = len(self.layer_indices)

        groups = []
        groups.append(nn.Sequential(*blocks[0: self.layer_indices[0]+1]))
        for i in range(num_groups-1):
            groups.append(nn.Sequential(*blocks[self.layer_indices[i]+1:self.layer_indices[i+1]+1]))

        final_groups = nn.Sequential(
            self.backbone.post_layernorm,
            self.backbone.head
        )

        return groups, final_groups

# =============================================================================
# 3. FULL SEGMENTATION MODEL (wrapper)
# =============================================================================

class MedSigLipSegmenter(nn.Module):
    """
    End-to-end MedSigLip segmentation model that combines the feature extractor
    and any decoder from above.

    Parameters
    ----------
    extractor : MedSigLipFeatureExtractor
        Produces a list of intermediate feature maps from MedSigLip.
    decoder : nn.Module
        Any decoder that accepts ``List[Tensor]`` and returns ``(B, C, h, w)``.
    image_size : int or tuple[int, int]
        Expected input image size (used to upsample logits to original resolution).
        If None, no final upsampling is done.
    """

    def __init__(
        self,
        extractor: MedSigLipFeatureExtractor,
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
        self.linear_probe = linear_probe
        print(f'MedSigLipSegmenter class: Encoder {extractor.__class__}, adapter {adapter.__class__} + linear_probe {linear_probe}, decoder {decoder.__class__}')
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

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
        logits = self.extractor(pixel_values)

        if self.adapter is not None:
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
        logits = [scale.to(torch.float32) for scale in logits]

        return self.decoder(logits, pixel_values.to(torch.float32))


# =============================================================================
# 4. FACTORY / CONVENIENCE CONSTRUCTORS
# =============================================================================

def build_segmenter(
    input_channels:   int  = 1,
    model_name: str = "google/medsiglip-448",
    decoder_type: str = "none",
    adapter_type: str = "concat",
    num_classes: int = 2,
    layer_indices: Optional[List[int]] = None,
    freeze_backbone: bool = True,
    image_size: int = 448,
    deep_supervision: bool = False,
    **decoder_kwargs,
) -> MedSigLipSegmenter:
    """
    Factory function to build a complete MedSigLip segmentation model.

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
        Whether to freeze the MedSigLip backbone.
    image_size : int
        Input image size (assumes square images).
    **decoder_kwargs
        Additional keyword arguments forwarded to the decoder constructor.
    """
    # Infer default layer indices based on model depth
    _depth_defaults = {
        12: [2, 5, 8, 12],
        24: [4, 11, 17, 24],
        40: [9, 19, 29, 40],
        27: [6, 13, 19, 27]
    }

    extractor = MedSigLipFeatureExtractor(
        model_name=model_name,
        layer_indices=layer_indices or [0, 10, 19, 27],  # safe default for 27-layer
        freeze_backbone=freeze_backbone,
        adapter='all'
    )
    hidden_dim = extractor.hidden_dim
    num_layers = len(extractor.layer_indices)

    # Preparing Adapter 
    if adapter_type == 'vitadapter':
        extractor = ViTAdapter(
            backbone=extractor,
            in_channels=3,
            num_heads=18,  # For ViT 1152
            num_interactions=4,
            out_channels=256,
            image_size=image_size,
            use_cls=False
        )
    # Override layer indices after extractor creation if needed
    # if layer_indices is None:
    #     depth = extractor.num_layers
    #     if depth in _depth_defaults:
    #         extractor.layer_indices = _depth_defaults[depth]

    adapter_map = {
        "linear": lambda: LinearDecoder(
            hidden_dim=hidden_dim,
            **decoder_kwargs,
        ),
        "concat": lambda: MultiScaleConcatDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "pyramid": lambda: MultiScalePyramidDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            image_size=image_size,
            **decoder_kwargs,
        ),
        "fpn": lambda: FPNLikeDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **decoder_kwargs,
        ),
        "upernet": lambda: UPerNetLikeDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            **decoder_kwargs,
        ),
        "upernetpupinterp": lambda: UPerNetInterpPUPAdapter(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_fusion='add',
            **decoder_kwargs,
        ),
        "upernetpupconv": lambda: UPerNetConvPUPAdapter(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_fusion='add',
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
        'vitadapter': None,
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
        "mask2formerhf": lambda: Mask2FormerDecoderHF(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_classes,
            image_size=image_size,
            **decoder_kwargs
        ),
        "unetr": lambda: UNETRDecoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
            num_classes=num_classes,
        ),
        "unetrfpn": lambda: UNETRFPNDecoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
            num_classes=num_classes,
        ),
        "segformer": lambda: SegFormerDecoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
            num_classes=num_classes,
        ),
        "none": None
    }

    if adapter_type not in adapter_map:
        raise ValueError(
            f"Unknown decoder_type '{adapter_map}'. "
            f"Choose from: {list(adapter_map.keys())}"
        )
    elif adapter_type in ['none', 'vitadapter']:
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

    return MedSigLipSegmenter(
        extractor=extractor,
        adapter=adapter,
        decoder=decoder,
        linear_probe=True if decoder is None else False,
        image_size=image_size,
        hidden_dim=hidden_dim*4 if adapter_type == 'concat' else hidden_dim,
        num_classes=num_classes
    )


# =============================================================================
# 5. QUICK SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """
    Run a shape-check smoke test with random tensors (does NOT download the
    actual MedSigLip weights — uses random initialization for speed).
    """
    print("=" * 70)
    print("MedSigLip Segmentation Decoders — Smoke Test (random weights)")
    print("=" * 70)

    def count_trainable_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    IMAGE_SHAPE = (1, 3, 448, 448)
    NUM_CLASSES = 2
    HIDDEN_DIM = 1152
    NUM_LAYERS = 4

    # Build extractor once (frozen backbone)
    backbone = MedSigLipFeatureExtractor(
        model_name="google/medsiglip-448",
        layer_indices=[2, 5, 8, 11],
        adapter="all",
        freeze_backbone=True,
    )
    backbone.eval()

    dummy = torch.randn(*IMAGE_SHAPE)
    with torch.no_grad():
        features = backbone(dummy)

    print(f"\nInput shape : {IMAGE_SHAPE}")
    print(f"Patch grid  : {features[0].shape[-1]}×{features[0].shape[-2]}  (patch_size=14, input=448)")
    print(f"Num features: {len(features)}  each {tuple(features[0].shape)}")
    print(f"Backbone trainable params: {count_trainable_params(backbone):,}  (frozen → 0)\n")
    print(f"{'Decoder':<35} {'Trainable Params':>18}  {'Output Shape'}")
    print("-" * 70)

    for name in ['unetrfpn', 'segformer', 'mask2former']:

        full_model = build_segmenter(adapter_type='vitadapter', decoder_type=name, input_channels=3)
        features = full_model(dummy)
        print(f"Num features: {len(features)}  each {tuple(features[0].shape)}")
        n_params = count_trainable_params(full_model)
        print(f"  {name:<33} {n_params:>18,}  {tuple(features.shape)}")
    print("-" * 70)
