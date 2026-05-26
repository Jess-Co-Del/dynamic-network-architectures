from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.vit_adapter_probes import *
from dynamic_network_architectures.building_blocks.mask2formerdecoder import Mask2Former, Mask2FormerDecoderHF
from dynamic_network_architectures.building_blocks.unetr_decoder import UNETRDecoder, UNETRFPNDecoder
from dynamic_network_architectures.building_blocks.segformer_decoder import SegFormerDecoder
from dynamic_network_architectures.building_blocks.upernet_decoder import UPerNetDecoder
from dynamic_network_architectures.building_blocks.vitadapter import ViTAdapter


class MedDINOv3FeatureExtractor(nn.Module):
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
        layer_indices: List[int] = [1, 10, 20, 23],  # Total 24 blocks
        adapter: str = 'last',
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels

        # ── Backbone ─────────────────────────────────────────────────────
        from dinov3.models.vision_transformer import vit_base

        # Initialize backbone
        self.backbone = vit_base(
            drop_path_rate=0.0, layerscale_init=1.0e-05,
            n_storage_tokens=4,
            qkv_bias = False, mask_k_bias= True
        )

        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="ricklisz123/MedDINOv3-ViTB-16-CT-3M", filename="model.pth")

        chkpt = torch.load(path, map_location="cpu")
        self.backbone.load_state_dict(chkpt, strict=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.patch_size: int = 16
        self.hidden_dim: int = 768
        self.num_layers: int = 12

        # ── Adapter strategy config ──────────────────────────────────────
        self.adapter = adapter
        if adapter == 'last':
            self.layer_indices = [-1]
        else:
            self.layer_indices = sorted(layer_indices)
            for idx in self.layer_indices:
                if idx < 0 or idx > self.num_layers:
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
        hidden_states = self.backbone.get_intermediate_layers(
            pixel_values,
            n=12  # Its a Vit-base with 12 intermediate layers
        )

        # hidden_states is a tuple of (num_layers) tensors
        # Each tensor shape: (B, seq_len, hidden_dim)
        # seq_len = 1 (CLS) + num_patches [+ num_register_tokens]
        features = []

        for idx in self.layer_indices:
            # +1 because hidden_states[0] is the embedding layer output
            patch_tokens = hidden_states[idx]  # (B, seq_len, D)

            # Truncate to exactly h * w tokens (safety for padding edge cases)
            #patch_tokens = patch_tokens[:, : h * w, :]

            # Reshape to spatial grid: (B, h*w, D) -> (B, D, h, w)
            feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, h, w)
            features.append(feat)

        return features

    # Expose sub-sequences of blocks so the adapter can interleave
    # Prepared for interleaved layer input injection
    def get_block_groups(self) -> List[nn.Sequential]:
        """
        Split transformer blocks into `num_groups` roughly equal groups.
        Dont forget to pass by:
            - first inputs go through self.backbone.patch_embed(x) and self.backbone.rope_embed(x)
            - after these blocks outputs fo through self.backbone.norm(f_vit)
        """
        blocks = list(self.backbone.blocks)
        num_groups = len(self.layer_indices)

        groups = []
        groups.append(nn.Sequential(*blocks[0: self.layer_indices[0]+1]))
        for i in range(num_groups-1):
            groups.append(nn.Sequential(*blocks[self.layer_indices[i]+1:self.layer_indices[i+1]+1]))

        final_groups = nn.Sequential(
            self.backbone.norm
        )
        return groups, final_groups

# =============================================================================
# 3. FULL SEGMENTATION MODEL (wrapper)
# =============================================================================

class ViTSegmenter(nn.Module):
    """
    End-to-end ViT model segmentation model that combines the feature extractor
    and any decoder from above.

    Parameters
    ----------
    extractor : ViT model
        Produces a list of intermediate feature maps from ViT model.
    decoder : nn.Module
        Any decoder that accepts ``List[Tensor]`` and returns ``(B, C, h, w)``.
    image_size : int or tuple[int, int]
        Expected input image size (used to upsample logits to original resolution).
        If None, no final upsampling is done.
    """

    def __init__(
        self,
        extractor,  # FeatureExtractor
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
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        print(f'ViT model class: Encoder {extractor.__class__}, adapter {adapter.__class__}, decoder {decoder.__class__}')

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
        logits= self.extractor(pixel_values)

        if self.adapter is not None:
            logits = self.adapter(logits)  # (B, num_classes, h', w')

        logits = [scale.to(torch.float32) for scale in logits]

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

        return self.decoder(logits, pixel_values)


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
):
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
    extractor = MedDINOv3FeatureExtractor(
        layer_indices=[2, 5, 8, 11],
        adapter="concat",
        freeze_backbone=True,
    )

    hidden_dim = extractor.hidden_dim
    num_layers = len(extractor.layer_indices)

    # Preparing Adapter 
    if adapter_type == 'vitadapter':
        extractor = ViTAdapter(
            backbone=extractor,
            in_channels=3,
            num_heads=12,  # For ViT 1024
            num_interactions=4,
            out_channels=256,
            image_size=image_size,
            use_cls=False
        )
    adapter_map = {'vitadapter': None, 'none': None}
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
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            input_channels=input_channels,
            image_size=image_size,
        ),
        "upernet": lambda: UPerNetDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pool_sizes=(1,2,3,4),
            #feature_size=hidden_dim,
            image_size=image_size,
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

    return ViTSegmenter(
        extractor=extractor,
        adapter=adapter,
        decoder=decoder,
        linear_probe=False,
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
    actual DINOv2 weights — uses random initialization for speed).
    """
    print("=" * 70)
    print("DINOv2 Segmentation Decoders — Smoke Test (random weights)")
    print("=" * 70)

    def count_trainable_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    IMAGE_SHAPE = (1, 3, 224, 224)
    NUM_CLASSES = 2
    HIDDEN_DIM = 768  # ViT-L
    NUM_LAYERS = 4

    # Build extractor once (frozen backbone)
    backbone = MedDINOv3FeatureExtractor(
        layer_indices=[2, 5, 8, 11],
        adapter="concat",
        freeze_backbone=True,
    )
    backbone.eval()

    dummy = torch.randn(*IMAGE_SHAPE)
    with torch.no_grad():
        features= backbone(dummy)

    print(f"\nInput shape : {IMAGE_SHAPE}")
    print(f"Patch grid  : {features[0].shape[-2]}×{features[0].shape[-1]}  (patch_size={backbone.patch_size}, input={IMAGE_SHAPE[-1]})")
    print(f"Num features: {len(features)}  each {tuple(features[0].shape)}")
    print(f"Backbone trainable params: {count_trainable_params(backbone):,}  (frozen → 0)\n")
    print(f"{'Mulstiscale Decoders':<35} {'Trainable Params':>18}  {'Output Shape'}")

    for name in ['unetrfpn', 'segformer', 'mask2former']:

        full_model = build_segmenter(adapter_type='vitadapter', decoder_type=name, input_channels=3)

        features = full_model(dummy)
        print(f"Num features: {len(features)}  each {tuple(features[0].shape)}")
        n_params = count_trainable_params(full_model)
        print(f"  {name:<33} {n_params:>18,}  {tuple(features.shape)}")
    print("-" * 70)


    print(f"{'Plain Decoders':<35} {'Trainable Params':>18}  {'Output Shape'}")

    for name in ['unetr', 'upernet']:

        full_model = build_segmenter(adapter_type='none', decoder_type=name, input_channels=3)

        features = full_model(dummy)
        print(f"Num features: {len(features)}  each {tuple(features[0].shape)}")
        n_params = count_trainable_params(full_model)
        print(f"  {name:<33} {n_params:>18,}  {tuple(features.shape)}")
    print("-" * 70)
