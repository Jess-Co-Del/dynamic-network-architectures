"""
MedSegDiff-V2-style conditioning (Wu et al., arXiv:2301.11798): a full standalone condition U-Net: its own
encoder AND decoder, trained with direct segmentation supervision, conditioning the denoising network through two
targeted mechanisms.

    1. Anchor conditioning is applied *in-encoder*, faithfully to the paper: the diffusion encoder's stage-0 output is
    gated by the condition model's predicted mask *before* it is fed forward into stage 1, so the anchor-modified
    feature genuinely propagates into every deeper encoder stage (including the bottleneck), not just the first skip
    connection handed to the decoder.
    2. Semantic-conditioning mechanism at the bottleneck (parameterized ablation by field ``semantic_fusion``):
    Includes the paper's SS-Former: ``"ss_former"`` (Fourier-domain cross-attention through a learnable timestep-adaptive
    band-pass filter coming from ``building_blocks/spectral_attention.py``), ``"cross_attention"`` (ss_former without the
    FFT step), ``"addition"`` and ``"concatenation"`` (baselines).

The condition model's own predicted mask is exposed as ``self.last_condition_logits`` after every ``forward()``
call, the trainer can add a direct supervision loss on it alongside the main diffusion loss.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.cond_unet import SinusoidalPositionEmbeddings
from dynamic_network_architectures.building_blocks.cond_residual_encoders import ConditionalResidualEncoder
from dynamic_network_architectures.building_blocks.cond_unet_decoder import ConditionalUNetDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import CondBasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.spectral_attention import (AnchorAttention, SSFormer,
                                                                              SpatialCrossAttention)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0

SEMANTIC_FUSION_MODES = ("ss_former", "addition", "concatenation", "cross_attention")


class MedSegDiffV2ConditionalResidualEncoderUNet(nn.Module):
    def __init__(self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: CondBasicBlockD = CondBasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        time_embedding_dim: int = 512,
        conditional_channels: int = None,
        semantic_fusion: str = "ss_former",
    ):
        super().__init__()
        if semantic_fusion not in SEMANTIC_FUSION_MODES:
            raise ValueError(f"semantic_fusion must be one of {SEMANTIC_FUSION_MODES}, got '{semantic_fusion}'")
        if conditional_channels is None:
            raise ValueError(
                "MedSegDiffV2ConditionalResidualEncoderUNet needs conditional_channels, the "
                             "condition model has no meaning without a conditioning image")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # diffusion branch: identical in construction to ConditionalResidualEncoderUNet's own encoder/decoder.
        self.encoder = ConditionalResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block,
            bottleneck_channels, return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
            time_embedding_dim=time_embedding_dim)
        self.decoder = ConditionalUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                              time_embedding_dim=time_embedding_dim)
        if not deep_supervision:
            # only the last stage's seg layer is used below; the others would get no gradient, which makes
            # multi-GPU DDP (nnU-Net does not set find_unused_parameters) raise.
            self.decoder.seg_layers = nn.ModuleList([self.decoder.seg_layers[-1]])

        # condition branch: a full, plain (no time embedding) U-Net: its own encoder AND decoder, the key
        # structural difference from every other conditioning architecture in this package. Same
        # features_per_stage as the diffusion branch, so bottleneck channel counts match for semantic fusion.
        self.condition_encoder = ResidualEncoder(
            conditional_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.condition_decoder = UNetDecoder(
            self.condition_encoder, num_classes, n_conv_per_stage_decoder,
            deep_supervision=False)
        self.condition_decoder.seg_layers = nn.ModuleList([self.condition_decoder.seg_layers[-1]])

        bottleneck_ch = self.encoder.output_channels[-1]
        self.anchor_attn = AnchorAttention(conv_op, num_classes)
        self.semantic_fusion_mode = semantic_fusion
        if semantic_fusion == "ss_former":
            self.semantic_fusion = SSFormer(bottleneck_ch, time_embedding_dim)
        elif semantic_fusion == "cross_attention":
            self.semantic_fusion = SpatialCrossAttention(bottleneck_ch)
        elif semantic_fusion == "concatenation":
            self.semantic_fusion = conv_op(2 * bottleneck_ch, bottleneck_ch, 1)
        else:  # addition: no learnable module needed
            self.semantic_fusion = None

        self.last_condition_logits: Optional[torch.Tensor] = None

    def _encode_mask_with_anchor(self, x: torch.Tensor, t_emb: torch.Tensor,
                                 cond_logits: torch.Tensor) -> List[torch.Tensor]:
        """
        ``self.encoder(x, t_emb)``, but with the anchor gate spliced in between stage 0 and stage 1: replicates
        ``ConditionalResidualEncoder.forward``'s own stem/stage loop (see that method) so the anchor-modified
        stage-0 feature is what stage 1 (and everything deeper) actually sees, not a post-hoc edit of the skip
        list after the real forward pass already ran through every stage using the *un*-modified feature.
        """
        enc = self.encoder
        h = enc.stem(x, t_emb) if enc.stem is not None else x
        skips = []
        for i, (pool, stage) in enumerate(zip(enc.pools, enc.stages)):
            if pool is not None:
                h = pool(h)
            h = stage(h, t_emb)
            if i == 0:
                h = self.anchor_attn(cond_logits, h)  # faithful in-encoder injection, before stage 1 consumes it
            skips.append(h)
        return skips

    def forward(self, x: torch.Tensor, t: torch.Tensor, image_conditional: torch.Tensor = None) -> torch.Tensor:
        if image_conditional is None:
            raise ValueError("MedSegDiffV2ConditionalResidualEncoderUNet requires image_conditional")

        t_emb = self.embed(t)

        cond_skips = self.condition_encoder(image_conditional)
        cond_logits = self.condition_decoder(cond_skips)
        self.last_condition_logits = cond_logits

        mask_skips = self._encode_mask_with_anchor(x, t_emb, cond_logits)

        bottleneck, cond_bottleneck = mask_skips[-1], cond_skips[-1]
        if self.semantic_fusion_mode == "ss_former":
            fused = self.semantic_fusion(bottleneck, cond_bottleneck, t_emb)
        elif self.semantic_fusion_mode == "cross_attention":
            fused = self.semantic_fusion(bottleneck, cond_bottleneck)
        elif self.semantic_fusion_mode == "concatenation":
            fused = self.semantic_fusion(torch.cat([bottleneck, cond_bottleneck], dim=1))
        else:  # addition
            fused = bottleneck + cond_bottleneck
        mask_skips[-1] = fused

        return self.decoder(mask_skips, t_emb)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), \
            "just give the image size without color/feature channels or batch channel."
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size)
                + self.condition_encoder.compute_conv_feature_map_size(input_size)
                + self.condition_decoder.compute_conv_feature_map_size(input_size))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
