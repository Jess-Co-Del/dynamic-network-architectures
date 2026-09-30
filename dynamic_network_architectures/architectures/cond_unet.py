from typing import Union, Type, List, Tuple, Optional

import torch, math, einops
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import CondBasicBlockD
from dynamic_network_architectures.building_blocks.cond_residual_encoders import ConditionalResidualEncoder, CondPlainConvEncoder
from dynamic_network_architectures.building_blocks.cond_unet_decoder import ConditionalUNetDecoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        self.weights = torch.exp(
            torch.arange(half_dim) * -(math.log(10000) / (half_dim - 1)))
        # self.embeddings = nn.Parameter(torch.randn(half_dim))  # Learned version of embedding

    def forward(self, time):
        time = einops.rearrange(time, 'b -> b 1')
        device = time.device
        embeddings =  time * einops.rearrange(self.weights, 'b -> 1 b').to(device)
        # embeddings = time[:, None] * self.weights[None, :].to(device) * 2 * math.pi  # Learned version of embedding
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CondPlainConvUNet(nn.Module):
    def __init__(self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
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
        nonlin_first: bool = False,
        block: CondBasicBlockD = CondBasicBlockD,
        time_embedding_dim: int = 512,
        conditional_channels: int = None,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"

        if conditional_channels is not None:
            self.conditional_channels = conditional_channels
            self.embed = nn.Sequential(  # [batch]
                SinusoidalPositionEmbeddings(time_embedding_dim),  # [batch, time_embedding_dim]
                nn.Linear(time_embedding_dim, time_embedding_dim),
                nn.GELU(),
                nn.Linear(time_embedding_dim, time_embedding_dim)
            )

        self.encoder = CondPlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first, time_embedding_dim=time_embedding_dim, block=block)
        self.conditional_encoder = CondPlainConvEncoder(conditional_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first, time_embedding_dim=time_embedding_dim, block=block)
        self.decoder = ConditionalUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first, time_embedding_dim=time_embedding_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, image_conditional: torch.Tensor = None):
        # if image_conditional is not None:
        #     image_first = torch.cat([image_conditional, x], dim=1)

        t_emb = self.embed(t)  # [batch, time_embedding_dim]

        skips = self.encoder(x, t_emb)
        if image_conditional is not None:
            conditional_skips = self.conditional_encoder(image_conditional, t_emb)
            skips = [skip_stage + cond_skip_stage for skip_stage, cond_skip_stage in zip(skips, conditional_skips)]
        return self.decoder(skips, t_emb)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class ConditionalResidualEncoderUNet(nn.Module):
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
    ):
        """
        """
        super().__init__()

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"

        if conditional_channels is not None:
            self.conditional_channels = conditional_channels
            self.embed = nn.Sequential(  # [batch]
                SinusoidalPositionEmbeddings(time_embedding_dim),  # [batch, time_embedding_dim]
                nn.Linear(time_embedding_dim, time_embedding_dim),
                nn.GELU(),
                nn.Linear(time_embedding_dim, time_embedding_dim)
            )

        self.encoder = ConditionalResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
                                       time_embedding_dim=time_embedding_dim)
        self.conditional_encoder = ConditionalResidualEncoder(conditional_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
                                       time_embedding_dim=time_embedding_dim)
        self.decoder = ConditionalUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                       time_embedding_dim=time_embedding_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, image_conditional: torch.Tensor = None):
        t_emb = self.embed(t)  # [batch, time_embedding_dim]

        skips = self.encoder(x, t_emb)
        if image_conditional is not None:
            conditional_skips = self.conditional_encoder(image_conditional, t_emb)
            skips = [skip_stage + cond_skip_stage for skip_stage, cond_skip_stage in zip(skips, conditional_skips)]
        return self.decoder(skips, t_emb)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ConcatConditionalResidualEncoderUNet(nn.Module):
    """
    Conditioning Baseline: no separate conditioning encoder at all. The CT image is simply
    channel-concatenated onto the noisy mask before a single (time-conditioned) ``ConditionalResidualEncoder``.
    """

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
    ):
        super().__init__()
        if conditional_channels is None:
            raise ValueError(
                "ConcatConditionalResidualEncoderUNet needs conditional_channels, the image_conditional channel "
                "count, to size its single encoder's input -- unlike the other classes here it has no code path "
                "that skips conditioning, since the condition image is concatenated in at the input.")
        self.conditional_channels = conditional_channels
        self.embed = nn.Sequential(  # [batch]
            SinusoidalPositionEmbeddings(time_embedding_dim),  # [batch, time_embedding_dim]
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        self.encoder = ConditionalResidualEncoder(
            input_channels + conditional_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
            nonlin_kwargs, block, bottleneck_channels, return_skips=True, disable_default_stem=False,
            stem_channels=stem_channels, time_embedding_dim=time_embedding_dim)
        self.decoder = ConditionalUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                       time_embedding_dim=time_embedding_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, image_conditional: torch.Tensor = None) -> torch.Tensor:
        if image_conditional is None:
            raise ValueError(
                "ConcatConditionalResidualEncoderUNet has no unconditional forward path: image_conditional is "
                "concatenated directly into the encoder's input, not fused in afterwards, so it can't be None.")
        t_emb = self.embed(t)  # [batch, time_embedding_dim]
        skips = self.encoder(torch.cat([x, image_conditional], dim=1), t_emb)
        return self.decoder(skips, t_emb)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class TimeInvariantConditionalResidualEncoderUNet(ConditionalResidualEncoderUNet):
    """
    ``ConditionalResidualEncoderUNet`` with the CT (conditioning) encoder replaced entirely by its non-conditional
    twin, plain ``ResidualEncoder``.
    """

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
    ):
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
            num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, stem_channels,
            time_embedding_dim, conditional_channels)
        # replace the parent's time-conditioned conditional_encoder with its plain, non-conditional twin -- note
        # no `block=` (that's CondBasicBlockD, for the Cond* encoders) and no `time_embedding_dim` (ResidualEncoder
        # doesn't take one), so this genuinely has no time-embedding machinery, not just an unused input to it.
        self.conditional_encoder = ResidualEncoder(
            conditional_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels)

    def _conditional_skips(self, image_conditional: torch.Tensor) -> List[torch.Tensor]:
        return self.conditional_encoder(image_conditional)

    def _encode_with_conditioning(self, x: torch.Tensor, t_emb: torch.Tensor,
                                  cond_skips: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Replicates ConditionalResidualEncoder.forward's own stem/stage loop, injecting the matching
        cond_skips stage into each encoder block's output before it feeds the next stage
        """
        enc = self.encoder
        h = enc.stem(x, t_emb) if enc.stem is not None else x
        skips = []
        for pool, stage, cond_skip in zip(enc.pools, enc.stages, cond_skips):
            if pool is not None:
                h = pool(h)
            h = stage(h, t_emb) + cond_skip
            skips.append(h)
        return skips

    def forward(self, x: torch.Tensor, t: torch.Tensor, image_conditional: torch.Tensor = None) -> torch.Tensor:
        t_emb = self.embed(t)
        if image_conditional is None:
            skips = self.encoder(x, t_emb)
        else:
            cond_skips = self._conditional_skips(image_conditional)
            skips = self._encode_with_conditioning(x, t_emb, cond_skips)
        return self.decoder(skips, t_emb)


# ---------------------------------------------------------------------------------------------------------------
# Gated-fusion variant: replaces ConditionalResidualEncoderUNet's fixed elementwise-sum skip fusion with a learned
# per-stage spatial gate, and stops conditioning the CT (image) encoder on the diffusion timestep.
# ---------------------------------------------------------------------------------------------------------------
class SpatialGatedFusion(nn.Module):
    """Learned, per-stage gated fusion of two feature streams, in place of a fixed elementwise sum.

    Follows the design of ``DualPathResponseFusionAttention`` in ``building_blocks/attention.py`` (two 1x1
    conv+norm branches -> GELU -> a sigmoid spatial gate derived from both -> the gate modulates one branch ->
    concat -> project back), but takes ``norm_op``/``norm_op_kwargs`` as constructor arguments instead of that
    module's hardcoded ``BatchNorm``, so callers conditioning a 3D network trained at a small batch size can pass
    ``InstanceNorm``,  BatchNorm's per-batch statistics get noisy at small 3D batch sizes, which is exactly the
    failure mode nnU-Net's own default to ``InstanceNorm`` exists to avoid.

    ``forward(trunk, gated)``: ``trunk`` is projected and passed through with a fixed activation, never gated:
    training can't zero out this stream. ``gated`` is spatially modulated by a sigmoid gate computed jointly from
    both streams, so its contribution can vary by location. The two are concatenated and a final 1x1 conv projects
    back to ``channels``, so callers don't need to know about the intermediate width.
    """

    def __init__(self, conv_op: Type[nn.Module], channels: int, norm_op: Type[nn.Module],
                 norm_op_kwargs: Union[dict, None] = None, reduction: int = 2):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {}
        inter_channels = max(channels // reduction, 8)

        self.trunk_proj = nn.Sequential(conv_op(channels, inter_channels, 1, bias=True),
                                        norm_op(inter_channels, **norm_op_kwargs))
        self.gated_proj = nn.Sequential(conv_op(channels, inter_channels, 1, bias=False),
                                        norm_op(inter_channels, **norm_op_kwargs))
        self.gate = nn.Sequential(conv_op(inter_channels, 1, 1, bias=True), norm_op(1, **norm_op_kwargs),
                                  nn.Sigmoid())
        self.act = nn.GELU()
        self.project_out = conv_op(2 * inter_channels, channels, 1, bias=True)

    def forward(self, trunk: torch.Tensor, gated: torch.Tensor) -> torch.Tensor:
        n_trunk = self.trunk_proj(trunk)
        n_trunk_out = self.act(n_trunk)
        n_gated = self.gated_proj(gated)
        psi = self.gate(self.act(n_trunk + n_gated))
        fused_gated = n_gated * psi
        return self.project_out(torch.cat([n_trunk_out, fused_gated], dim=1))


class ConditionalResidualEncoderDecoderUNet(TimeInvariantConditionalResidualEncoderUNet):
    """
    ``TimeInvariantConditionalResidualEncoderUNet`` (CT encoder replaced by the plain, non-conditional
    ``ResidualEncoder``) extended so the CT (conditioning) branch is a full encoder-*decoder* instead of an
    encoder alone. ``conditional_decoder`` is correspondingly a plain ``UNetDecoder`` too.
    """

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
    ):
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
            num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, stem_channels,
            time_embedding_dim, conditional_channels)
        self.conditional_decoder = UNetDecoder(
            self.conditional_encoder, num_classes, n_conv_per_stage_decoder, deep_supervision=True)
        # one non-bottleneck stage per encoder stage: project the condition decoder's num_classes-channel output
        # back up to that stage's feature width before summing it into the mask encoder's skip.
        self.cond_proj = nn.ModuleList([
            conv_op(num_classes, channels, 1) for channels in self.encoder.output_channels[:-1]
        ])
        # the condition decoder's own full-resolution prediction, exposed after every forward() so a trainer can
        # supervise it directly (see nnUNetTrainerDDIMCondEncDec in nnUNetDiffuser).
        self.last_condition_logits: Optional[torch.Tensor] = None

    def _fuse_stage(self, mask_feat: torch.Tensor, cond_feat: torch.Tensor, stage: int, side: str) -> torch.Tensor:
        """
        The per-stage fusion operation shared by _encode_with_conditioning and _decode_with_conditioning --
        plain addition here (``side``, one of ``"encoder"``/``"decoder"``, is ignored: addition is symmetric
        regardless of which side is calling). FusionConditionalResidualEncoderDecoderUNet below overrides this
        with a learned gate instead -- a *separate* one for the encoder and the decoder side of each resolution,
        since the two injection points see different mask-feature statistics even though they share a resolution
        and a cond_proj/cond_decoder_outputs source.
        """
        return mask_feat + cond_feat

    def _encode_with_conditioning(self, x: torch.Tensor, t_emb: torch.Tensor, cond_skips: List[torch.Tensor],
                                  cond_decoder_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Overrides the parent's _encode_with_conditioning for this class's own two-source fusion: the
        (projected) condition decoder output for every non-bottleneck stage, the condition encoder's own
        bottleneck skip for the last one
        """
        enc = self.encoder
        h = enc.stem(x, t_emb) if enc.stem is not None else x
        n_stages = len(enc.stages)
        skips = []
        for i, (pool, stage) in enumerate(zip(enc.pools, enc.stages)):
            if pool is not None:
                h = pool(h)
            h = stage(h, t_emb)
            cond_feat = cond_skips[i] if i == n_stages - 1 else self.cond_proj[i](cond_decoder_outputs[i])
            h = self._fuse_stage(h, cond_feat, i, "encoder")
            skips.append(h)
        return skips

    def _decode_with_conditioning(self, skips: List[torch.Tensor], t_emb: torch.Tensor,
                                  cond_decoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Mirrors ConditionalUNetDecoder.forward's own transpconv/stage loop, injecting the same
        cond_proj-projected condition-decoder signal used for the matching (same-resolution) encoder stage into
        every mask decoder block's output too.
        """
        dec = self.decoder
        n_stages = len(self.encoder.output_channels)
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(dec.stages)):
            x = dec.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = dec.stages[s](x, t_emb)
            i = n_stages - s - 2
            x = self._fuse_stage(x, self.cond_proj[i](cond_decoder_outputs[i]), i, "decoder")
            if dec.deep_supervision:
                seg_outputs.append(dec.seg_layers[s](x))
            elif s == len(dec.stages) - 1:
                seg_outputs.append(dec.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if dec.deep_supervision else seg_outputs[0]

    def forward(self, x: torch.Tensor, t: torch.Tensor, image_conditional: torch.Tensor = None) -> torch.Tensor:
        t_emb = self.embed(t)
        if image_conditional is None:
            skips = self.encoder(x, t_emb)
            return self.decoder(skips, t_emb)
        cond_skips = self._conditional_skips(image_conditional)
        cond_decoder_outputs = self.conditional_decoder(cond_skips)  # deep_supervision=True -> a list
        # index 0 is the full-resolution prediction (UNetDecoder.forward inverts the list so the largest
        # segmentation output comes first) -- the condition model's actual mask prediction.
        self.last_condition_logits = cond_decoder_outputs[0]
        skips = self._encode_with_conditioning(x, t_emb, cond_skips, cond_decoder_outputs)
        return self._decode_with_conditioning(skips, t_emb, cond_decoder_outputs)

    def compute_conv_feature_map_size(self, input_size):
        return (super().compute_conv_feature_map_size(input_size)
                + self.conditional_encoder.compute_conv_feature_map_size(input_size)
                + self.conditional_decoder.compute_conv_feature_map_size(input_size))


class FusionConditionalResidualEncoderDecoderUNet(ConditionalResidualEncoderDecoderUNet):
    """
    ``ConditionalResidualEncoderDecoderUNet`` (full condition encoder-decoder, cascading in-block injection into
    both the mask encoder and the mask decoder) with one further change: every per-stage fusion (``_fuse_stage``)
    is a learned ``SpatialGatedFusion`` gate instead of a fixed elementwise sum: the mask stream (``trunk``) can
    never be zeroed out by training, while the condition signal's contribution (``gated``) is spatially and
    adaptively weighted, at every stage of both the encoder and the decoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_fusion_gates = nn.ModuleList([
            SpatialGatedFusion(self.encoder.conv_op, channels, self.encoder.norm_op, self.encoder.norm_op_kwargs)
            for channels in self.encoder.output_channels
        ])
        self.decoder_fusion_gates = nn.ModuleList([
            SpatialGatedFusion(self.encoder.conv_op, channels, self.encoder.norm_op, self.encoder.norm_op_kwargs)
            for channels in self.encoder.output_channels[:-1]
        ])

    def _fuse_stage(self, mask_feat: torch.Tensor, cond_feat: torch.Tensor, stage: int, side: str) -> torch.Tensor:
        gates = self.encoder_fusion_gates if side == "encoder" else self.decoder_fusion_gates
        return gates[stage](mask_feat, cond_feat)


class legacypaper_ConditionalResidualEncoderUNet(nn.Module):
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
    ):
        """
        """
        super().__init__()
        self.ddim_steps = ddim_steps

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"

        if conditional_channels is not None:
            self.conditional_channels = conditional_channels
            self.embed = nn.Sequential(  # [batch]
                SinusoidalPositionEmbeddings(time_embedding_dim),  # [batch, time_embedding_dim]
                nn.Linear(time_embedding_dim, time_embedding_dim),
                nn.GELU(),
                nn.Linear(time_embedding_dim, time_embedding_dim)
            )

        self.encoder = ConditionalResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
                                       time_embedding_dim=time_embedding_dim)
        self.conditional_encoder = ConditionalResidualEncoder(conditional_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
                                       time_embedding_dim=time_embedding_dim)
        self.decoder = ConditionalUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                       time_embedding_dim=time_embedding_dim)

    def forward(self, x: torch.Tensor = None, image_conditional: torch.Tensor = None, ddim: bool = False):
        if ddim:
            return self.ddim_sample_p(image_conditional=image_conditional)
        
        x = (x * 2) - 1

        times = torch.zeros((x.shape[0],), device=x.device).float().uniform_(self.sample_range[0],
                                                                  self.sample_range[1])  # [batch]
        t_emb = self.embed(times)  # [batch, time_embedding_dim]

        alpha, sigma = self.log_snr_to_alpha_sigma(
            self.alpha_cosine_log_snr(times).view(*times.shape, *((1,) * (x.ndim - times.ndim)))
        )  # [batch, time_embedding_dim, 1, 1, 1]
        noised_x = alpha * x + sigma * torch.randn_like(x)

        skips = self.encoder(noised_x, t_emb)
        if image_conditional is not None:
            conditional_skips = self.conditional_encoder(image_conditional, t_emb)
            skips = [skip_stage + cond_skip_stage for skip_stage, cond_skip_stage in zip(skips, conditional_skips)]
        return self.decoder(skips, t_emb)

    @torch.no_grad()
    def ddim_sample_p(self, image_conditional: torch.Tensor):
        x_T = torch.randn(image_conditional.shape, device=image_conditional.device)
        time_pairs = self._get_sampling_timesteps(image_conditional.shape[0], device=image_conditional.device)
        for times_now, times_next in time_pairs:

            alpha_now, sigma_now = self.log_snr_to_alpha_sigma(
                self.alpha_cosine_log_snr(times_now).view(*times_now.shape, *((1,) * (x_T.ndim - times_now.ndim)))
            )
            alpha_next, sigma_next = self.log_snr_to_alpha_sigma(
                self.alpha_cosine_log_snr(times_next).view(*times_now.shape, *((1,) * (x_T.ndim - times_now.ndim)))
            )

            t_emb = self.embed(self.alpha_cosine_log_snr(times_now))

            skips = self.encoder(x_T, t_emb)
            if image_conditional is not None:
                conditional_skips = self.conditional_encoder(image_conditional, t_emb)
                skips = [skip_stage + cond_skip_stage for skip_stage, cond_skip_stage in zip(skips, conditional_skips)]
            pred = self.decoder(skips, t_emb)

            pred = (torch.sigmoid(pred) * 2) - 1
            pred_noise = (x_T - alpha_now * pred) / sigma_now.clamp(min=1e-8)
            pred = pred * alpha_next + pred_noise * sigma_next
        return pred

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
        
    def log(self, t, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def beta_linear_log_snr(self, t):
        return -torch.log(torch.expm1(1e-4 + 10 * (t ** 2)))

    def alpha_cosine_log_snr(self, t, ns=0.0002, ds=0.00025):
        # not sure if this accounts for beta being clipped to 0.999 in discrete version
        return -self.log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

    def log_snr_to_alpha_sigma(self, log_snr):
        return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.ddim_steps):
            t_now = 1 - (step / self.ddim_steps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.ddim_steps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times
