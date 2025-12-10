from typing import Union, Type, List, Tuple

import torch, math, einops
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import CondBasicBlockD
from dynamic_network_architectures.building_blocks.cond_residual_encoders import ConditionalResidualEncoder, CondPlainConvEncoder
from dynamic_network_architectures.building_blocks.cond_unet_decoder import ConditionalUNetDecoder
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
                                       time_embedding_dim)

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
        init_last_bn_before_add_to_0(module)


class paper_ConditionalResidualEncoderUNet(nn.Module):
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
                                       time_embedding_dim)

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
