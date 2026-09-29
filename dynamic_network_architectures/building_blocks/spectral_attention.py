"""
Spectral (Fourier-domain) and spatial cross-attention building blocks for MedSegDiff-V2-style conditioning
(Wu et al., "MedSegDiff-V2", arXiv:2301.11798).
"""

from __future__ import annotations

from typing import Optional, Type

import torch
from torch import nn


def _gaussian_kernel(kernel_size: int, sigma: float, dim: int, channels: int) -> torch.Tensor:
    """
    Depthwise-conv weight tensor (channels, 1, *([kernel_size] * dim)) for a Gaussian blur, one copy per
    channel, the starting point for AnchorAttention's *learnable* smoothing kernel (weights stay trainable).
    """
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    grids = torch.meshgrid(*([coords] * dim), indexing="ij")
    kernel = torch.exp(-sum(g ** 2 for g in grids) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, *kernel.shape).repeat(channels, 1, *([1] * dim))


class NeuralBandPassFilter(nn.Module):
    """
    A learnable, timestep-adaptive band-pass filter over a frequency-bin axis.

    MedSegDiff-V2: "NBP-Filter learns to pass a specific spectrum while constraining the others", with a
    timestep embedding driving scaling and shifting (adaptive-norm style). Implemented as a learnable depthwise
    1D gain (via a small conv over the flattened frequency axis, so it works at any bottleneck resolution without
    baking a fixed frequency-bin count into the parameter count) rather than one free parameter per frequency
    bin, then FiLM-modulated by the timestep embedding.
    """

    def __init__(self, channels: int, time_embedding_dim: int, kernel_size: int = 7):
        super().__init__()
        self.gain = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, 2 * channels))

    def forward(self, freq_feat: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """``freq_feat``: (B, C, F) real-valued features over F frequency-bin tokens."""
        scale, shift = self.film(t_emb).chunk(2, dim=-1)  # each (B, C)
        gate = torch.sigmoid(self.gain(freq_feat))
        return freq_feat * gate * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class SpatialCrossAttention(nn.Module):
    """
    Plain (non-spectral) multi-head cross-attention between two same-shape feature maps, flattened to spatial
    tokens. Usable standalone as the ``"cross_attention"`` semantic-fusion option, and reused internally by
    ``SSFormer`` for its spatial refinement step.
    """

    def __init__(self, channels: int, num_heads: int = 4, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(channels, num_heads, batch_first=True),
                "norm_q": nn.LayerNorm(channels),
                "norm_kv": nn.LayerNorm(channels),
            })
            for _ in range(num_blocks)
        ])

    def forward(self, query_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = query_map.shape
        q = query_map.flatten(2).transpose(1, 2)  # (B, N, C)
        kv = kv_map.flatten(2).transpose(1, 2)
        for block in self.blocks:
            attended, _ = block["attn"](block["norm_q"](q), block["norm_kv"](kv), block["norm_kv"](kv))
            q = q + attended
        return q.transpose(1, 2).reshape(b, c, *spatial)


class SSFormer(nn.Module):
    """
    Semantic conditioning via cross-attention in the Fourier domain (MedSegDiff-V2's SS-Former).

    1. ``rfftn`` both embeddings over their spatial dims (always in float32 -- FFT under bf16/fp16 autocast is
       unsupported/unstable, so this casts in and back out regardless of the caller's autocast context).
    2. The real/imaginary parts (concatenated) become a token sequence over frequency bins; cross-attend
       (condition = K/V, diffusion = Q) to get an affinity-weighted frequency representation.
    3. Gate that through a ``NeuralBandPassFilter``, conditioned on the diffusion timestep.
    4. ``irfftn`` back to the spatial domain, added residually to the original diffusion embedding.
    5. Two ``SpatialCrossAttention`` blocks refine the noise/semantic interaction in the spatial domain --
       a simplification of the paper's "two *symmetric*" blocks (both here take the diffusion stream as query).
    """

    def __init__(self, channels: int, time_embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.freq_proj_q = nn.Linear(channels * 2, channels)
        self.freq_proj_kv = nn.Linear(channels * 2, channels)
        self.freq_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.band_pass = NeuralBandPassFilter(channels, time_embedding_dim)
        self.freq_out = nn.Linear(channels, channels * 2)
        self.spatial_refine = SpatialCrossAttention(channels, num_heads=num_heads, num_blocks=2)

    @staticmethod
    def _to_freq_tokens(z: torch.Tensor) -> torch.Tensor:
        """complex (B, C, *F) -> real (B, N, 2C): real/imag parts concatenated on the channel axis, then
        flattened to a token sequence."""
        ri = torch.cat([z.real, z.imag], dim=1)
        return ri.flatten(2).transpose(1, 2)

    def forward(self, diffusion_embedding: torch.Tensor, condition_embedding: torch.Tensor,
               t_emb: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = diffusion_embedding.shape
        spatial_dims = tuple(range(2, diffusion_embedding.ndim))
        orig_dtype = diffusion_embedding.dtype

        d_freq = torch.fft.rfftn(diffusion_embedding.float(), dim=spatial_dims, norm="ortho")
        c_freq = torch.fft.rfftn(condition_embedding.float(), dim=spatial_dims, norm="ortho")
        freq_shape = d_freq.shape[2:]

        q = self.freq_proj_q(self._to_freq_tokens(d_freq))
        kv = self.freq_proj_kv(self._to_freq_tokens(c_freq))
        attended, _ = self.freq_attn(q, kv, kv)  # (B, N, C)

        gated = self.band_pass(attended.transpose(1, 2), t_emb).transpose(1, 2)  # (B, N, C)

        ri = self.freq_out(gated)  # (B, N, 2C)
        ri = ri.transpose(1, 2).reshape(b, 2 * c, *freq_shape)
        fused_freq = torch.complex(ri[:, :c].contiguous(), ri[:, c:].contiguous())
        fused_spatial = torch.fft.irfftn(fused_freq, s=spatial, dim=spatial_dims, norm="ortho").to(orig_dtype)

        out = diffusion_embedding + fused_spatial
        return self.spatial_refine(out, condition_embedding)


class AnchorAttention(nn.Module):
    """
    Injects a condition model's own predicted mask into a denoising encoder's first stage.

    MedSegDiff-V2: ``sigmoid(conv1x1(max(smooth(anchor), anchor))) * f_d^0 + f_d^0``. ``smooth`` is a learnable
    depthwise conv initialized as a Gaussian blur ("learnable Gaussian kernel"); the ``conv1x1`` projects the
    condition model's ``num_classes`` channels down to a single spatial gate, broadcast across every channel of
    ``diffusion_feat``.
    """

    def __init__(self, conv_op: Type[nn.Module], num_classes: int, kernel_size: int = 5, sigma: float = 1.5):
        super().__init__()
        dim = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}[conv_op]
        self.smooth = conv_op(num_classes, num_classes, kernel_size, padding=kernel_size // 2,
                              groups=num_classes, bias=False)
        with torch.no_grad():
            self.smooth.weight.copy_(_gaussian_kernel(kernel_size, sigma, dim, num_classes))
        self.gate = nn.Sequential(conv_op(num_classes, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, anchor_logits: torch.Tensor, diffusion_feat: torch.Tensor) -> torch.Tensor:
        smoothed = self.smooth(anchor_logits)
        combined = torch.maximum(smoothed, anchor_logits)
        gate = self.gate(combined)
        return gate * diffusion_feat + diffusion_feat
