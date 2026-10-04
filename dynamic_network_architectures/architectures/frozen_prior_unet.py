"""
Frozen/trainable deterministic-prior conditioning architectures.

Unlike every other conditioning architecture in this package, the conditioning branch here is not necessarily
trained jointly
"""

from __future__ import annotations

import torch

from dynamic_network_architectures.architectures.cond_unet import ConditionalResidualEncoderDecoderUNet
from dynamic_network_architectures.building_blocks.spectral_attention import SpatialCrossAttention


class FrozenPriorConditionalResidualEncoderDecoderUNet(ConditionalResidualEncoderDecoderUNet):
    """
    ``ConditionalResidualEncoderDecoderUNet`` whose conditioning branch (``conditional_encoder``/
    ``conditional_decoder``) is meant to be loaded from an already-trained, frozen deterministic nnU-Net checkpoint
    instead of trained jointly from scratch
    """

    def train(self, mode: bool = True):
        super().train(mode)
        self.conditional_encoder.eval()
        self.conditional_decoder.eval()
        return self


class HiDiffConditionalResidualEncoderDecoderUNet(ConditionalResidualEncoderDecoderUNet):
    """
    ``ConditionalResidualEncoderDecoderUNet`` with one further change, carried over from HiDiff (Chen et al.,
    "HiDiff: Hybrid Diffusion Framework for Medical Image Segmentation", arXiv:2407.03548): the conditioning
    branch's own bottleneck feature cross-attends into the diffusion encoder's bottleneck too (HiDiff's
    "X-Former"), on top of the usual per-stage additive injection every stage already gets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bottleneck_channels = self.encoder.output_channels[-1]
        self.prior_exchange = SpatialCrossAttention(bottleneck_channels)

    def compute_prior(self, image_conditional: torch.Tensor):
        """The conditioning branch's own skips and (deep-supervision) decoder outputs -- factored out of
        ``forward`` so a trainer can compute this once per batch and pass it back in via ``forward``'s ``_prior``
        argument, instead of running the whole conditioning encoder-decoder twice per step (once to derive the
        prior signal, once again inside ``forward``'s own conditioning computation)."""
        cond_skips = self._conditional_skips(image_conditional)
        cond_decoder_outputs = self.conditional_decoder(cond_skips)
        return cond_skips, cond_decoder_outputs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        image_conditional: torch.Tensor = None,
        _prior=None
    ) -> torch.Tensor:
        t_emb = self.embed(t)
        if image_conditional is None:
            skips = self.encoder(x, t_emb)
            return self.decoder(skips, t_emb)
        cond_skips, cond_decoder_outputs = _prior if _prior is not None else self.compute_prior(image_conditional)
        self.last_condition_logits = cond_decoder_outputs[0]
        skips = self._encode_with_conditioning(x, t_emb, cond_skips, cond_decoder_outputs)
        # HiDiff's X-Former, prior -> diffusion direction only.
        skips[-1] = self.prior_exchange(skips[-1], cond_skips[-1])
        return self._decode_with_conditioning(skips, t_emb, cond_decoder_outputs)


class HiDiffBernoulliConditionalResidualEncoderDecoderUNet(HiDiffConditionalResidualEncoderDecoderUNet):
    """
    ``HiDiffConditionalResidualEncoderDecoderUNet`` with the conditioning branch forced into eval mode (same
    ``train()``: used in Trainer which combines this architecture with a discrete Bernoulli diffusion process
    instead of the continuous Gaussian DDIM.
    """

    def train(self, mode: bool = True):
        super().train(mode)
        self.conditional_encoder.eval()
        self.conditional_decoder.eval()
        return self
