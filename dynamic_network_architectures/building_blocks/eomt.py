"""
Encoder-only Mask Transformer (EoMT)
=====================================
Based on: "Your ViT is Secretly an Image Segmentation Model"
          Kerssies et al., CVPR 2025 (Highlight)
          https://arxiv.org/abs/2503.19108

Architecture
------------
Queries and image-patch tokens are jointly processed through the **last
`num_query_blocks` transformer block groups** of a frozen DINOv2 backbone.
The earlier block groups run in "image-only" mode (no queries).

At every query-aware block boundary an intermediate mask prediction is
generated.  During training this prediction is used to build a sparse
attention mask so each query only attends to its own foreground patches
(masked attention).  At inference masked attention can be disabled for
maximum throughput.

A *mask-annealing* schedule gradually reduces the probability of applying
the attention mask as training progresses, preventing over-reliance on the
mask and enabling masked-attention-free inference.

Flash Attention
---------------
Flash Attention 2 is used for all attention operations through
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which
dispatches to Flash Attention when CUDA is available.  For the masked-
attention path we pass a boolean `attn_mask` tensor; SDPA handles the
sparse pattern efficiently.

Integration with DINOv2FeatureExtractor
-----------------------------------------
`DINOv2FeatureExtractor.get_block_groups()` returns a list of
`nn.Sequential` block groups split at `layer_indices`.  EoMT uses these
groups directly:
  - groups[:-num_query_groups]  →  image-only encoding
  - groups[-num_query_groups:]  →  joint image + query encoding

Outputs
-------
The model returns a dict:
    {
        "class_logits": Tensor[B, Q, num_classes + 1],
        "mask_logits":  Tensor[B, Q, H, W],          # original resolution
        "aux_outputs":  List[{"class_logits", "mask_logits"}],  # per layer
    }

The caller is responsible for loss computation (Hungarian matching + CE +
Dice / BCE on masks).

Usage example
-------------
    extractor = DINOv2FeatureExtractor(
        model_name="facebook/dinov2-large",
        layer_indices=[5, 11, 17, 23],
        freeze_backbone=True,
    )
    model = EoMT(
        encoder=extractor,
        num_classes=2,        # liver + tumour (background handled separately)
        num_queries=100,
        num_query_groups=2,   # last 2 block groups become query-aware
        mask_annealing=True,
    )
    out = model(pixel_values)   # [B, 1, H, W] CT slices
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors
    From https://huggingface.co/spaces/Roll20/pet_score/blob/d589c4f0848517e467d038251ef123abdfd0423b/lib/timm/models/layers/norm.py
    """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class ScaleBlock(nn.Module):
    """Transposed Conv upsampling followed by a Conv-BN-GELU block."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_dim, in_dim,
            kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.norm = LayerNorm2d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.act(self.up(x))))


def _build_upscale_head(hidden_dim: int, patch_size: int) -> nn.Sequential:
    """
    Build a stack of ScaleBlocks to go from patch-grid → ×4 resolution.

    With DINOv2-Large (patch_size=14) this gives 14/4 ≈ 3.5 px per output
    step, so we upsample ×4 total, which is enough to produce a
    reasonable mask at ~1/4 of the input resolution.  A final ×4 bilinear
    upsample in the forward pass brings it to full resolution.
    """
    num_ups = max(1, round(math.log2(patch_size)-1))  # e.g. patch=14 → 3
    layers: List[nn.Module] = []
    in_dim = hidden_dim
    for i in range(num_ups):
        #out_dim = in_dim // 2 if i < num_ups - 1 else in_dim // 2
        #out_dim = max(out_dim, 64)
        layers.append(ScaleBlock(in_dim, in_dim))
        #in_dim = out_dim
    return nn.Sequential(*layers), in_dim


# ---------------------------------------------------------------------------
# Flash-Attention self-attention block (replaces DINOv2 block in fwd pass)
# ---------------------------------------------------------------------------

def _flash_self_attn(
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    num_heads: int,
    qkv_proj: nn.Linear,
    out_proj: nn.Linear,
    dropout_p: float = 0.1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Multi-head self-attention using torch SDPA (Flash Attention backend).

    Parameters
    ----------
    x          : (B, N, D)
    attn_mask  : (B, num_heads, N, N) or (B, 1, N, N) additive mask, or None
    num_heads  : int
    qkv_proj   : nn.Linear  D → 3*D
    out_proj   : nn.Linear  D → D
    scale      : optional pre-computed softmax scale (1/sqrt(head_dim))
    """
    B, N, D = x.shape
    head_dim = D // num_heads

    qkv = qkv_proj(x)  # (B, N, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)
    # Reshape to (B, num_heads, N, head_dim)
    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

    # SDPA dispatches to Flash Attention 2 on CUDA
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,  # additive float mask; None → full attention
        dropout_p=dropout_p,
        scale=scale,
    )  # (B, num_heads, N, head_dim)

    out = out.transpose(1, 2).reshape(B, N, D)
    return out_proj(out)


# ---------------------------------------------------------------------------
# Masked-Attention EoMT block wrapper
# ---------------------------------------------------------------------------

class EoMTBlock(nn.Module):
    """
    Wraps a single encoder transformer block and re-runs its
    attention using Flash Attention + optional query-masked attention.

    Only the attention sub-layer is re-implemented; the MLP, LayerNorm, and
    residual connections are preserved from the original block.

    The HuggingFace Dinov2Layer has:
        block.attention.attention  →  Dinov2SelfAttention
            .query / .key / .value  →  nn.Linear each (D → D)
            .out_proj               →  nn.Linear (D → D)
        block.attention.output.dense  →  same as out_proj (some versions)
        block.layer_scale / block.layer_scale2  →  optional LayerScale
        block.norm1 / block.norm2  →  LayerNorm
        block.mlp  →  feed-forward
    """

    def __init__(self, hf_block: nn.Module, num_heads: int) -> None:
        super().__init__()
        self.block = hf_block
        self.num_heads = num_heads

        # Resolve projections robustly across HF versions
        if hasattr(hf_block, "attn"):
                attn = hf_block.attn
        else:
            attn = hf_block.attention
        # Build a fused QKV projection for efficiency
        if hasattr(attn, "query"):
            attn = attn.query
        D = attn.qkv.weight.shape[0]
        self.D = D
        #self.qkv =# nn.Linear(D, 3 * D, bias=True)
        with torch.no_grad():
            self.qkv = attn.qkv
        # Output projection
        self.out_proj = attn.proj

    def _get_norms_and_scales(self):
        b = self.block
        norm1 = b.norm1
        norm2 = b.norm2
        ls1 = getattr(b, "layer_scale", None)
        ls2 = getattr(b, "layer_scale2", None)
        return norm1, norm2, ls1, ls2

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, N, D)  — N = num_patches [+ num_queries]
        attn_mask: (B, 1, N, N) additive float mask or None
                   −inf where attention is suppressed, 0 elsewhere.
        """
        norm1, norm2, ls1, ls2 = self._get_norms_and_scales()

        # --- Attention sub-layer ---
        residual = x
        x_normed = norm1(x)
        attn_out = _flash_self_attn(
            x_normed,
            attn_mask=attn_mask,
            num_heads=self.num_heads,
            qkv_proj=self.qkv,
            out_proj=self.out_proj,
        )
        if ls1 is not None:
            attn_out = ls1(attn_out)
        x = residual + attn_out

        # --- MLP sub-layer (unchanged from original block) ---
        residual = x
        mlp_out = self.block.mlp(norm2(x))
        if ls2 is not None:
            mlp_out = ls2(mlp_out)
        x = residual + mlp_out

        return x


# ---------------------------------------------------------------------------
# EoMT
# ---------------------------------------------------------------------------

class EoMT(nn.Module):
    """
    Encoder-only Mask Transformer for binary / multi-class segmentation.

    Parameters
    ----------
    encoder : DINOv2FeatureExtractor
        Must be initialised with ``adapter != 'last'`` and at least
        ``num_query_groups`` layer groups (i.e., len(layer_indices) >=
        num_query_groups).
    num_classes : int
        Number of foreground classes (background / no-object handled by an
        extra head slot).
    num_queries : int
        Number of learnable object queries Q.
    num_query_groups : int
        Number of block groups (from the end) that process queries jointly
        with image patches.  Typically 2 for DINOv2-Large with 4 groups.
    mask_annealing : bool
        If True, apply mask annealing during training (probability of
        applying the attention mask decreases each call via
        ``set_mask_ratio()``).
    mask_ratio : float
        Initial probability of applying the masked attention per-query.
        Call ``set_mask_ratio(r)`` each epoch to decay it.
    upscale_factor : int
        Additional bilinear upsampling after the ScaleBlock stack so that
        mask_logits are returned at the **original** input resolution.
        Default 4 gives reasonable quality; set to the actual patch_size //
        scale if you want finer control.
    """

    def __init__(
        self,
        encoder,                    # DINOv2FeatureExtractor instance
        num_classes: int = 2,
        num_queries: int = 2,
        num_query_groups: int = 2,
        mask_annealing: bool = True,
        mask_ratio: float = 1.0,
        upscale_factor: int = 4,
        num_heads: int = 6,
        return_dict: bool = False
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_query_groups = num_query_groups
        self.mask_annealing = mask_annealing
        self.mask_ratio = mask_ratio
        self.upscale_factor = upscale_factor
        self.return_dict = return_dict

        D = encoder.hidden_dim
        num_heads = encoder.backbone.num_heads
        self.num_heads = num_heads
        self.patch_size = encoder.patch_size
        self.num_register_tokens = encoder.backbone.num_register_tokens

        # ── Learnable queries ────────────────────────────────────────────
        self.q = nn.Embedding(num_queries, D)
        nn.init.trunc_normal_(self.q.weight, std=0.02)

        # ── Prediction heads ─────────────────────────────────────────────
        # Classification: Q → (num_classes + 1)  "+1" = no-object
        self.class_head = nn.Linear(D, num_classes+1)

        # Mask: 3-layer MLP with GELU, projects D → D/4 for dot product
        mid = D // 2
        self.mask_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, D),
        )

        # Patch feature projection for mask dot product
        self.patch_proj = nn.Linear(D, D)

        # Upscaling stack: patch-grid → ×(patch_size/4) → ×4 final
        upscale_seq, last_dim = _build_upscale_head(D, self.patch_size)
        self.upscale = upscale_seq
        
        # The upscaled patch features are in reduced-dim space; queries in mid-dim
        # so we compute mask_logits = (query_feat [B,Q,mid]) x (patch_feat [B,mid,H',W'])
        # then upsample H',W' → H,W

        # ── Wrap last num_query_groups block groups with EoMT blocks ─────
        # get_block_groups() returns Sequential groups; we need individual blocks
        block_groups = encoder.get_block_groups('all')  # List[nn.Sequential]

        # Image-only groups (run normally via HF model — no change needed)
        num_image_only_groups = len(block_groups) - num_query_groups
        assert num_image_only_groups >= 0, (
            f"num_query_groups={num_query_groups} exceeds total groups "
            f"{len(block_groups)}.  Increase layer_indices or reduce "
            f"num_query_groups."
        )
        # Query-aware groups: re-wrap each block with Plain Blocks
        self.plain_blocks = nn.Sequential(
            *[group for group in block_groups[:num_image_only_groups]]
        )

        # Query-aware groups: re-wrap each block with EoMTBlock
        self.query_blocks = nn.ModuleList()
        for group in block_groups[-num_query_groups:]:
            self.query_blocks.append(EoMTBlock(group, num_heads))

        # LayerNorm from backbone (applied after all blocks)
        self.layernorm = nn.LayerNorm(D, eps=1e-8)
        self.final_norm = encoder.backbone.layernorm

    # ── Mask-annealing API ───────────────────────────────────────────────

    def set_mask_ratio(self, ratio: float) -> None:
        """Call once per epoch: ratio=1.0 → always masked, 0.0 → never."""
        self.mask_ratio = float(ratio)

    # ── Internal helpers ─────────────────────────────────────────────────
    def _predict(
        self,
        q_feats: torch.Tensor,   # (B, Q, D)
        x_patch: torch.Tensor,   # (B, N, D)
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Compute intermediate mask logits at patch resolution.

        Returns
        -------
        mask_logits : (B, Q, h, w)
        """
        # q_emb = self.mask_head(q_feats)       # (B, Q, D)
        # p_emb = self.patch_proj(x_patch)      # (B, N, D)
        p_spatial = x_patch.permute(0, 2, 1).reshape(
            x_patch.shape[0], -1, h, w
        )                                      # (B, D, h, w)

        # Dot product: (B,Q,D) x (B,D,h*w) → (B,Q,h*w) → (B,Q,h,w)
        #B, Q, D = q_emb.shape
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q_feats), self.upscale(p_spatial)
        )                                      # (B, Q, h, w)

        return mask_logits

    def _build_attn_mask(
        self,
        mask_logits: torch.Tensor,   # (B, Q, h, w)
        num_patch: int,              # h * w
        grid_size: tuple,
        num_query: int,              # Q
    ) -> torch.Tensor:
        """
        Convert spatial mask logits into a (B, 1, Q+N, Q+N) additive
        attention mask where:
          - Query → Patch:  −inf if patch is outside query's foreground
          - All other pairs: 0 (full attention preserved)

        During training, mask_annealing randomly drops mask constraints for
        some queries (probability 1 − mask_ratio).
        """
        interpolated = F.interpolate(
            mask_logits,
            grid_size,
            mode="bilinear",
        )
        B, Q, h, w = interpolated.shape
        N = h * w + self.num_register_tokens

        # (B, Q, N)
        patch_mask = (interpolated.view(B, Q, -1) > 0)  # True = attend

        # Build full (B, 1, Q+N, Q+N) mask
        # Rows = queries (0..Q-1) and patches (Q..Q+N-1)
        # Cols = same order
        total = N + Q
        full_mask = torch.ones(B, 1, total, total, dtype=torch.bool, device=mask_logits.device)
        full_mask[:, 0, :Q, Q+self.num_register_tokens:] = patch_mask

        if self.training and self.mask_annealing and self.mask_ratio < 1.0:
            # For each query independently, keep full attention with prob
            # (1 - mask_ratio) → set entire row to True
            keep = torch.rand(B, Q, 1, device=mask_logits.device) > self.mask_ratio
            # Query rows (0..Q-1), Patch columns (Q..Q+N-1)
            full_mask[:, 0, :Q, Q+self.num_register_tokens:][keep] = 1   # restrict query→patch attention

        full_mask = full_mask.float().masked_fill(~full_mask, -1e9)
        # All other entries remain 0 (patch→patch, patch→query, query→query
        # retain full attention)
        return full_mask   # (B, 1, Q+N, Q+N)

    def _run_image_only_blocks(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Run the image-only block groups through the backbone.

        Uses the backbone's native forward up to the split point by
        re-using the original HF block infrastructure.

        Returns
        -------
        x     : (B, 1+N, D)  — CLS token + patch tokens (pre-LN)
        h, w  : patch grid dims
        """
        backbone = self.encoder.backbone

        # Preprocess
        pv = self.encoder._preprocess(pixel_values)
        B, _, H, W = pv.shape
        h = H // self.patch_size
        w = W // self.patch_size

        # Embedding layer
        # HuggingFace Dinov2Model: backbone.embeddings(pixel_values)
        x = backbone.embeddings(pv)  # (B, 1+N, D)  — includes CLS

        # Run image-only block groups
        x = self.plain_blocks(x)

        return x, h, w

    def _run_query_blocks(
        self,
        x: torch.Tensor,        # (B, 1+N, D)
        queries: torch.Tensor,  # (B, Q, D)
        h: int,
        w: int,
        masked_attn_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Run the query-aware block groups, computing intermediate predictions.

        Returns
        -------
        x_patch  : (B, N, D)  — final patch features (no CLS)
        q_feats  : (B, Q, D)  — final query features
        aux_list : auxiliary predictions from intermediate layers
        """
        B = x.shape[0]
        Q = queries.shape[1]
        N = h * w
        cls_token = x[:, :1, :]         # (B, 1, D)
        x_patch = x[:, 1:, :]           # (B, N, D)

        # Concatenate: [cls | patch | queries]
        # We keep CLS to avoid disturbing positional embeddings; queries appended
        xq = torch.cat([queries, cls_token, x_patch], dim=1)
        xq = self.layernorm(xq)

        # Shape: (B, 1 + N + Q, D)
        # Index layout: 0 = CLS, 1..N = patches, N+1..N+Q = queries

        aux_list: List[Dict] = []
        attn_mask: Optional[torch.Tensor] = None

        for idx, eomtblock in enumerate(self.query_blocks):
            # Extract patch and query features
            x_patch_cur = xq[:, Q+self.num_register_tokens:, :]    # (B, N, D)
            q_cur = xq[:, :Q, :]            # (B, Q, D)

            # Build intermediate mask prediction
            mask_logits = self._predict(q_cur, x_patch_cur, h, w)
            #class_logits = self.class_head(q_cur)

            aux_list.append({
                #"class_logits": class_logits,
                "mask_logits": mask_logits,
            })

            # Update attention mask for the next block
            if masked_attn_enabled and self.mask_ratio > 0:
                attn_mask = self._build_attn_mask(mask_logits, N, h, self.num_queries)
            # from torchvision.utils import save_image
            # save_image(attn_mask[0,0], f'img_block_{idx}.png')

            xq = eomtblock(xq, attn_mask=attn_mask)

        x_patch_final = xq[:, Q+self.num_register_tokens:, :]
        q_final = xq[:, :Q, :]
        return x_patch_final, q_final, aux_list

    # ── Public forward ───────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        masked_attn_enabled: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pixel_values       : (B, C, H, W) — C=1 for CT slices
        masked_attn_enabled: override the training/inference default.
                             None → True during training, False during eval.

        Returns
        -------
        dict with keys:
            "class_logits" : (B, Q, num_classes + 1)
            "mask_logits"  : (B, Q, H, W)  — full input resolution
            "aux_outputs"  : list of dicts with same keys (per-layer)
        """
        if masked_attn_enabled is None:
            masked_attn_enabled = self.training

        B = pixel_values.shape[0]

        # 1. Image-only encoding
        x, h, w = self._run_image_only_blocks(pixel_values)

        # 2. Expand queries to batch
        queries = self.q.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # 3. Joint query + patch encoding with masked attention
        x_patch, q_feats, aux_list = self._run_query_blocks(
            x, queries, h, w, masked_attn_enabled=masked_attn_enabled
        )

        # 4. Apply backbone LayerNorm to patch tokens
        x_patch = self.final_norm(x_patch)    # (B, N, D)

        # 5. Final predictions
        class_logits = self.class_head(q_feats)   # (B, Q, C+1)
        mask_logits_patch = self._predict(q_feats, x_patch, h, w)
        # mask_logits_patch: (B, Q, h, w)

        # Final bilinear upsample to full resolution
        H_in = pixel_values.shape[2]
        W_in = pixel_values.shape[3]
        mask_logits_full = F.interpolate(
            mask_logits_patch,
            size=(H_in, W_in),
            mode="bilinear",
            align_corners=False,
        )  # (B, Q, H, W)

        if self.return_dict:
            # Upsample aux mask_logits to full resolution too
            aux_outputs = []
            for aux in aux_list:
                aux_mask = F.interpolate(
                    aux["mask_logits"],
                    size=(H_in, W_in),
                    mode="bilinear",
                    align_corners=False,
                )
                aux_outputs.append({
                    "class_logits": aux["class_logits"],
                    "mask_logits": aux_mask,
                })

            return {
                "class_logits": class_logits,         # (B, Q, num_classes + 1)
                "mask_logits":  mask_logits_full,     # (B, Q, H, W)
                "aux_outputs":  aux_outputs,
            }
        else:
            return mask_logits_full


# ---------------------------------------------------------------------------
# Mask-annealing scheduler (call set_mask_ratio each epoch)
# ---------------------------------------------------------------------------

class MaskAnnealingScheduler:
    """
    Linearly decays mask_ratio from ``start`` to ``end`` over
    ``num_epochs`` epochs.

    Example
    -------
        scheduler = MaskAnnealingScheduler(model, num_epochs=50)
        for epoch in range(num_epochs):
            scheduler.step(epoch)
            train_one_epoch(model, ...)
    """

    def __init__(
        self,
        model: EoMT,
        num_epochs: int,
        start: float = 1.0,
        end: float = 0.0,
    ) -> None:
        self.model = model
        self.num_epochs = num_epochs
        self.start = start
        self.end = end

    def step(self, epoch: int) -> None:
        ratio = self.start + (self.end - self.start) * epoch / max(self.num_epochs - 1, 1)
        ratio = max(self.end, min(self.start, ratio))
        self.model.set_mask_ratio(ratio)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
from functools import partial

if __name__ == "__main__":
    import sys
    print("Running EoMT smoke test …")

    # Minimal mock of DINOv2FeatureExtractor for CPU testing

    class DINOv2FeatureExtractor(nn.Module):
        def __init__(
            self,
            model_name: str = 'dinov2_vits14',
            layer_indices: list = [5,12,18,24],
            freeze_backbone: bool = True
        ):
            super().__init__()
            self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
            self.layer_indices = layer_indices
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            self.patch_size: int = self.encoder.patch_size
            self.hidden_dim: int = self.encoder.embed_dim
            self.layernorm = nn.LayerNorm(self.hidden_dim)

            self.register_buffer(
                "pixel_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "pixel_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )
            self.backbone = self.encoder
            self.backbone.embeddings = partial(self.embeddings)
            self.backbone.layernorm = partial(self.layernorm)
            self.backbone.num_register_tokens = 1

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

        def embeddings(self, x):
            x = self.encoder.patch_embed(x)
            # Torch hub does not come with CLS token
            x = torch.cat([x, torch.zeros([x.shape[0], 1, x.shape[2]]).to(x.device)], dim=1)
            return x

        def get_block_groups(self, split: str = 'config') -> List[nn.Sequential]:
            """
            Split transformer blocks into `num_groups` roughly equal groups.
            Dont forget to pass by:
                - first inputs go through self.backbone.embeddings(x)
                - after these blocks outputs fo through self.backbone.layernorm(f_vit)
            """
            blocks = list(self.encoder.blocks)
            if split == 'all':
                return blocks
            else:
                num_groups = len(self.layer_indices)

                groups = []
                groups.append(nn.Sequential(*blocks[0: self.layer_indices[0]+1]))
                for i in range(num_groups-1):
                    groups.append(nn.Sequential(*blocks[self.layer_indices[i]+1:self.layer_indices[i+1]+1]))
                return groups
        def forward(self, x):
            return x

    extractor = DINOv2FeatureExtractor()
    model = EoMT(
        encoder=extractor,
        num_classes=2,
        num_queries=100,
        num_query_groups=3,
        mask_annealing=True,
        mask_ratio=0.05,
    )
    model.eval()

    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        out = model(x, masked_attn_enabled=True)

    print(f"  mask_logits  : {out.shape}")    # (2, 20, 128, 128)
    #print(f"  class_logits : {out['class_logits'].shape}")   # (2, 20, 3)
    # print(f"  mask_logits  : {out['mask_logits'].shape}")    # (2, 20, 128, 128)
    #print(f"  aux_outputs  : {len(out['aux_outputs'])} intermediate predictions")
    print("Smoke test passed ✓")
