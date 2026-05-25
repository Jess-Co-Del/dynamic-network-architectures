"""
ijepa_3d_backbone.py — Shared 3D Vision Transformer Backbone
=============================================================
Inflates a pretrained 2D I-JEPA ViT-G/16 into a 3D ViT that accepts
volumetric inputs: [B, C, D, H, W].

Inflation strategy (from VideoMAE / TimeSformer):
  - 2D patch embedding Conv2d(C_in, D_model, 16, 16) is replaced by
    Conv3d(C_in, D_model, (patch_d, 16, 16), stride=(patch_d, 16, 16))
  - The Conv3d weights are initialised by copying the 2D weights and
    dividing by patch_d  (temporal average inflation).
  - All other transformer weights (attention, FFN, LayerNorm) are
    reused as-is — they operate on the token dimension which is
    independent of spatial arrangement.
  - Positional encodings are replaced with learnable 3D sin-cos encodings
    that factorise over (depth, height, width).

This module is imported by all four architecture files.

Dependencies:
    pip install torch transformers

References:
    Tran et al.  "A Closer Look at Spatiotemporal Convolutions." CVPR 2018.
    Bertasius et al. "Is Space-Time Attention All You Need?" ICML 2021.
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023. arXiv:2301.08243
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────────────────
# 3D Sin-Cos Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_sincos_pos_embed(
    grid_d: int,
    grid_h: int,
    grid_w: int,
    embed_dim: int,
    temperature: float = 10000.0,
) -> Tensor:
    """
    Factorised 3D sin-cos positional encoding.

    Splits embed_dim into three equal parts (d, h, w) and applies
    independent sin-cos encodings along each spatial axis.

    Args:
        grid_d:    Number of depth patches  (D / patch_d).
        grid_h:    Number of height patches (H / patch_h = H / 16).
        grid_w:    Number of width patches  (W / patch_w = W / 16).
        embed_dim: Token embedding dimension (must be divisible by 6).

    Returns:
        pos_embed: [1, grid_d * grid_h * grid_w, embed_dim]
    """
    # Distribute embed_dim across 3 axes as evenly as possible,
    # each allocation must be even (sin+cos pairs)
    base    = (embed_dim // 3) // 2 * 2   # largest even number <= embed_dim/3
    dim_h   = base
    dim_w   = base
    dim_d   = embed_dim - dim_w - dim_h   # absorbs the remainder; also forced even
                                           # because embed_dim is always even for ViTs

    assert dim_w > 0 and dim_w % 2 == 0, (
        f"embed_dim={embed_dim} produced invalid dim_w={dim_w}. "
        f"embed_dim must be even."
    )

    def sincos_1d(positions: Tensor, dim: int) -> Tensor:
        half  = dim // 2
        omega = torch.arange(half, dtype=torch.float32) / half
        omega = 1.0 / (temperature ** omega)
        out   = positions.unsqueeze(1) * omega.unsqueeze(0)
        return torch.cat([out.sin(), out.cos()], dim=1)   # [N, dim]

    d_pos = torch.arange(grid_d, dtype=torch.float32)
    h_pos = torch.arange(grid_h, dtype=torch.float32)
    w_pos = torch.arange(grid_w, dtype=torch.float32)

    gd, gh, gw = torch.meshgrid(d_pos, h_pos, w_pos, indexing='ij')

    pe_d = sincos_1d(gd.flatten(), dim_d)   # [N, dim_d]
    pe_h = sincos_1d(gh.flatten(), dim_h)   # [N, dim_h]
    pe_w = sincos_1d(gw.flatten(), dim_w)   # [N, dim_w]

    pos_embed = torch.cat([pe_d, pe_h, pe_w], dim=1)   # [N, embed_dim]
    return pos_embed.unsqueeze(0)                        # [1, N, embed_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 3D Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Volumetric patch embedding via Conv3d.

    Splits a [B, C, D, H, W] volume into non-overlapping 3D patches
    of size (patch_d, patch_h, patch_w) and linearly projects each
    patch to embed_dim.

    Weights are inflated from the pretrained 2D I-JEPA patch embedding:
        2D weight shape: [embed_dim, C_in, patch_h, patch_w]
        3D weight shape: [embed_dim, C_in, patch_d, patch_h, patch_w]
        Inflation:       w_3d = w_2d.unsqueeze(2).repeat(1,1,patch_d,1,1) / patch_d

    Args:
        input_channels:Input image channels (1 for CT/MRI, 3 for RGB video).
        embed_dim:     Token embedding dimension (1408 for ViT-G).
        patch_d:       Depth patch size.
        patch_h:       Height patch size (16 for vitg16).
        patch_w:       Width patch size  (16 for vitg16).
        pretrained_weight: Optional 2D Conv2d weight tensor to inflate.
        pretrained_bias:   Optional 2D Conv2d bias tensor.
    """

    def __init__(
        self,
        input_channels:    int            = 1,
        embed_dim:         int            = 1408,
        patch_d:           int            = 2,
        patch_h:           int            = 16,
        patch_w:           int            = 16,
        pretrained_weight: Optional[Tensor] = None,
        pretrained_bias:   Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.patch_d    = patch_d
        self.patch_h    = patch_h
        self.patch_w    = patch_w
        self.embed_dim  = embed_dim

        self.proj = nn.Conv3d(
            input_channels, embed_dim,
            kernel_size=(patch_d, patch_h, patch_w),
            stride=(patch_d, patch_h, patch_w),
        )

        if pretrained_weight is not None:
            self._inflate_weights(pretrained_weight, pretrained_bias)

    def _inflate_weights(
        self,
        w2d: Tensor,
        b2d: Optional[Tensor] = None,
    ) -> None:
        """
        Inflate 2D Conv2d weights [D, C, h, w] →
        3D Conv3d weights [D, C, patch_d, h, w].

        Division by patch_d ensures that the sum over the depth
        dimension equals the original 2D dot product — preserving
        the pretrained feature statistics at initialisation.
        """
        # Handle channel mismatch (e.g. 1-channel medical images vs 3-channel pretrain)
        C_pretrain = w2d.shape[1]
        C_target   = self.proj.weight.shape[1]

        if C_pretrain != C_target:
            # Average across channels then repeat — standard adaptation strategy
            w2d = w2d.mean(dim=1, keepdim=True).repeat(1, C_target, 1, 1)

        # Inflate: repeat along depth axis then normalise
        w3d = (w2d
               .unsqueeze(2)                                    # [D, C, 1, h, w]
               .repeat(1, 1, self.patch_d, 1, 1)               # [D, C, patch_d, h, w]
               .div_(self.patch_d))                             # normalise

        with torch.no_grad():
            self.proj.weight.copy_(w3d)
            if b2d is not None:
                self.proj.bias.copy_(b2d)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: [B, C, D, H, W]

        Returns:
            tokens:   [B, N_tokens, embed_dim]
            grid_dhw: (grid_d, grid_h, grid_w) — needed for spatial reshape
        """
        B, C, D, H, W = x.shape
        x       = self.proj(x)   # [B, embed_dim, grid_d, grid_h, grid_w]
        grid_d  = D // self.patch_d
        grid_h  = H // self.patch_h
        grid_w  = W // self.patch_w

        tokens = x.flatten(2).transpose(1, 2)   # [B, N, embed_dim]
        return tokens, (grid_d, grid_h, grid_w)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block (reused from 2D I-JEPA, operates on token sequences)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Standard ViT transformer block.
    Identical to I-JEPA's internal blocks — weights are loaded directly.

    Pre-norm architecture: LayerNorm → Attention → residual
                           LayerNorm → FFN       → residual
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
        attn_dropout:float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, N, D]"""
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3D I-JEPA ViT
# ─────────────────────────────────────────────────────────────────────────────

class IJEPAViT3D(nn.Module):
    """
    3D Vision Transformer inflated from pretrained 2D I-JEPA ViT-G/16.

    Architecture:
        PatchEmbed3D  →  [B, N_tokens, 1408]
        + 3D sin-cos positional encoding
        → 40 Transformer blocks (weights copied from I-JEPA)
        → LayerNorm
        → [B, N_tokens, 1408]

    The CLS token is dropped — we work with patch tokens directly.

    Args:
        input_channels:Input volume channels (1 for CT, 3 for RGB video).
        patch_d:       Depth patch size (2 recommended for thin slices).
        patch_h:       Height patch size (16, matching I-JEPA pretrain).
        patch_w:       Width patch size  (16, matching I-JEPA pretrain).
        freeze:        If True, freeze all transformer blocks.
        output_hidden_states: Return all intermediate layer outputs.
        model_id:      HuggingFace I-JEPA checkpoint to inflate from.
    """

    # ViT-G architecture constants
    EMBED_DIM  : int = 1408
    NUM_HEADS  : int = 16
    NUM_LAYERS : int = 40
    MLP_RATIO  : float = 48 / 11   # ≈ 4.363 — ViT-G uses 6144/1408

    def __init__(
        self,
        input_channels:          int  = 1,
        patch_d:              int  = 2,
        patch_h:              int  = 16,
        patch_w:              int  = 16,
        freeze:               bool = True,
        output_hidden_states: bool = False,
        model_id:             str  = "facebook/ijepa_vitg16_22k",
    ) -> None:
        super().__init__()

        self.embed_dim            = self.EMBED_DIM
        self.patch_d              = patch_d
        self.patch_h              = patch_h
        self.patch_w              = patch_w
        self.output_hidden_states = output_hidden_states

        # ── Load pretrained 2D I-JEPA to extract weights ─────────────────
        print(f"[IJEPAViT3D] Loading 2D I-JEPA from '{model_id}' ...")
        model_2d = AutoModel.from_pretrained(model_id)

        # ── 3D Patch Embedding (inflated from 2D) ────────────────────────
        w2d = model_2d.embeddings.patch_embeddings.projection.weight.data
        b2d = model_2d.embeddings.patch_embeddings.projection.bias.data

        self.patch_embed = PatchEmbed3D(
            input_channels    = input_channels,
            embed_dim         = self.EMBED_DIM,
            patch_d           = patch_d,
            patch_h           = patch_h,
            patch_w           = patch_w,
            pretrained_weight = w2d,
            pretrained_bias   = b2d,
        )

        # ── Positional encoding (replaced with learnable 3D sin-cos) ─────
        # Actual grid sizes are computed at forward time from input shape.
        # We register a placeholder; pos_embed is built on first call.
        self.register_buffer('pos_embed', None)
        self._pos_embed_shape: Optional[Tuple[int,int,int]] = None

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim    = self.EMBED_DIM,
                num_heads    = self.NUM_HEADS,
                mlp_ratio    = self.MLP_RATIO,
            )
            for _ in range(self.NUM_LAYERS)
        ])
        self._load_transformer_weights(model_2d)

        self.norm = nn.LayerNorm(self.EMBED_DIM)
        # Copy LayerNorm from 2D model's final norm
        self.norm.weight.data.copy_(model_2d.layernorm.weight.data)
        self.norm.bias.data.copy_(model_2d.layernorm.bias.data)

        del model_2d   # free memory

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        print("[IJEPAViT3D] Inflation complete.")

    def _load_transformer_weights(self, model_2d: nn.Module) -> None:
        """
        Copy attention + FFN weights from 2D I-JEPA transformer blocks.

        HuggingFace IJepaModel stores blocks in model_2d.encoder.layer.
        Each block has:
            attention.attention.{query,key,value}  → MultiheadAttention q,k,v proj
            attention.output.dense                 → out_proj
            intermediate.dense                     → mlp[0]
            output.dense                           → mlp[3]
            layernorm_before / layernorm_after      → norm1 / norm2
        """
        for i, (block_3d, block_2d) in enumerate(
            zip(self.blocks, model_2d.encoder.layer)
        ):
            # ── Attention weights ─────────────────────────────────────────
            attn_2d = block_2d.attention.attention

            # MultiheadAttention in PyTorch stores Q,K,V as single
            # in_proj_weight [3*D, D] and in_proj_bias [3*D]
            D = self.EMBED_DIM
            q_w = attn_2d.query.weight.data
            k_w = attn_2d.key.weight.data
            v_w = attn_2d.value.weight.data
            block_3d.attn.in_proj_weight.data.copy_(
                torch.cat([q_w, k_w, v_w], dim=0)
            )

            q_b = attn_2d.query.bias.data
            k_b = attn_2d.key.bias.data
            v_b = attn_2d.value.bias.data
            block_3d.attn.in_proj_bias.data.copy_(
                torch.cat([q_b, k_b, v_b], dim=0)
            )

            out_w = block_2d.attention.output.dense.weight.data
            out_b = block_2d.attention.output.dense.bias.data
            block_3d.attn.out_proj.weight.data.copy_(out_w)
            block_3d.attn.out_proj.bias.data.copy_(out_b)

            # ── FFN weights ───────────────────────────────────────────────
            block_3d.mlp[0].weight.data.copy_(block_2d.intermediate.dense.weight.data)
            block_3d.mlp[0].bias.data.copy_(block_2d.intermediate.dense.bias.data)
            block_3d.mlp[3].weight.data.copy_(block_2d.output.dense.weight.data)
            block_3d.mlp[3].bias.data.copy_(block_2d.output.dense.bias.data)

            # ── LayerNorm weights ─────────────────────────────────────────
            block_3d.norm1.weight.data.copy_(block_2d.layernorm_before.weight.data)
            block_3d.norm1.bias.data.copy_(block_2d.layernorm_before.bias.data)
            block_3d.norm2.weight.data.copy_(block_2d.layernorm_after.weight.data)
            block_3d.norm2.bias.data.copy_(block_2d.layernorm_after.bias.data)

    def _get_pos_embed(
        self, grid_d: int, grid_h: int, grid_w: int
    ) -> Tensor:
        """
        Returns 3D sin-cos positional encoding, building it on first call
        or when the grid size changes.
        """
        current_shape = (grid_d, grid_h, grid_w)
        if self._pos_embed_shape != current_shape:
            self._pos_embed_shape = current_shape
            pe = build_3d_sincos_pos_embed(
                grid_d, grid_h, grid_w, self.EMBED_DIM
            )
            # Store on same device as model
            device = next(self.parameters()).device
            self.register_buffer('pos_embed', pe.to(device))
        return self.pos_embed

    def forward(self, x: Tensor) -> dict:
        """
        Args:
            x: [B, C, D, H, W]

        Returns dict with:
            'last_hidden_state': [B, N_tokens, 1408]
            'hidden_states':     tuple of [B, N_tokens, 1408]  (if requested)
            'grid_dhw':          (grid_d, grid_h, grid_w)
        """
        # ── Patch embedding ───────────────────────────────────────────────
        tokens, grid_dhw = self.patch_embed(x)    # [B, N, 1408]
        grid_d, grid_h, grid_w = grid_dhw

        # ── Add 3D positional encoding ────────────────────────────────────
        pos = self._get_pos_embed(grid_d, grid_h, grid_w)   # [1, N, 1408]
        tokens = tokens + pos

        # ── Transformer blocks ────────────────────────────────────────────
        hidden_states = []
        for block in self.blocks:
            tokens = block(tokens)
            if self.output_hidden_states:
                hidden_states.append(tokens)

        tokens = self.norm(tokens)   # [B, N, 1408]

        out = {
            'last_hidden_state': tokens,
            'grid_dhw':          grid_dhw,
        }
        if self.output_hidden_states:
            out['hidden_states'] = tuple(hidden_states)

        return out

    def tokens_to_spatial(self, tokens: Tensor, grid_dhw: Tuple[int,int,int]) -> Tensor:
        """
        Reshape flat token sequence to volumetric feature map.

        Args:
            tokens:   [B, N, D]  where N = grid_d * grid_h * grid_w
            grid_dhw: (grid_d, grid_h, grid_w)

        Returns:
            spatial: [B, D, grid_d, grid_h, grid_w]
        """
        B, N, embed = tokens.shape
        gd, gh, gw  = grid_dhw
        return (tokens
                .permute(0, 2, 1)                      # [B, embed, N]
                .reshape(B, embed, gd, gh, gw))        # [B, embed, gd, gh, gw]
