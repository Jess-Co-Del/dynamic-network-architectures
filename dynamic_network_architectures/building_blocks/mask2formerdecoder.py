"""
Mask2Former Decoder for Semantic/Panoptic Segmentation

Architecture:
  - Backbone:       facebook/ijepa_vitg16_22k  (frozen or fine-tuned)
  - Pixel Decoder:  Multi-scale Deformable Attention FPN (simplified)
  - Transformer Decoder: Mask2Former-style with masked cross-attention
  - Heads:          class logits + binary mask per query

References:
  - Mask2Former: Cheng et al., 2022 (arXiv:2112.01527)
  - I-JEPA:      Assran et al., 2023 (arXiv:2301.08243)
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # Hungarian matching
from transformers import AutoImageProcessor
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,          # deformable-attention pixel decoder
    Mask2FormerTransformerModule, # the transformer decoder with masked attn
)
import transformers.initialization as init


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PIXEL DECODER
#     Takes multi-scale ViT features → produces upsampled feature pyramid
#     + a high-resolution (H/4) pixel embedding map for mask generation
# ─────────────────────────────────────────────────────────────────────────────

class PixelDecoder(nn.Module):
    """
    Lightweight FPN-style pixel decoder.

    Because ViT-p14 produces a single spatial resolution (14×14),
    we simulate a feature pyramid by tapping 4 intermediate layers
    and progressively upsampling them top-down.

    Output scales (for 224×224 input):
        P5: 14×14   (stride 16 — deepest, most semantic)
        P4: 28×28   (stride 8)
        P3: 56×56   (stride 4)
        P2: 112×112 (stride 2  — finest, used for mask dot-product)
    """

    def __init__(self, in_channels=1408, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim

        # 1×1 lateral projections — one per tapped ViT layer
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, 1),
                nn.GroupNorm(32, feature_dim),
            )
            for _ in range(4)  # P2, P3, P4, P5
        ])

        # 3×3 output convolutions after top-down fusion
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        # Final upsampling to stride-2 (112×112) for pixel embeddings
        # Used in mask generation via dot product with query embeddings
        self.mask_features = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim, 2, stride=2),  # 56→112
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: list of 4 tensors [B, hidden_dim, 14, 14]
                                  from ViT layers (L/4, L/2, 3L/4, L)
        Returns:
            mask_features:  [B, 256, 112, 112]  — for mask dot-product
            decoder_memory: list of 3 tensors at different scales
                            [(B,256,14,14), (B,256,28,28), (B,256,56,56)]
        """
        # Memory tensors for cross-attention: P5, P4, P3 (14, 28, 56)
        return multi_scale_features[0], (multi_scale_features[1], multi_scale_features[2], multi_scale_features[3])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MASKED CROSS-ATTENTION
#     The core Mask2Former innovation: each query only attends to spatial
#     positions where it previously predicted a foreground mask.
#     This forces locality and prevents queries from "stealing" each other's regions.
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCrossAttention(nn.Module):
    """
    Cross-attention with a binary attention mask derived from the
    previous iteration's mask prediction.

    For query i at position (h,w): if the previous mask prediction
    at (h,w) < 0.5, that position is masked out (set to -inf before softmax).
    This forces each query to specialize on its own spatial region.
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries, memory, attention_mask=None):
        """
        Args:
            queries:        [B, N_queries, D]
            memory:         [B, H*W, D]   — flattened spatial features
            attention_mask: [B*heads, N_queries, H*W] binary mask
                            True = IGNORE this position (PyTorch convention)
        Returns:
            queries:  [B, N_queries, D]  (residual connection applied)
        """
        # PyTorch MHA expects attn_mask shape [B*heads, tgt_len, src_len]
        # or [tgt_len, src_len] — we pass per-head mask
        residual = queries
        out, _ = self.attn(
            query=queries,
            key=memory,
            value=memory,
            attn_mask=attention_mask,   # None in first layer (no prior mask)
        )
        return self.norm(residual + out)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MASK2FORMER TRANSFORMER DECODER LAYER
#     Each layer: Masked Cross-Attn → Self-Attn → FFN
#     + predicts intermediate masks for the next layer's attention mask
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=2048, dropout=0.0):
        super().__init__()

        # ① Masked cross-attention — queries attend to pixel memory
        self.masked_cross_attn = MaskedCrossAttention(embed_dim, num_heads, dropout)

        # ② Self-attention — queries communicate globally with each other
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(embed_dim)

        # ③ FFN — standard 2-layer MLP
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, queries, memory, attention_mask=None):
        """
        Args:
            queries:        [B, N_queries, D]
            memory:         [B, H*W, D]
            attention_mask: [B*heads, N_queries, H*W]  or None
        Returns:
            queries: [B, N_queries, D]
        """
        # ① Masked cross-attention
        queries = self.masked_cross_attn(queries, memory, attention_mask)

        # ② Self-attention
        residual = queries
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm_sa(residual + sa_out)

        # ③ FFN
        residual = queries
        queries = self.norm_ffn(residual + self.ffn(queries))

        return queries


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL MASK2FORMER DECODER
#     Stacks L decoder layers, cycling through multi-scale memory.
#     At each layer, computes intermediate mask predictions used as
#     attention masks for the next layer.
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerDecoder(nn.Module):
    def __init__(
        self,
        config: Mask2FormerConfig, backbone_out_ch: int):
        super().__init__()
        self.config = config
        self.transformer_decoder = Mask2FormerTransformerModule(in_features=backbone_out_ch, config=config)


    def forward(self, mask_features, multi_scale_memory):
        """
        Args:
            mask_features:       [B, 256, 112, 112]  high-res pixel embeddings
            multi_scale_memory:  list of [B, 256, H, W] at scales 14, 28, 56

        Returns:
            pred_logits: [B, num_queries, num_classes+1]  — final class predictions
            pred_masks:  [B, num_queries, H, W]            — final mask predictions
            aux_outputs: list of (logits, masks) per intermediate layer
                         used for auxiliary loss during training
        """
        
        # Transformer decoder with masked attention
        mask2former_output = self.transformer_decoder(
            multi_scale_features=multi_scale_memory,
            mask_features=mask_features,
            output_hidden_states=True,
        )

        #mask_logits = transformer_out.masks_queries_logits[-1]  # (B, N,H,W)

        return mask2former_output

    @staticmethod
    def _build_attention_mask(pred_mask, target_hw, num_heads, B):
        """
        Convert predicted mask [B, Q, H', W'] → binary attention mask
        [B*num_heads, Q, H*W] for MHA.

        Regions where pred_mask < 0 are "background" → masked out (True = ignore).
        """
        Q = pred_mask.shape[1]
        # Resize mask to match memory spatial size
        mask_resized = F.interpolate(
            pred_mask, size=target_hw, mode='bilinear', align_corners=False
        )  # [B, Q, H, W]

        # Binarize: True where we should IGNORE (background regions)
        attn_mask = (mask_resized.sigmoid().flatten(2) < 0.5)  # [B, Q, H*W]

        # Expand for multi-head: [B*heads, Q, H*W]
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(B * num_heads, Q, -1)

        # Safety: if ALL positions are masked for a query, unmask everything
        # (prevents NaN from softmax over all -inf)
        fully_masked = attn_mask.all(dim=-1, keepdim=True)
        attn_mask = attn_mask.masked_fill(fully_masked, False)

        return attn_mask


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HUNGARIAN MATCHER
#     During training, we need to optimally match predicted queries to
#     ground-truth masks. This is the "set prediction" paradigm from DETR.
#     Uses scipy's linear_sum_assignment (Hungarian algorithm).
# ─────────────────────────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_mask=5.0, cost_dice=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask  = cost_mask
        self.cost_dice  = cost_dice

    @torch.no_grad()
    def forward(self, pred_logits, pred_masks, targets):
        """
        Args:
            pred_logits: [B, Q, num_classes+1]
            pred_masks:  [B, Q, H, W]
            targets:     list of dicts, each with:
                           'labels':  [num_gt] int tensor
                           'masks':   [num_gt, H, W] float tensor
        Returns:
            List of (query_indices, gt_indices) tuples, one per image
        """
        B, Q = pred_logits.shape[:2]
        indices = []

        for b in range(B):
            tgt_labels = targets[b]['labels']     # [num_gt]
            tgt_masks  = targets[b]['masks']      # [num_gt, H, W]
            num_gt     = len(tgt_labels)

            if num_gt == 0:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            # Class cost: -log prob of the correct class for each (query, gt) pair
            pred_prob = pred_logits[b].softmax(-1)       # [Q, C+1]
            cost_cls  = -pred_prob[:, tgt_labels]        # [Q, num_gt]

            # Resize predicted masks to match GT resolution
            pred_m = F.interpolate(
                pred_masks[b].unsqueeze(0),
                size=tgt_masks.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(0).sigmoid()  # [Q, H, W]

            pred_flat = pred_m.flatten(1)    # [Q, H*W]
            tgt_flat  = tgt_masks.flatten(1) # [num_gt, H*W]

            # Binary cross-entropy cost
            cost_bce = (
                - tgt_flat.unsqueeze(0) * F.logsigmoid(pred_flat.unsqueeze(1))
                - (1 - tgt_flat.unsqueeze(0)) * F.logsigmoid(-pred_flat.unsqueeze(1))
            ).mean(-1)  # [Q, num_gt]

            # Dice cost — penalizes mask shape mismatch more than BCE
            numerator   = 2 * (pred_flat.unsqueeze(1) * tgt_flat.unsqueeze(0)).sum(-1)
            denominator = pred_flat.unsqueeze(1).sum(-1) + tgt_flat.unsqueeze(0).sum(-1)
            cost_dice   = 1 - (numerator + 1) / (denominator + 1)  # [Q, num_gt]

            # Combined cost matrix
            C = (self.cost_class * cost_cls
               + self.cost_mask  * cost_bce
               + self.cost_dice  * cost_dice).cpu().numpy()  # [Q, num_gt]

            row_idx, col_idx = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.long),
                torch.as_tensor(col_idx, dtype=torch.long)
            ))

        return indices


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING LOSS
#     Combines: classification CE + mask BCE + mask Dice
#     Applied at EVERY decoder layer (auxiliary deep supervision)
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerLoss(nn.Module):
    def __init__(self, num_classes=150, cost_class=1.0,
                 cost_mask=5.0, cost_dice=2.0,
                 weight_ce=2.0, weight_mask=5.0, weight_dice=5.0):
        super().__init__()
        self.matcher     = HungarianMatcher(cost_class, cost_mask, cost_dice)
        self.weight_ce   = weight_ce
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.num_classes = num_classes

        # Background class gets lower weight (standard practice)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = 0.1   # "no-object" class
        self.register_buffer('empty_weight', empty_weight)

    def _dice_loss(self, pred, target):
        pred   = pred.sigmoid().flatten(1)
        target = target.flatten(1)
        num    = 2 * (pred * target).sum(1)
        den    = pred.sum(1) + target.sum(1)
        return (1 - (num + 1) / (den + 1)).mean()

    def loss_for_layer(self, pred_logits, pred_masks, targets, indices):
        B = pred_logits.shape[0]
        losses = {}

        # ── Classification loss ───────────────────────────────────────────
        # Assign "no object" label to unmatched queries
        tgt_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.long, device=pred_logits.device
        )
        for b, (q_idx, gt_idx) in enumerate(indices):
            if len(q_idx):
                tgt_classes[b, q_idx] = targets[b]['labels'][gt_idx]

        losses['ce'] = F.cross_entropy(
            pred_logits.transpose(1, 2),   # [B, C+1, Q]
            tgt_classes,
            weight=self.empty_weight
        )

        # ── Mask losses (only on matched queries) ────────────────────────
        bce_loss  = torch.tensor(0.0, device=pred_logits.device)
        dice_loss = torch.tensor(0.0, device=pred_logits.device)
        num_matched = 0

        for b, (q_idx, gt_idx) in enumerate(indices):
            if len(q_idx) == 0:
                continue
            # Resize GT masks to predicted resolution
            gt_masks = F.interpolate(
                targets[b]['masks'][gt_idx].unsqueeze(1).float(),
                size=pred_masks.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(1)  # [num_matched, H, W]

            pred_m = pred_masks[b, q_idx]  # [num_matched, H, W]

            bce_loss  += F.binary_cross_entropy_with_logits(pred_m, gt_masks)
            dice_loss += self._dice_loss(pred_m, gt_masks)
            num_matched += len(q_idx)

        if num_matched > 0:
            losses['bce']  = bce_loss  / num_matched
            losses['dice'] = dice_loss / num_matched
        else:
            losses['bce']  = bce_loss
            losses['dice'] = dice_loss

        return losses

    def forward(self, pred_logits, pred_masks, aux_outputs, targets):
        """
        Args:
            pred_logits:  [B, Q, C+1]
            pred_masks:   [B, Q, H, W]
            aux_outputs:  list of {'pred_logits': ..., 'pred_masks': ...}
            targets:      list of {'labels': ..., 'masks': ...}
        Returns:
            total_loss (scalar), loss_dict (for logging)
        """
        # Match on final predictions
        indices = self.matcher(pred_logits, pred_masks, targets)

        # Final layer loss
        losses = self.loss_for_layer(pred_logits, pred_masks, targets, indices)

        total = (self.weight_ce   * losses['ce']
               + self.weight_mask * losses['bce']
               + self.weight_dice * losses['dice'])

        # Auxiliary losses (intermediate layers) — same weight, same matching
        for aux in aux_outputs:
            aux_indices = self.matcher(aux['pred_logits'], aux['pred_masks'], targets)
            aux_losses  = self.loss_for_layer(
                aux['pred_logits'], aux['pred_masks'], targets, aux_indices
            )
            total += (self.weight_ce   * aux_losses['ce']
                    + self.weight_mask * aux_losses['bce']
                    + self.weight_dice * aux_losses['dice'])

        loss_dict = {k: v.item() for k, v in losses.items()}
        return total, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL DECODER — glues everything together
# ─────────────────────────────────────────────────────────────────────────────

class Mask2Former(nn.Module):
    """
    Mask2Former decoder.

    Supports:
      - Semantic segmentation  (num_queries ≈ num_classes, no instance distinction)
      - Instance segmentation  (num_queries > num_classes, e.g. 100)
      - Panoptic segmentation  (num_queries = 100-200, with thing/stuff labels)
    """

    def __init__(
        self,
        hidden_dim:   int = 1,
        num_classes   = 3,
        num_queries   = 3,
        image_size    = 224,
        patch_size    = 16,
        deep_supervision: bool = False
    ):
        super().__init__()

        # ── Pixel Decoder ────────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(
            in_channels=hidden_dim,
            feature_dim=hidden_dim
        )

        # ── Transformer Decoder ──────────────────────────────────────────
        config = make_mask2former_config(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            mask_feature_size=hidden_dim,
            decoder_layers=9,
        )

        self.transformer_decoder = Mask2FormerDecoder(
            config, backbone_out_ch=hidden_dim
        )

        # ── Prediction head for classes -──────────────────────────────────
        # self.class_head = nn.Linear(hidden_dim, num_classes + 1)   # +1 for no-object

        self.image_processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-ade-semantic"
        )
        self.image_size = image_size
        self.grid_size  = image_size // patch_size   # 14

    def forward(self, multi_scale, targets=None):
        """
        Args:
            multi_scale: list of [B, hidden_dim, grid_size, grid_size] tensors
            targets:      list of {'labels': ..., 'masks': ...}  (training only)

        Returns (inference):
            pred_logits: [B, Q, num_classes+1]
            pred_masks:  [B, Q, 224, 224]   (upsampled to full res)

        Returns (training):
            loss, loss_dict
        """
        #print([output.shape for output in multi_scale])

        # ── 1. Pixel decoder → high-res pixel embeddings + memory ────────
        mask_features, decoder_memory = self.pixel_decoder(multi_scale)

        # ── 2. Transformer decoder → query embeddings + mask predictions ─
        mask2former_output = self.transformer_decoder(
            mask_features, decoder_memory
        )
        mask_logits = mask2former_output.masks_queries_logits[-1]
        #class_logits = self.class_head(mask2former_output.last_hidden_state)

        # ── 3. Upsample masks to full image resolution ───────────────────
        mask_logits_full = F.interpolate(
            mask_logits, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        )  # (B, N, H, W)

        return mask_logits_full

    def predict_semantic(self, masks_queries_logits, class_logits):
        """
        Inference helper: collapses queries → per-pixel semantic labels.
        Standard Mask2Former inference procedure.
        Partially copied from 
        transformers/models/mask2former/modeling_mask2former.py
        """
        target_size=(self.image_size, self.image_size)

        # [B, Q, C+1] × [B, Q, H, W] → [B, C, H, W]
        mask_probs  = masks_queries_logits.sigmoid()               # [B, Q, H, W]
        mask_logits_up = F.interpolate(
            mask_probs, size=target_size, mode="bilinear", align_corners=False
        )  # (B, N, H, W)
        class_probs = class_logits.softmax(-1)[..., :-1]            # [B, Q, C] drop no-obj
        # Weighted sum: each query contributes its class prob × mask prob
        seg_logits  = torch.einsum('bqc,bqhw->bchw', class_probs, mask_logits_up)

        return seg_logits


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL Mask2Former — Hugging Face implementation
# ─────────────────────────────────────────────────────────────────────────────

def make_mask2former_config(
    num_classes: int,
    num_queries: int = 100,
    hidden_dim: int = 256,
    encoder_layers: int = 6,
    decoder_layers: int = 9,        # paper uses 9 (3 groups × 3 scales)
    num_attention_heads: int = 8,
    dim_feedforward: int = 2048,
    mask_feature_size: int = 256,
    feature_strides: List[int] = None,
    use_auxiliary_loss: bool = True,
    pre_norm: bool = False,
) -> Mask2FormerConfig:
    """
    Build a Mask2FormerConfig without a built-in backbone.
    Set backbone_config=None so the HF model doesn't instantiate Swin —
    you will feed feature maps from your own backbone directly.
 
    num_classes   : number of semantic/instance classes (WITHOUT the null class;
                    HF adds +1 internally for the "no-object" slot)
    num_queries   : N learnable object queries (100 for COCO, 150 for ADE20K)
    hidden_dim    : query/feature channel dimension throughout the decoder (256)
    decoder_layers: total transformer decoder layers; must be divisible by 3
                    because Mask2Former cycles through 3 feature scales per group
    """
    if feature_strides is None:
        feature_strides = [4, 8, 16, 32]
 
    cfg = Mask2FormerConfig(
        backbone_config=None,          # ← we bring our own backbone
        feature_size=hidden_dim,
        mask_feature_size=mask_feature_size,
        hidden_dim=hidden_dim,
        encoder_feedforward_dim=dim_feedforward,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_attention_heads=num_attention_heads,
        dim_feedforward=dim_feedforward,
        num_queries=num_queries,
        use_auxiliary_loss=use_auxiliary_loss,
        pre_norm=pre_norm,
        feature_strides=feature_strides,
        # loss weights (paper defaults)
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        # point sampling for efficient loss (paper: 12544 = 112×112)
        train_num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    # num_labels is set separately; HF reads it from here for the class head
    cfg.num_labels = num_classes
    return cfg


class Mask2FormerDecoderHF(nn.Module):
    """
    HUGGINGFACE mport from transformers library.
    """
    def __init__(
        self,
        hidden_dim:   int = 1,
        num_classes   = 3,
        num_queries   = 3,
        feature_size    = 256,
        image_size    = 16,
        deep_supervision: bool = False):
        super().__init__()
        
        self.image_size = image_size

        config = make_mask2former_config(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=feature_size,
            decoder_layers=9,
        )

        self.decoder = Mask2FormerDecoder(config, backbone_out_ch=hidden_dim)

    def forward(self, multiscale_features: List[torch.Tensor]):
        # ── 1. Decoder ───────────────────────────────────────────────────
        print('CHECKING', [features.shape for features in multiscale_features])
        mask_logits = self.decoder(multiscale_features)
        print('CHECKING', mask_logits.shape)
        # ── 2. Upsample masks to full image resolution ───────────────────
        pred_masks_full = F.interpolate(
            mask_logits, size=self.image_size,
            mode='bilinear', align_corners=False
        )   # [B, Q, image_size, image_size]
        return pred_masks_full


class Mask2FormerDecoder_FromScratch(nn.Module):
    """
    Instantiates only the pixel decoder + transformer decoder from a config,
    no pretrained weights.  Designed for training from scratch or fine-tuning
    where you initialise the backbone separately.

    The pixel decoder is the Mask2FormerPixelDecoder (deformable attention encoder).
    The transformer decoder is Mask2FormerMaskedAttentionDecoder.

    Parameters
    ----------
    config          : Mask2FormerConfig built with make_mask2former_config()
    backbone_out_ch : channel depth of each backbone feature level
    """

    def __init__(self, config: Mask2FormerConfig, backbone_out_ch: int):
        super().__init__()
        self.config = config
        C = config.feature_size       # 256 by default
 
        # Input projections: backbone channels → C
        # Mask2Former pixel decoder expects 3 scales: 1/8, 1/16, 1/32
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_out_ch, C, 1, bias=False),
                nn.GroupNorm(32, C),
            )
            for _ in range(3)   # one per scale: 1/32, 1/16, 1/8
        ])
 
        # 1/4 mask feature projection (highest resolution, no deformable attn)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(backbone_out_ch, C, 1, bias=False),
            nn.GroupNorm(32, C),
        )
 
        # Pixel decoder (deformable DETR encoder as multi-scale feature refiner)
        self.pixel_decoder = Mask2FormerPixelDecoder(config, feature_channels=[C, C, C])
 
        # Transformer decoder (masked attention)
        self.transformer_decoder = Mask2FormerTransformerModule(in_features=C, config=config)
 
        # Class prediction head: D → num_classes + 1
        self.class_head = nn.Linear(C, config.num_labels + 1)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        init.constant_(input_projection.bias, 0)

        elif isinstance(module, Mask2FormerPixelDecoder):
            init.normal_(module.level_embed, std=0)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])

        elif isinstance(module, Mask2FormerLoss):
            empty_weight = torch.ones(module.num_labels + 1)
            empty_weight[-1] = module.eos_coef
            init.copy_(module.empty_weight, empty_weight)

        if hasattr(module, "reference_points"):
            init.xavier_uniform_(module.reference_points.weight, gain=1.0)
            init.constant_(module.reference_points.bias, 0.0)

    def forward(
        self,
        backbone_features: List[torch.Tensor],
        # expected: [f_1_4, f_1_8, f_1_16, f_1_32]  finest → coarsest
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Returns
        -------
        class_logits : (B, N, num_classes+1)
        mask_logits  : (B, N, H/4, W/4)
        aux_logits   : list of (B, N, H/4, W/4) from intermediate decoder layers
        """
        f_1_4, f_1_8, f_1_16, f_1_32 = backbone_features

        # Project each scale to C channels
        p_1_32 = self.input_proj[0](f_1_32)
        p_1_16 = self.input_proj[1](f_1_16)
        p_1_8  = self.input_proj[2](f_1_8)
        mask_features = self.mask_proj(f_1_4)   # (B, C, H/4, W/4)

        # Pixel decoder: refine multi-scale features
        # HF expects features in order [coarse, ..., fine] = [1/32, 1/16, 1/8]
        pixel_dec_out = self.pixel_decoder(
            features=[p_1_32, p_1_16, p_1_8],
            output_hidden_states=False,
        )
        # mask_features come from the 1/4 projection, not the pixel decoder
        # (the pixel decoder only processes 1/32..1/8 and returns refined versions)
        multi_scale_features = pixel_dec_out.multi_scale_features

        # Transformer decoder with masked attention
        transformer_out = self.transformer_decoder(
            multi_scale_features=multi_scale_features,
            mask_features=mask_features,
            output_hidden_states=True,
        )

        #queries = transformer_out.last_hidden_state          # (B, N, C)
        #class_logits = self.class_head(queries)              # (B, N, C+1)
        mask_logits = transformer_out.masks_queries_logits[-1]  # tuple of B Tensors shaped (N,H,W)
        # mask_logits  = mask_logits_all[-1]                   # final layer
        #aux_logits   = list(mask_logits_all[:-1])
 
        return mask_logits
