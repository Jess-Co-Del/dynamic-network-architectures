"""
I-JEPA + Mask2Former Decoder for Semantic/Panoptic Segmentation

Architecture:
  - Backbone:       facebook/ijepa_vitg16_22k  (frozen or fine-tuned)
  - Pixel Decoder:  Multi-scale Deformable Attention FPN (simplified)
  - Transformer Decoder: Mask2Former-style with masked cross-attention
  - Heads:          class logits + binary mask per query

References:
  - Mask2Former: Cheng et al., 2022 (arXiv:2112.01527)
  - I-JEPA:      Assran et al., 2023 (arXiv:2301.08243)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment  # Hungarian matching


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PIXEL DECODER
#     Takes multi-scale ViT features → produces upsampled feature pyramid
#     + a high-resolution (H/4) pixel embedding map for mask generation
# ─────────────────────────────────────────────────────────────────────────────

class PixelDecoder(nn.Module):
    """
    Lightweight FPN-style pixel decoder.

    Because ViT-G produces a single spatial resolution (14×14),
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
            multi_scale_features: list of 4 tensors [B, 1408, 14, 14]
                                  from ViT layers (9, 19, 29, 39)
        Returns:
            mask_features:  [B, 256, 112, 112]  — for mask dot-product
            decoder_memory: list of 3 tensors at different scales
                            [(B,256,14,14), (B,256,28,28), (B,256,56,56)]
        """
        # Project all lateral features
        laterals = [self.lateral[i](f) for i, f in enumerate(multi_scale_features)]
        # laterals[3] = deepest (14×14), laterals[0] = shallowest (14×14)
        # We upsample from deep → shallow

        # Top-down fusion
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            # Double spatial resolution at each step via interpolation scale
            # Since all ViT features are 14×14, we force resolution doubling
            target_h = laterals[i].shape[-2] * (2 ** (len(laterals) - 2 - i))
            target_w = laterals[i].shape[-1] * (2 ** (len(laterals) - 2 - i))
            upsampled = F.interpolate(laterals[i + 1],
                                      size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
            laterals[i] = laterals[i] + F.interpolate(
                upsampled, size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=False
            )

        # Apply output convs with progressive upsampling
        outs = []
        for i, lat in enumerate(laterals):
            scale_factor = 2 ** i  # 1×, 2×, 4×, 8×
            upsampled = F.interpolate(lat, scale_factor=scale_factor,
                                      mode='bilinear', align_corners=False)
            outs.append(self.output_convs[i](upsampled))
        # outs: [14×14, 28×28, 56×56, 112×112]

        # High-res pixel embeddings for mask generation (stride-2)
        mask_features = self.mask_features(outs[2])  # 56×56 → 112×112

        # Memory tensors for cross-attention: P5, P4, P3 (14, 28, 56)
        decoder_memory = [outs[0], outs[1], outs[2]]

        return mask_features, decoder_memory


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
        num_classes=150,       # ADE20K; 133 for COCO panoptic
        num_queries=100,
        embed_dim=256,
        num_heads=8,
        ffn_dim=2048,
        num_layers=9,          # Mask2Former paper uses 9 layers
        dropout=0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim   = embed_dim
        self.num_layers  = num_layers

        # Learnable query embeddings (content) and query positional encodings
        self.query_feat  = nn.Embedding(num_queries, embed_dim)
        self.query_embed = nn.Embedding(num_queries, embed_dim)  # positional

        # 9 decoder layers, cycling through 3 memory scales
        # layer 0 → P5(14×14), layer 1 → P4(28×28), layer 2 → P3(56×56),
        # layer 3 → P5 again, ...
        self.layers = nn.ModuleList([
            Mask2FormerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Prediction heads (shared across layers for intermediate supervision)
        self.class_head = nn.Linear(embed_dim, num_classes + 1)   # +1 = "no object"
        self.mask_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)

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
        B = mask_features.shape[0]

        # Initialize queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)   # [B, Q, D]
        pos_enc = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        queries = queries + pos_enc

        # Flatten and pre-compute pixel embeddings for mask dot-product
        # mask_features: [B, D, 112, 112] → [B, 112*112, D]
        B, D, Hm, Wm = mask_features.shape
        mask_feat_flat = mask_features.view(B, D, -1).permute(0, 2, 1)   # [B, 12544, 256]

        aux_outputs = []
        prev_mask = None  # No attention mask for the first layer

        for i, layer in enumerate(self.layers):
            # Cycle through memory scales: layer 0→P5, 1→P4, 2→P3, 3→P5, ...
            mem = multi_scale_memory[i % len(multi_scale_memory)]
            Bm, Dm, Hk, Wk = mem.shape
            memory_flat = mem.view(Bm, Dm, -1).permute(0, 2, 1)  # [B, H*W, D]

            # Build attention mask from previous mask prediction
            attn_mask = self._build_attention_mask(
                prev_mask, target_hw=(Hk, Wk),
                num_heads=8, B=B
            ) if prev_mask is not None else None

            # Run decoder layer
            queries = layer(queries, memory_flat, attn_mask)

            # Intermediate mask prediction (used as next layer's attention mask)
            normed_q = self.decoder_norm(queries)
            mask_emb = self.mask_embed(normed_q)              # [B, Q, D]
            pred_mask = torch.einsum(                          # dot product
                'bqd,bpd->bqp', mask_emb, mask_feat_flat
            ).reshape(B, self.num_queries, Hm, Wm)            # [B, Q, 112, 112]

            prev_mask = pred_mask.detach()  # stop gradient for mask → attention conversion

            # Save intermediate predictions for auxiliary loss
            pred_cls = self.class_head(normed_q)
            aux_outputs.append({'pred_logits': pred_cls, 'pred_masks': pred_mask})

        # Final predictions = last aux output
        pred_logits = aux_outputs[-1]['pred_logits']   # [B, Q, num_classes+1]
        pred_masks  = aux_outputs[-1]['pred_masks']    # [B, Q, 112, 112]

        return pred_logits, pred_masks, aux_outputs[:-1]

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
# 7.  FULL MODEL — glues everything together
# ─────────────────────────────────────────────────────────────────────────────

class IJEPAMask2Former(nn.Module):
    """
    I-JEPA ViT-G/16 backbone + Mask2Former decoder.

    Supports:
      - Semantic segmentation  (num_queries ≈ num_classes, no instance distinction)
      - Instance segmentation  (num_queries > num_classes, e.g. 100)
      - Panoptic segmentation  (num_queries = 100-200, with thing/stuff labels)
    """

    OUT_LAYER_INDICES = (9, 19, 29, 39)   # tap every 10th layer of ViT-G (40 layers)

    def __init__(
        self,
        input_channels:   int = 1,
        num_classes   = 2,   # ADE20K semantic; 133 for COCO panoptic
        num_queries   = 100,
        embed_dim     = 256,
        freeze_backbone = True,
        image_size    = 224,
        patch_size    = 16,
        deep_supervision: bool = False
    ):
        super().__init__()
        hidden_size = 1408   # ViT-G hidden dim

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(
            "facebook/ijepa_vitg16_22k",
            output_hidden_states=True
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Pixel Decoder ────────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(in_channels=hidden_size,
                                          feature_dim=embed_dim)

        # ── Transformer Decoder ──────────────────────────────────────────
        self.transformer_decoder = Mask2FormerDecoder(
            num_classes  = num_classes,
            num_queries  = num_queries,
            embed_dim    = embed_dim,
            num_layers   = 9,
        )

        self.image_size = image_size
        self.grid_size  = image_size // patch_size   # 14

    def _extract_multiscale_features(self, pixel_values):
        """
        Run I-JEPA backbone and tap 4 intermediate layer outputs.

        Returns: list of [B, 1408, 14, 14] tensors
        """
        B = pixel_values.shape[0]
        outputs = self.backbone(pixel_values=pixel_values)
        # hidden_states: tuple of length num_layers+1

        features = []
        for layer_idx in self.OUT_LAYER_INDICES:
            tokens  = outputs.hidden_states[layer_idx]     # [B, 197, 1408]
            spatial = (tokens  # [:, 1:, :]                    # drop CLS → [B, 196, 1408]
                       .permute(0, 2, 1)                   # [B, 1408, 196]
                       .reshape(B, -1, self.grid_size, self.grid_size))  # [B, 1408, 14, 14]
            features.append(spatial)
        return features

    def forward(self, pixel_values, targets=None):
        """
        Args:
            pixel_values: [B, 3, 224, 224]
            targets:      list of {'labels': ..., 'masks': ...}  (training only)

        Returns (inference):
            pred_logits: [B, Q, num_classes+1]
            pred_masks:  [B, Q, 224, 224]   (upsampled to full res)

        Returns (training):
            loss, loss_dict
        """
        # ── 1. Extract multi-scale features from I-JEPA ──────────────────
        multi_scale = self._extract_multiscale_features(pixel_values)

        # ── 2. Pixel decoder → high-res pixel embeddings + memory ────────
        mask_features, decoder_memory = self.pixel_decoder(multi_scale)

        # ── 3. Transformer decoder → query embeddings + mask predictions ─
        pred_logits, pred_masks, aux_outputs = self.transformer_decoder(
            mask_features, decoder_memory
        )

        # ── 4. Upsample masks to full image resolution ───────────────────
        pred_masks_full = F.interpolate(
            pred_masks, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False
        )   # [B, Q, 224, 224]

        if targets is not None:
            # Training: compute loss
            criterion = Mask2FormerLoss(
                num_classes=self.transformer_decoder.class_head.out_features - 1)
            loss, loss_dict = criterion(
                pred_logits, pred_masks, aux_outputs, targets)
            return loss, loss_dict

        return pred_logits, pred_masks_full

    @torch.no_grad()
    def predict_semantic(self, pixel_values):
        """
        Inference helper: collapses queries → per-pixel semantic labels.
        Standard Mask2Former inference procedure.
        """
        pred_logits, pred_masks = self.forward(pixel_values)
        # [B, Q, C+1] × [B, Q, H, W] → [B, C, H, W]
        mask_probs  = pred_masks.sigmoid()                         # [B, Q, H, W]
        class_probs = pred_logits.softmax(-1)[..., :-1]            # [B, Q, C] drop no-obj
        # Weighted sum: each query contributes its class prob × mask prob
        seg_logits  = torch.einsum('bqc,bqhw->bchw', class_probs, mask_probs)
        return seg_logits.argmax(dim=1)  # [B, H, W]  — per-pixel class index
