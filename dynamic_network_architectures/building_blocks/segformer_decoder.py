from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvBNReLU
from dynamic_network_architectures.building_blocks.unetr_decoder import MergeBlock


class SegFormerDecoder(nn.Module):
    """
    SegFormer All-MLP Decoder Head (paper: "SegFormer: Simple and Efficient
    Design for Semantic Segmentation with Transformers", Xie et al. 2021).

    Args:
        input_channels  : Channels of the raw image input (typically 3 for RGB).
        hidden_dim:   unified decoder channel width C (default 256)
        image_size      : int or (H, W). Used as the default upsample target.
        num_classes: number of segmentation classes K
        dropout:     dropout rate before the final linear
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        image_size: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        # 1. Per-stage linear projections: Ci → embed_dim
        self.linear_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 2. Fusion MLP: 4*embed_dim → embed_dim, with BN + GELU
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim),
            nn.BatchNorm2d(hidden_dim),   # applied after reshape; see forward()
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # BN needs to be applied in (N,C,H,W) space, so we build it separately
        self.fusion_linear = nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.fusion_norm   = nn.BatchNorm2d(hidden_dim)
        self.fusion_act    = nn.GELU()
        self.fusion_drop   = nn.Dropout(dropout)

        # 3. Segmentation head: embed_dim → num_classes
        # ── Image-level skip (stride-1) → final merge at full resolution ─
        emb_dim = 64
        self.inputconv = nn.Sequential(
            ConvBNReLU(input_channels, emb_dim, padding=1),
            ConvBNReLU(emb_dim, emb_dim, padding=1),
        )
        self.mergehead  = MergeBlock(emb_dim, hidden_dim, hidden_dim)
        self.seg_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

        self.embed_dim   = hidden_dim
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_and_upsample(
        feat: torch.Tensor,        # (N, C, H, W)
        linear: nn.Linear,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        N, C, H, W = feat.shape
        # flatten spatial dims → (N, H*W, C), project, restore
        x = feat.permute(0, 2, 3, 1).reshape(N, H * W, C)   # (N, HW, C)
        x = linear(x)                                         # (N, HW, embed_dim)
        x = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)       # (N, embed_dim, H, W)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pyramid: list[torch.Tensor],
        inputs: torch.Tensor,
        image_hw:  Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            pyramid: [f1, f2, f3, f4] feature maps from encoder stages.
                      f1 is the highest resolution (H/4 × W/4).
        Returns:
            logits of shape (N, num_classes, H/4, W/4).
            Call F.interpolate(..., scale_factor=4) outside if you need H×W.
        """
        if image_hw is None:
            image_hw = self.image_size

        image_skip = self.inputconv(inputs)   # (B, F, H, W)
        assert len(pyramid) == len(self.linear_projections)

        # Target spatial size = resolution of stage-1 features (finest)
        target_h, target_w = pyramid[0].shape[2], pyramid[0].shape[3]

        # 1. Project every stage to embed_dim and upsample to (H/4, W/4)
        projected = [
            self._project_and_upsample(feat, proj, (target_h, target_w))
            for feat, proj in zip(pyramid, self.linear_projections)
        ]

        # 2. Concatenate along channel dim → (N, 4*embed_dim, H/4, W/4)
        x = torch.cat(projected, dim=1)

        # 3. Fusion MLP in (N,C,H,W) space
        N, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H * W, -1)  # (N, HW, 4C)
        x = self.fusion_linear(x)                          # (N, HW, C)
        x = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)    # (N, C, H, W)
        x = self.fusion_norm(x)
        x = self.fusion_act(x)
        x = self.fusion_drop(x)

        # 4. Segmentation head 
        x = F.interpolate(x, size=image_hw, mode="bilinear", align_corners=False)
        x = self.mergehead(image_skip, x)
        x = self.seg_head(x)

        return x  # (N, K, H/4, W/4)