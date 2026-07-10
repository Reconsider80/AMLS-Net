"""AMLS-Net: Adaptive Multi-Level Skip Learning Network.

Paper: Exploring Adaptive Multi-Level Skip Learning in Networks for Medical Image Segmentation
Code: https://github.com/Reconsider80/AMLS-Net

Architecture:
  Stem -> E1/E2 (SA) -> E3/E4 (VSSK) -> LSC -> FRD decoder (D4/D3 VSSK + FRD) -> head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frd import FRDBlock, UpBlock
from .lsc import LearnableSkipConnection
from .sa_block import Downsample, SABlock
from .vssk import VSSKBlock


class Stem(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,H/2,W/2,C)
        x = self.net(x)
        return x.permute(0, 2, 3, 1).contiguous()


class AMLSNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        dims: tuple[int, ...] = (64, 128, 256, 512),
        sa_depths: tuple[int, ...] = (2, 2),
        vssk_depths: tuple[int, ...] = (2, 2),
        num_heads: tuple[int, ...] = (4, 4, 8, 8),
        d_state: int = 16,
        fuse_dim: int = 128,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        no_kan: bool = False,
    ):
        super().__init__()
        assert len(dims) == 4
        self.dims = dims
        self.num_classes = num_classes

        self.stem = Stem(in_chans, dims[0])

        # Encoder E1/E2: SA blocks
        self.e1 = SABlock(dims[0], num_heads=num_heads[0], drop=drop, attn_drop=attn_drop, depth=sa_depths[0])
        self.down1 = Downsample(dims[0], dims[1])
        self.e2 = SABlock(dims[1], num_heads=num_heads[1], drop=drop, attn_drop=attn_drop, depth=sa_depths[1])
        self.down2 = Downsample(dims[1], dims[2])

        # Encoder E3/E4: VSSK blocks
        self.e3 = VSSKBlock(
            dims[2], depth=vssk_depths[0], d_state=d_state, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, no_kan=no_kan,
        )
        self.down3 = Downsample(dims[2], dims[3])
        self.e4 = VSSKBlock(
            dims[3], depth=vssk_depths[1], d_state=d_state, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, no_kan=no_kan,
        )

        # Learnable skip connections
        self.lsc = LearnableSkipConnection(list(dims), fuse_dim=fuse_dim, drop=drop)

        # Decoder: bottleneck VSSK + FRD up path
        self.d4_vssk = VSSKBlock(
            dims[3], depth=vssk_depths[1], d_state=d_state, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, no_kan=no_kan,
        )
        self.up3 = UpBlock(dims[3], dims[2])
        self.frd3 = FRDBlock(dims[2], dims[2], dims[2], d_state=d_state)
        self.d3_vssk = VSSKBlock(
            dims[2], depth=vssk_depths[0], d_state=d_state, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, no_kan=no_kan,
        )

        self.up2 = UpBlock(dims[2], dims[1])
        self.frd2 = FRDBlock(dims[1], dims[1], dims[1], d_state=d_state)

        self.up1 = UpBlock(dims[1], dims[0])
        self.frd1 = FRDBlock(dims[0], dims[0], dims[0], d_state=d_state)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.seg_head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], num_classes, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits (B, num_classes, H, W)
        """
        input_size = x.shape[-2:]

        # Encoder
        f1 = self.e1(self.stem(x))                 # (B,H/2,W/2,C1)
        f2 = self.e2(self.down1(f1))               # (B,H/4,W/4,C2)
        f3 = self.e3(self.down2(f2))               # (B,H/8,W/8,C3)
        f4 = self.e4(self.down3(f3))               # (B,H/16,W/16,C4)

        skips = self.lsc([self._to_bchw(f1), self._to_bchw(f2), self._to_bchw(f3), self._to_bchw(f4)])
        l1, l2, l3, l4 = skips

        # Decoder
        d4 = self._to_bchw(self.d4_vssk(self._to_bhwc(l4)))
        d3_up = self.up3(d4)
        d3 = self.frd3(l3, d3_up)
        d3 = self._to_bchw(self.d3_vssk(self._to_bhwc(d3)))

        d2_up = self.up2(d3)
        d2 = self.frd2(l2, d2_up)

        d1_up = self.up1(d2)
        d1 = self.frd1(l1, d1_up)

        out = self.seg_head(self.final_up(d1))
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out


def build_amls_net(num_classes: int = 1, in_chans: int = 3, **kwargs) -> AMLSNet:
    return AMLSNet(in_chans=in_chans, num_classes=num_classes, **kwargs)
