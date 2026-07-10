"""Feature Recalibration Decoder (FRD) block (paper Sec. 3.5).

Two branches:
  skip path:  Linear -> Conv -> SiLU -> SSM
  decoder path: Linear -> SiLU
Fused by Hadamard product: E_out = L_n ⊗ D_n
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ss2d import SS2D


class FRDBlock(nn.Module):
    """Feature Recalibration Decoder unit."""

    def __init__(self, skip_dim: int, dec_dim: int, out_dim: int, d_state: int = 16):
        super().__init__()
        self.skip_linear = nn.Conv2d(skip_dim, out_dim, 1)
        self.skip_conv = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.ssm = SS2D(d_model=out_dim, d_state=d_state, expand=1)

        self.dec_linear = nn.Conv2d(dec_dim, out_dim, 1)
        self.out_norm = nn.BatchNorm2d(out_dim)
        self.out_act = nn.GELU()

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip: L_n  (B, C_s, H, W)
            dec:  D_n  (B, C_d, H, W)  — already spatially aligned
        """
        if skip.shape[-2:] != dec.shape[-2:]:
            skip = F.interpolate(skip, size=dec.shape[-2:], mode="bilinear", align_corners=False)

        # Branch 1: skip -> SSM (paper Eqs. 9-10)
        ln = self.skip_conv(self.skip_linear(skip))
        ln = F.silu(ln)
        ln = ln.permute(0, 2, 3, 1).contiguous()
        ln = self.ssm(ln)
        ln = ln.permute(0, 3, 1, 2).contiguous()

        # Branch 2: decoder features (paper Eq. 11)
        dn = F.silu(self.dec_linear(dec))

        # Hadamard fusion (paper Eq. 12)
        out = ln * dn
        return self.out_act(self.out_norm(out))


class UpBlock(nn.Module):
    """2x upsample + optional channel projection."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.up(x))
