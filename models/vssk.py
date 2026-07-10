"""VSSK: Vision State Space + KAN hybrid block (paper Sec. 3.3)."""
from __future__ import annotations
import torch
import torch.nn as nn
from .kan import KANBlock
from .ss2d import VSSBlock

class SpatialConvMix(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).contiguous()

class VSSKBlock(nn.Module):
    def __init__(self, dim: int, depth: int = 1, d_state: int = 16, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0, no_kan: bool = False):
        super().__init__()
        self.vss = VSSBlock(dim=dim, depth=depth, drop_path=drop_path, attn_drop=attn_drop, d_state=d_state)
        self.norm = nn.LayerNorm(dim)
        self.spatial = SpatialConvMix(dim)
        self.kan = KANBlock(dim=dim, drop=drop, drop_path=drop_path, no_kan=no_kan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vss(x)
        b, h, w, c = x.shape
        x_bar = self.spatial(self.norm(x))
        tokens = self.kan(x_bar.reshape(b, h * w, c), h, w)
        return tokens.view(b, h, w, c) + x_bar
