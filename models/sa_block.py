"""SAM2-Adaptive (SA) encoder block (paper Sec. 3.2).

Uses windowed self-attention + bottleneck adapters for memory-efficient
low-level feature refinement (SAM2/Hiera-style local attention).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """Bottleneck adapter (paper Eq. 1): S(x)=GELU(x Fd) Fu."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.down = nn.Linear(dim, mid)
        self.up = nn.Linear(mid, dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


def window_partition(x: torch.Tensor, window_size: int):
    # x: (B, H, W, C)
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowSAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.adapter = Adapter(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        hp, wp = h + pad_h, w + pad_w
        windows = window_partition(x, ws)  # (nW*B, ws*ws, C)
        n, l, _ = windows.shape
        qkv = self.qkv(windows).reshape(n, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(n, l, c)
        out = self.proj_drop(self.proj(out))
        out = out + self.adapter(out)
        out = window_reverse(out, ws, hp, wp)
        if pad_h or pad_w:
            out = out[:, :h, :w, :].contiguous()
        return out


class SABlock(nn.Module):
    """SAM2-adaptive transformer block for low-level encoder stages E1/E2."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0, depth: int = 2, window_size: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": WindowSAAttention(dim, num_heads=num_heads, window_size=window_size,
                                          attn_drop=attn_drop, proj_drop=drop),
                "norm2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                    nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop),
                ),
                "adapter_mlp": Adapter(dim),
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        for blk in self.blocks:
            x = x + blk["attn"](blk["norm1"](x))
            b, h, w, c = x.shape
            tokens = x.reshape(b, h * w, c)
            mlp_out = blk["mlp"](blk["norm2"](tokens))
            tokens = tokens + mlp_out + blk["adapter_mlp"](mlp_out)
            x = tokens.view(b, h, w, c)
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 3, 1, 2).contiguous())
        return self.norm(x.permute(0, 2, 3, 1).contiguous())
