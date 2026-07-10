"""Learnable Skip Connections (LSC): CAB + SAB (paper Sec. 3.4).

Memory-efficient realization of the paper's multi-scale channel/spatial attention:
- CAB attends across scales on channel tokens
- SAB uses pooled multi-scale keys/values for spatial recalibration
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionBlock(nn.Module):
    """CAB: cross-scale attention along the channel/scale axis (paper Eq. 6)."""

    def __init__(self, dim: int, num_scales: int = 4, num_heads: int = 4):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim**-0.5

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        target_h = max(f.shape[2] for f in feats)
        target_w = max(f.shape[3] for f in feats)
        aligned = [
            F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            for f in feats
        ]
        b = aligned[0].shape[0]
        # (B, HW, S, C)
        x = torch.stack(aligned, dim=1).permute(0, 3, 4, 1, 2)
        x = x.reshape(b * target_h * target_w, self.num_scales, self.dim)

        qkv = self.qkv(x).reshape(-1, self.num_scales, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(-1, self.num_scales, self.dim)
        out = self.proj(out).reshape(b, target_h, target_w, self.num_scales, self.dim)
        out = out.permute(0, 3, 4, 1, 2)
        return [out[:, i] for i in range(self.num_scales)]


class SpatialAttentionBlock(nn.Module):
    """SAB: spatial attention with pooled multi-scale context (paper Eq. 7)."""

    def __init__(self, dim: int, num_scales: int = 4, pool_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pool_size = pool_size

        self.q_projs = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(num_scales)])
        self.k_proj = nn.Conv2d(dim * num_scales, dim, 1)
        self.v_proj = nn.Conv2d(dim * num_scales, dim, 1)
        self.out_projs = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(num_scales)])
        self.gamma = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_scales)])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        target_h = max(f.shape[2] for f in feats)
        target_w = max(f.shape[3] for f in feats)
        aligned = [
            F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            for f in feats
        ]
        b = aligned[0].shape[0]
        ctx = torch.cat(aligned, dim=1)
        ctx_p = F.adaptive_avg_pool2d(ctx, output_size=(self.pool_size, self.pool_size))
        k = self.k_proj(ctx_p).reshape(b, self.num_heads, self.head_dim, -1)
        v = self.v_proj(ctx_p).reshape(b, self.num_heads, self.head_dim, -1)

        outs = []
        for i, feat in enumerate(aligned):
            q = self.q_projs[i](feat).reshape(b, self.num_heads, self.head_dim, -1)
            attn = (q.transpose(-2, -1) @ k) * self.scale  # (B, heads, HW, P)
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # (B, heads, C/h, HW)
            out = out.reshape(b, self.dim, target_h, target_w)
            out = self.out_projs[i](out)
            outs.append(feat + self.gamma[i] * out)
        return outs


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        t = t + self.mlp(self.norm(t))
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class LearnableSkipConnection(nn.Module):
    """LSC producing refined multi-scale skip features L1..L4."""

    def __init__(
        self,
        in_dims: list[int],
        fuse_dim: int = 128,
        num_heads: int = 4,
        drop: float = 0.0,
        pool_size: int = 16,
    ):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.projs = nn.ModuleList([nn.Conv2d(d, fuse_dim, 1) for d in in_dims])
        self.cab = ChannelAttentionBlock(fuse_dim, num_scales=len(in_dims), num_heads=num_heads)
        self.sab = SpatialAttentionBlock(
            fuse_dim, num_scales=len(in_dims), pool_size=pool_size, num_heads=num_heads
        )
        self.mlps = nn.ModuleList([ResidualMLP(fuse_dim, drop=drop) for _ in in_dims])
        self.out_projs = nn.ModuleList([nn.Conv2d(fuse_dim, d, 1) for d in in_dims])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        sizes = [(f.shape[2], f.shape[3]) for f in feats]
        projected = [proj(f) for proj, f in zip(self.projs, feats)]
        cab_out = self.cab(projected)
        sab_out = self.sab(cab_out)
        refined = [mlp(f) for mlp, f in zip(self.mlps, sab_out)]
        outs = []
        for i, (r, out_proj) in enumerate(zip(refined, self.out_projs)):
            r = F.interpolate(r, size=sizes[i], mode="bilinear", align_corners=False)
            outs.append(out_proj(r))
        return outs
