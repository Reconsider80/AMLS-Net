"""Vision State Space (VSS) with 2D Selective Scan (SS2D).

Core logic adapted from VMamba / MedVKAN:
  https://github.com/MzeroMiko/VMamba
  https://github.com/beginner-cjh/MedVKAN

Uses mamba_ssm.selective_scan_fn when available; otherwise a pure-PyTorch fallback.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import DropPath


def _repeat(tensor: torch.Tensor, pattern: str, **axes):
    """Minimal einops.repeat replacement used by SS2D init helpers."""
    if pattern == "n -> d n":
        return tensor.unsqueeze(0).expand(axes["d"], -1).contiguous()
    if pattern == "d n -> r d n":
        return tensor.unsqueeze(0).expand(axes["r"], -1, -1).contiguous()
    if pattern == "n1 -> r n1":
        return tensor.unsqueeze(0).expand(axes["r"], -1).contiguous()
    raise ValueError(f"Unsupported repeat pattern: {pattern}")

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_scan

    HAS_MAMBA = True
except Exception:  # pragma: no cover
    _mamba_scan = None
    HAS_MAMBA = False


def selective_scan_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch selective scan (slower, CUDA-free fallback).

    Args:
        u: (B, D, L)
        delta: (B, D, L)
        A: (D, N)
        B: (B, K, N, L) or (B, N, L) — we expect flattened K*D layout matching mamba API
        C: same as B
        D: (D,)
    """
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    # Handle 4-direction packed layout: B has shape (B, K, N, L), A/D are (K*D, N)/(K*D,)
    # Mirror the mamba_ssm call convention used in SS2D.forward_core.
    bsz, d_in, length = u.shape
    n = A.shape[1]

    if B.dim() == 4:
        # (B, K, N, L) -> (B, K*D?); MedVKAN packs xs as (B, K*D, L) and B as (B, K, N, L)
        k = B.shape[1]
        d_model = d_in // k
        ys = []
        for ki in range(k):
            uk = u[:, ki * d_model : (ki + 1) * d_model, :]
            dk = delta[:, ki * d_model : (ki + 1) * d_model, :]
            Ak = A[ki * d_model : (ki + 1) * d_model]
            Dk = None if D is None else D[ki * d_model : (ki + 1) * d_model]
            Bk = B[:, ki]  # (B, N, L)
            Ck = C[:, ki]
            ys.append(_scan_one(uk, dk, Ak, Bk, Ck, Dk))
        return torch.cat(ys, dim=1)

    return _scan_one(u, delta, A, B, C, D)


def _scan_one(u, delta, A, B, C, D):
    bsz, d_in, length = u.shape
    n = A.shape[1]
    if B.dim() == 3 and B.shape[1] == n:
        # (B, N, L)
        pass
    elif B.dim() == 3 and B.shape[1] == d_in:
        # (B, D, N) style not used
        raise ValueError(f"Unexpected B shape {B.shape}")

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    x = u.new_zeros((bsz, d_in, n))
    ys = []
    for i in range(length):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (B, D, L)
    if D is not None:
        y = y + u * D[None, :, None]
    return y


def selective_scan_fn(*args, **kwargs):
    if HAS_MAMBA:
        return _mamba_scan(*args, **kwargs)
    # Drop unsupported kwargs for fallback
    kwargs.pop("z", None)
    kwargs.pop("return_last_state", None)
    return selective_scan_pytorch(*args, **kwargs)


class SS2D(nn.Module):
    """2D Selective Scan module (paper Figure 3)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(
            torch.stack(
                [
                    nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False).weight
                    for _ in range(4)
                ],
                dim=0,
            )
        )
        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        self._init_dt(dt_scale, dt_init, dt_min, dt_max, dt_init_floor)

        self.A_logs = self._A_log_init(d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def _init_dt(self, dt_scale, dt_init, dt_min, dt_max, dt_init_floor):
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        for i in range(4):
            if dt_init == "constant":
                nn.init.constant_(self.dt_projs_weight[i], dt_init_std)
            else:
                nn.init.uniform_(self.dt_projs_weight[i], -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_projs_bias[i].copy_(inv_dt)

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, merge=True):
        A = _repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner)
        A_log = torch.log(A)
        if copies > 1:
            A_log = _repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        p = nn.Parameter(A_log)
        p._no_weight_decay = True
        return p

    @staticmethod
    def _D_init(d_inner, copies=1, merge=True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = _repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        p = nn.Parameter(D)
        p._no_weight_decay = True
        return p

    def forward_core(self, x: torch.Tensor):
        b, c, h, w = x.shape
        l = h * w
        k = 4
        x_hwwh = torch.stack(
            [x.view(b, -1, l), torch.transpose(x, 2, 3).contiguous().view(b, -1, l)],
            dim=1,
        ).view(b, 2, -1, l)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs.view(b, k, -1, l), self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts.view(b, k, -1, l), self.dt_projs_weight)

        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        bs = bs.float().view(b, k, -1, l)
        cs = cs.float().view(b, k, -1, l)
        ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            bs,
            cs,
            ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, l)
        wh_y = torch.transpose(out_y[:, 1].view(b, -1, w, h), 2, 3).contiguous().view(b, -1, l)
        invwh_y = torch.transpose(inv_y[:, 1].view(b, -1, w, h), 2, 3).contiguous().view(b, -1, l)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y1, y2, y3, y4 = self.forward_core(x)
        y = (y1 + y2 + y3 + y4).transpose(1, 2).contiguous().view(b, h, w, -1)
        y = self.out_norm(y) * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSLayer(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, attn_drop: float = 0.0, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, dropout=attn_drop, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.ss2d(self.norm(x)))


class VSSBlock(nn.Module):
    """Dual-branch VSS block used inside VSSK (paper Sec. 3.3)."""

    def __init__(self, dim: int, depth: int = 1, drop_path: float = 0.0, attn_drop: float = 0.0, d_state: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                VSSLayer(
                    dim=dim,
                    drop_path=drop_path if not isinstance(drop_path, list) else drop_path[i],
                    attn_drop=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
