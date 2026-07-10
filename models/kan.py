"""Kolmogorov-Arnold Network (KAN) layers.

Adapted from U-KAN / MedVKAN:
  https://github.com/CUHK-AIM-Group/U-KAN
  https://github.com/beginner-cjh/MedVKAN
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class KANLinear(nn.Module):
    """B-spline parameterized KAN linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)]).clamp_min(1e-6)
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)]).clamp_min(1e-6)
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class DWConvBNAct(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.act(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class KANLayer(nn.Module):
    """Two-step KAN with depthwise conv spatial mixing (paper Eqs. 2-5)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        no_kan: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or max(in_features // 8, 8)
        self.hidden_features = hidden_features
        self.out_features = out_features

        if no_kan:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = KANLinear(in_features, hidden_features, grid_size=3, spline_order=3)
            self.fc2 = KANLinear(hidden_features, out_features, grid_size=3, spline_order=3)

        self.dw1 = DWConvBNAct(hidden_features)
        self.dw2 = DWConvBNAct(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x1 = self.fc1(x.reshape(b * n, c)).reshape(b, n, self.hidden_features)
        x1 = self.dw1(x1, h, w)
        x1 = self.fc2(x1.reshape(b * n, self.hidden_features)).reshape(b, n, self.out_features)
        x1 = self.dw2(x1, h, w)
        return self.drop(x + x1)


class KANBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0, drop_path: float = 0.0, no_kan: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = KANLayer(dim, hidden_features=max(dim // 8, 8), out_features=dim, drop=drop, no_kan=no_kan)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return x + self.drop_path(self.layer(self.norm(x), h, w))
