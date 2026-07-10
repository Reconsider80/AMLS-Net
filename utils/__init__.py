"""Losses and metrics for AMLS-Net (Dice + BCE as in the paper)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)
            num_classes = logits.shape[1]
            targets_oh = F.one_hot(targets.clamp_min(0), num_classes).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
            inter = (probs * targets_oh).sum(dims)
            den = probs.sum(dims) + targets_oh.sum(dims)
            dice = (2 * inter + self.smooth) / (den + self.smooth)
            return 1 - dice.mean()

        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        den = probs.sum() + targets.sum()
        return 1 - (2 * inter + self.smooth) / (den + self.smooth)


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss used in the paper."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            # multi-class
            ce = self.ce(logits, targets.long())
            dice = self.dice(logits, targets.long())
            return self.bce_weight * ce + self.dice_weight * dice
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def dice_coef(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> float:
    if logits.shape[1] > 1:
        preds = logits.argmax(dim=1)
        num_classes = logits.shape[1]
        dices = []
        for c in range(1, num_classes):  # skip background
            p = (preds == c).float()
            t = (targets == c).float()
            inter = (p * t).sum()
            dices.append(((2 * inter + smooth) / (p.sum() + t.sum() + smooth)).item())
        return float(sum(dices) / max(len(dices), 1))

    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum()
    return ((2 * inter + smooth) / (preds.sum() + targets.sum() + smooth)).item()


def hausdorff_distance(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Approximate HD95 using distance transforms (binary / per-class mean)."""
    import cv2
    import numpy as np

    def _hd95(pred: np.ndarray, gt: np.ndarray) -> float:
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        if not pred.any() and not gt.any():
            return 0.0
        if not pred.any() or not gt.any():
            return float(max(pred.shape))
        pred_dt = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 5)
        gt_dt = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
        pred_border = pred & (cv2.dilate((~pred).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        gt_border = gt & (cv2.dilate((~gt).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if not pred_border.any() or not gt_border.any():
            d1 = pred_dt[gt].max() if gt.any() else 0.0
            d2 = gt_dt[pred].max() if pred.any() else 0.0
        else:
            d1 = pred_dt[gt_border].max()
            d2 = gt_dt[pred_border].max()
        return float(max(d1, d2))

    if logits.shape[1] > 1:
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
        gt = targets[0].detach().cpu().numpy()
        vals = []
        for c in range(1, logits.shape[1]):
            vals.append(_hd95(pred == c, gt == c))
        return float(sum(vals) / max(len(vals), 1))

    pred = (torch.sigmoid(logits) > 0.5)[0, 0].detach().cpu().numpy()
    gt = targets[0, 0].detach().cpu().numpy() > 0.5
    return _hd95(pred, gt)
