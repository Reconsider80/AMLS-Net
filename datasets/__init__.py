"""Dataset utilities for AMLS-Net medical image segmentation."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(folder: Path) -> list[Path]:
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    return files


class MedicalSegDataset(Dataset):
    """Generic image/mask folder dataset.

    Expected layout:
        root/
          images/  xxx.png
          masks/   xxx.png   (same stem)
    or:
        root/
          train/images ...
          train/masks ...
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 256,
        num_classes: int = 1,
        augment: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment

        self.images = _list_images(self.image_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.masks = []
        for img in self.images:
            cand = None
            for ext in IMG_EXTS:
                p = self.mask_dir / f"{img.stem}{ext}"
                if p.exists():
                    cand = p
                    break
            if cand is None:
                raise FileNotFoundError(f"Mask for {img.name} not found in {self.mask_dir}")
            self.masks.append(cand)

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        if np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return image, mask

    def __getitem__(self, idx: int):
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise RuntimeError(f"Failed to read mask {self.masks[idx]}")

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.num_classes == 1:
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask[None, ...]).float()
        else:
            mask = mask.astype(np.int64)
            mask = torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, "name": self.images[idx].stem}


def build_dataloaders(cfg: dict):
    from torch.utils.data import DataLoader

    train_set = MedicalSegDataset(
        image_dir=cfg["train_image_dir"],
        mask_dir=cfg["train_mask_dir"],
        img_size=cfg.get("img_size", 256),
        num_classes=cfg.get("num_classes", 1),
        augment=True,
    )
    val_set = MedicalSegDataset(
        image_dir=cfg["val_image_dir"],
        mask_dir=cfg["val_mask_dir"],
        img_size=cfg.get("img_size", 256),
        num_classes=cfg.get("num_classes", 1),
        augment=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )
    return train_loader, val_loader
