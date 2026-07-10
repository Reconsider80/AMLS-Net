"""Evaluate AMLS-Net and optionally save predictions."""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import MedicalSegDataset
from models import AMLSNet
from utils import dice_coef, hausdorff_distance
from utils.misc import ensure_dir, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Test AMLS-Net")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = MedicalSegDataset(
        image_dir=cfg.get("test_image_dir", cfg["val_image_dir"]),
        mask_dir=cfg.get("test_mask_dir", cfg["val_mask_dir"]),
        img_size=cfg.get("img_size", 256),
        num_classes=cfg.get("num_classes", 1),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = AMLSNet(
        in_chans=cfg.get("in_chans", 3),
        num_classes=cfg.get("num_classes", 1),
        dims=tuple(cfg.get("dims", [64, 128, 256, 512])),
        sa_depths=tuple(cfg.get("sa_depths", [2, 2])),
        vssk_depths=tuple(cfg.get("vssk_depths", [2, 2])),
        fuse_dim=cfg.get("fuse_dim", 128),
        no_kan=cfg.get("no_kan", False),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    if args.save_dir:
        ensure_dir(args.save_dir)

    dices, hds = [], []
    for batch in tqdm(loader, desc="Testing"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        dices.append(dice_coef(logits, masks))
        hds.append(hausdorff_distance(logits, masks))

        if args.save_dir:
            if logits.shape[1] > 1:
                pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            else:
                pred = (torch.sigmoid(logits) > 0.5)[0, 0].cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(args.save_dir, f"{batch['name'][0]}.png"), pred)

    print(f"mDice: {np.mean(dices)*100:.2f} ± {np.std(dices)*100:.2f}")
    print(f"HD:    {np.mean(hds):.2f} ± {np.std(hds):.2f}")


if __name__ == "__main__":
    main()
