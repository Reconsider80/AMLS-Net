"""Train AMLS-Net.

Paper setup (Sec. 5):
  - Optimizer: Adam
  - LR: 2e-4 with cosine annealing to 1e-5
  - Epochs: 300
  - Loss: Dice + BCE
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import build_dataloaders
from models import AMLSNet
from utils import DiceBCELoss, dice_coef
from utils.misc import count_parameters, ensure_dir, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train AMLS-Net")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        total_dice += dice_coef(logits, masks)
        n += 1
    return total_loss / max(n, 1), total_dice / max(n, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    work_dir = cfg.get("work_dir", "runs/amls_net")
    ckpt_dir = os.path.join(work_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    train_loader, val_loader = build_dataloaders(cfg)

    model = AMLSNet(
        in_chans=cfg.get("in_chans", 3),
        num_classes=cfg.get("num_classes", 1),
        dims=tuple(cfg.get("dims", [64, 128, 256, 512])),
        sa_depths=tuple(cfg.get("sa_depths", [2, 2])),
        vssk_depths=tuple(cfg.get("vssk_depths", [2, 2])),
        fuse_dim=cfg.get("fuse_dim", 128),
        drop=cfg.get("drop", 0.0),
        no_kan=cfg.get("no_kan", False),
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Parameters: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")
    print(f"mamba_ssm available: {__import__('models.ss2d', fromlist=['HAS_MAMBA']).HAS_MAMBA}")

    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("epochs", 300),
        eta_min=cfg.get("min_lr", 1e-5),
    )

    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    epochs = cfg.get("epochs", 300)
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        train_loss = running / max(len(train_loader), 1)
        val_loss, val_dice = validate(model, val_loader, criterion, device, cfg.get("num_classes", 1))
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} time={time.time()-t0:.1f}s"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "cfg": cfg,
        }
        torch.save(state, os.path.join(ckpt_dir, "latest.pth"))
        if val_dice >= best_dice:
            best_dice = val_dice
            state["best_dice"] = best_dice
            torch.save(state, os.path.join(ckpt_dir, "best.pth"))
            print(f"  * new best dice={best_dice:.4f}")

    print(f"Training done. Best Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
