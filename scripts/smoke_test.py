"""Forward-pass smoke test for AMLS-Net."""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import AMLSNet
from models.ss2d import HAS_MAMBA
from utils.misc import count_parameters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, mamba_ssm={HAS_MAMBA}")

    # Lightweight config for smoke test
    model = AMLSNet(
        in_chans=3,
        num_classes=1,
        dims=(32, 64, 128, 256),
        sa_depths=(1, 1),
        vssk_depths=(1, 1),
        fuse_dim=64,
        no_kan=True,  # faster CPU smoke
    ).to(device)
    model.eval()

    total, trainable = count_parameters(model)
    print(f"params: {total/1e6:.2f}M (trainable {trainable/1e6:.2f}M)")

    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"input={tuple(x.shape)} -> output={tuple(y.shape)}")
    assert y.shape == (1, 1, 128, 128), y.shape
    print("smoke test OK")


if __name__ == "__main__":
    main()
