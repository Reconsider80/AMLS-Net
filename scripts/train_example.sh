#!/usr/bin/env bash
# Quick training examples for AMLS-Net

# Single-organ (256x256) — edit paths in configs/default.yaml first
python train.py --config configs/default.yaml

# Multi-lesion fundus (960x960)
python train.py --config configs/fundus.yaml

# Evaluation
python test.py --config configs/default.yaml \
  --checkpoint runs/amls_net/checkpoints/best.pth \
  --save_dir preds/
