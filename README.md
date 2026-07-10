# Exploring Adaptive Multi-Level Skip Learning in Networks for Medical Image Segmentation

![image](https://github.com/Reconsider80/AMLS-Net/blob/main/AMLS-Net.png)

Official PyTorch implementation of **AMLS-Net** (Adaptive Multi-Level Skip Learning Network) for medical image segmentation.

## Overview

AMLS-Net addresses static skip connections and fragmented feature fusion in U-shaped networks with:

- **SA block**: SAM2-based adaptive encoder with lightweight adapters
- **VSSK block**: hybrid Vision State Space (Mamba) + KAN for high-level semantics
- **LSC**: Learnable Skip Connections (CAB + SAB) for adaptive multi-scale fusion
- **FRD**: Feature Recalibration Decoder with SSM-guided skip/decoder alignment

![Table1](https://github.com/Reconsider80/AMLS-Net/blob/main/Table1.png)

## Requirements

```bash
pip install -r requirements.txt
# Optional (faster SS2D on CUDA):
pip install causal-conv1d mamba-ssm
```

## Dataset layout

```
data/<DATASET>/
  train/images/
  train/masks/
  val/images/
  val/masks/
  test/images/
  test/masks/
```

Edit paths in `configs/default.yaml` or `configs/fundus.yaml`.

## Train / Test

```bash
python scripts/smoke_test.py
python train.py --config configs/default.yaml
python test.py --config configs/default.yaml \
  --checkpoint runs/amls_net/checkpoints/best.pth \
  --save_dir preds/
```

### Hyperparameters (paper)

| Setting | Single-organ | Multi-organ | Multi-lesion |
|---------|--------------|-------------|--------------|
| Size | 256×256 | 320/512 | 960×960 |
| Batch | 18 | 18 | 3 |
| Optimizer | Adam | Adam | Adam |
| LR | 2e-4 → 1e-5 (cosine) | same | same |
| Epochs | 300 | 300 | 300 |
| Loss | Dice + BCE | Dice + BCE | Dice + BCE |

## Project structure

```
AMLS-Net/
├── AMLS-Net.py          # model (single-file)
├── AMLS-Net.png         # architecture figure
├── Table1.png
├── models/              # modular implementation
│   ├── amls_net.py
│   ├── sa_block.py
│   ├── vssk.py
│   ├── ss2d.py
│   ├── kan.py
│   ├── lsc.py
│   └── frd.py
├── configs/
├── datasets/
├── utils/
├── train.py
└── test.py
```



