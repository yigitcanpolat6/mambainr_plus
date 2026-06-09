# MambaINRPlus

Multivariate Time-Series Imputation with Mamba and Implicit Neural Representations

## Architecture

**MambaINRPlus** combines:
- Mask-aware multi-scale tokenizer
- Optional Mamba/SSM temporal encoder (with PyTorch fallback)
- Efficient channel-correlation module
- Adaptive group-based INR decoder
- Trend/seasonal/residual decomposition
- Robust imputation losses

## Setup

```bash
pip install -r requirements.txt
```

Optional (for Mamba support):
```bash
pip install mamba-ssm
```

## Training

```bash
python train.py --config configs/weather.yaml
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --config configs/weather.yaml
```

## Hardware

Tested on:
- NVIDIA RTX 4070 Laptop
- Intel i7-13620H
- 32 GB RAM
- Windows 11

## Model Output

- Input: `x [B, L, C]`, `mask [B, L, C]`, `time [B, L]`
- Output: `pred [B, L, C]`

The model predicts every variable at every timestamp.
