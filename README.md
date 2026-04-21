# NYCU Visual Recognition HW2

## Introduction

The main model is a DINO-style DETR detector with a ResNet-50 backbone. It keeps the DETR set-prediction formulation, including object queries, a Transformer encoder-decoder, Hungarian matching, and bounding-box/class prediction heads. To improve convergence and small-object localization, the DETR components are modified with multi-scale features, deformable attention, two-stage proposals, denoising training, sigmoid focal loss, and auxiliary decoder losses.

## Environment Setup

Recommended environment:

```bash
python -m pip install torch torchvision pycocotools pillow scipy matplotlib tqdm numpy
```

The code was tested with Python 3.13, PyTorch 2.9.1, CUDA-enabled GPU, and pycocotools. No external dataset is used.

Dataset layout:

```text
AI-HW2/
├── train.py
├── predict.py
├── model.py
├── dataset.py
├── utils.py
└── nycu-hw2-data/
    ├── train/
    ├── valid/
    ├── test/
    ├── train.json
    └── valid.json
```

## Usage

Train the main DINO-style DETR model:

```bash
python train.py
```

Resume training:

```bash
python train.py --resume ./outputs/<timestamp>/checkpoint.pth --epochs 30
```

Generate `pred.json` and `pred.zip`:

```bash
python predict.py --zip
```

Generate predictions from a specific checkpoint:

```bash
python predict.py --checkpoint ./outputs/<timestamp>/best.pth --zip
```

## Performance Snapshot
<img width="2558" height="1324" alt="Screenshot from 2026-04-21 18-18-05" src="https://github.com/user-attachments/assets/03501262-8dd1-48d9-8d65-9b578173e193" />
