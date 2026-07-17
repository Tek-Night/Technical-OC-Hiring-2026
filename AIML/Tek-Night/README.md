# Image upscaling using deep learning

## Overview

Computer vision task that reconstructs high-res images from low-res images.

Objective: Upscale images by a 4x factor while preserving important details such as edges, textures and structures

Approaches used and compared:

- Bicubic interpolation
- SRCNN
- ESPCN

Evaluated both Reconstruction quality and computational efficiency

---

# Problem statement

Low-res images lose information due to downsampling and compression

The models implemented aim to recover a high res version of the low res images.


# Dataset

## DIV2K dataset
    Used DIV2K dataset for 800 HR images.
   
Split: 
   Training: 700
   Validation: 50
   Testing: 50

## Preprocessing

High res image
   |
   |
128x128 HR patch
   |
   |
Downsampling x4
   |
   |
32x32 LR image


The task is then formulated as:

LR image -> neural network -> HR image

# Models

## Bicubic interpolation

Estimates missing pixels using surrounding pixel information without learning from data

## SRCNN

Super resolution convulational neural network is one of the classic deep learning models for image super-res

- Operates on high res space
- Simple CNN architecture

## ESPCN

Efficient subpixel conv neural network is an improved neural network that improves efficiency by performing feature extraction on low res space and applies upscaling at the end.

- Lower computational cost
- Efficient upscaling using PixelShuffle

# Training 

## Framework

- PyTorch

- GPU: RTX 4070 Laptop

## Hyperparameters

Loss function: L1 Loss 

   Directly minimizes absolute pixel-wise difference.
   Chosen over L2 loss because L1 Loss is less sensitive to outliers and sometimes gives sharper reconstructions.

Optimizer: Adam

Lr: 0.0001

Epochs: 50

Batch size: 16


# Evaluation metrics

## PSNR

- Peak signal-to-noise ratio measures pixel level similarity 
- Higher is better

## SSIM

- Structural similarity index measures preservation of luminance, contrat, structural info.
- Higher is better

## Inference time

Avg forward pass latency

Lower is better

# Result analysis

- Bicubic interpolation achieves strong SSIM and PSNR performance because eval metrics are pixel based
- SRCNN provides good reconstruction quality using a simple CNN arch
- ESPCN provides much better inference time because of higher computing efficiency

Although the neural networks do not outperform bicubic interpolation, they could outperform and scale better on more training and complex scenarios.

# Installation

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

# Training

For ESPCN:

```bash
python train.py --config configs/espcn_x4.json
```

For SRCNN:

```bash
python train.py --config configs/srcnn_x4.json
```

# Evaluation

```bash
python evaluate.py \
--config configs/espcn_x4.json \
--checkpoint checkpoints/espcn_best.pth
```

# Inference

For ESPCN:

```bash
python inference.py --config configs\espcn_x4.json --checkpoint checkpoints\espcn_best.pth --image demo_images\your_image.png
```

For SRCNN:

```bash
python inference.py --config configs\srcnn_x4.json --checkpoint checkpoints\srcnn_best.pth --image demo_images\your_image.png
```
# Benchmark

```bash
python benchmark.py
```

# What can be improved:

- Training on larger datasets
- Deeper architectures can be implemented