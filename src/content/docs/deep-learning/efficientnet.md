---
title: EfficientNet - Compound Scaling for CNNs
description: Learn how EfficientNet uses neural architecture search and compound scaling of depth, width, and resolution to achieve strong accuracy per parameter.
---

EfficientNet is a family of convolutional networks (B0 through B7 and beyond) that achieves high accuracy with far fewer parameters and FLOPs than earlier CNNs by scaling network dimensions in a principled, balanced way.

## The Scaling Problem

Practitioners traditionally scaled a CNN along a single axis: more layers (depth), more channels (width), or larger inputs (resolution). Scaling only one dimension saturates quickly. Intuitively, a larger image needs more depth for a bigger receptive field and more width to capture finer patterns.

## Compound Scaling

EfficientNet scales all three dimensions together using a single coefficient phi:

```text
depth      d = alpha ^ phi
width      w = beta  ^ phi
resolution r = gamma ^ phi

subject to  alpha * beta^2 * gamma^2 ~= 2   (FLOPs roughly double per unit phi)
```

A small grid search on the baseline finds alpha, beta, gamma; then phi is raised to produce larger models.

## The Baseline: EfficientNet-B0

The B0 backbone was found through neural architecture search optimizing accuracy under a FLOPs target. Its core building block is the **MBConv**:

- Inverted residual: 1x1 expansion, depthwise convolution, 1x1 projection
- **Squeeze-and-excitation** channel attention inside the block
- Swish (SiLU) activation

## Using It

```python
import timm
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=10)
```

## EfficientNetV2

V2 adds Fused-MBConv layers in early stages (faster on accelerators), progressive learning that grows image size and regularization during training, and a scaling rule tuned for training speed.

## Practical Notes

High accuracy per FLOP does not always mean low latency; depthwise convolutions can be memory-bound on some hardware, so benchmark on your target device.
