---
title: Squeeze-and-Excitation Networks - Channel Attention
description: Learn how Squeeze-and-Excitation blocks recalibrate feature channels using global context, and how this lightweight attention improves many CNN architectures.
---

Squeeze-and-Excitation (SE) networks introduced a simple module that lets a convolutional network reweight its feature channels based on global information. SE-Net won the ILSVRC 2017 classification challenge, and SE blocks were later adopted in architectures such as EfficientNet and MobileNetV3.

## Motivation

Convolutions mix information locally, so each channel's activation is computed without knowing the global content of the image. SE adds an explicit mechanism to model interdependencies between channels.

## The Three Steps

1. **Squeeze**: global average pooling collapses each HxW channel map to one scalar, giving a channel descriptor of size C
2. **Excitation**: a small bottleneck MLP produces per-channel weights in (0, 1)
3. **Scale**: multiply each original channel by its weight

```python
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w
```

The reduction ratio r (commonly 16) controls the bottleneck width and parameter cost.

## Where to Insert It

SE blocks plug into residual blocks before the skip addition, adding only a small percentage of parameters and compute while consistently improving accuracy.

## Related Ideas

- **CBAM** adds spatial attention alongside channel attention
- **ECA-Net** replaces the MLP with a cheap 1D convolution
- Self-attention generalizes the idea of dynamic reweighting to positions and tokens

## Limitations

Global pooling discards spatial information, so SE captures only channel-wise context; the gain on already strong models is modest.
