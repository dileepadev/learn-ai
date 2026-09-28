---
title: ConvNeXt - Modernizing Convolutional Networks
description: Learn how ConvNeXt applies transformer-era design choices to a pure ConvNet, matching vision transformer accuracy while keeping convolutional simplicity and efficiency.
---

ConvNeXt is a family of pure convolutional networks created by gradually modernizing a ResNet with design choices borrowed from Vision Transformers. The result matched or exceeded Swin Transformer accuracy, showing that much of ViT's advantage came from training recipes and macro design rather than attention itself.

## The Modernization Roadmap

Starting from ResNet-50, the authors applied a sequence of changes, measuring each one:

| Change | Idea borrowed from |
|--------|--------------------|
| Stronger training recipe (AdamW, long schedules, heavy augmentation) | Transformer training |
| Stage compute ratio changed to 3:3:9:3 | Swin stage ratios |
| "Patchify" stem: 4x4 stride-4 convolution | ViT patch embedding |
| Depthwise convolution, wider channels | Self-attention's per-channel mixing |
| Inverted bottleneck | Transformer MLP expansion |
| Larger 7x7 kernels | Larger receptive field of attention |
| GELU instead of ReLU, fewer activations | Transformer blocks |
| LayerNorm instead of BatchNorm | Transformer normalization |

## The ConvNeXt Block

```text
x -> 7x7 depthwise conv
  -> LayerNorm
  -> 1x1 conv (expand 4x)
  -> GELU
  -> 1x1 conv (project)
  -> (+) residual
```

The structure mirrors a transformer block: a spatial mixing step (depthwise conv in place of attention) followed by a channel-mixing MLP.

## Why It Matters

- Simple to implement and deploy, with standard convolution operators
- Strong transfer to detection and segmentation
- Efficient on hardware optimized for convolutions
- A reminder to compare architectures under equal training recipes

ConvNeXt V2 adds a fully convolutional masked-autoencoder pretraining scheme and Global Response Normalization.
