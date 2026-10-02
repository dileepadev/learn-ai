---
title: Swin Transformer - Hierarchical Vision Transformer with Shifted Windows
description: Learn how Swin Transformer uses windowed self-attention and shifted windows to build a hierarchical, linear-complexity backbone for dense vision tasks.
---

The Swin Transformer (Shifted Window Transformer) adapts the transformer architecture to vision in a way that scales to high-resolution images and works as a general-purpose backbone for classification, detection, and segmentation.

## Problems with the Plain ViT

A standard Vision Transformer computes global self-attention over all patches, which costs quadratic time and memory in the number of patches, and it produces a single-resolution feature map. Dense tasks need multi-scale features like those a CNN produces.

## Windowed Self-Attention

Swin partitions the feature map into non-overlapping windows (for example 7x7 patches) and computes self-attention only within each window. Complexity becomes linear in image size because window size is fixed.

## Shifted Windows

Isolated windows cannot exchange information. Swin alternates two block types:

1. **W-MSA**: attention within regular windows
2. **SW-MSA**: attention within windows shifted by half the window size

The shift lets neighboring windows connect across layers, giving cross-window communication while keeping cost low. An efficient cyclic-shift implementation with attention masking avoids padding overhead.

## Hierarchical Structure

```text
Stage 1: H/4  x W/4  , C channels
Stage 2: H/8  x W/8  , 2C   (patch merging)
Stage 3: H/16 x W/16 , 4C
Stage 4: H/32 x W/32 , 8C
```

Patch merging concatenates 2x2 neighboring patches and projects them, halving resolution and doubling channels, much like CNN downsampling. This produces feature pyramids compatible with detectors and segmentation heads such as FPN and UperNet.

## Relative Position Bias

Instead of absolute position embeddings, Swin adds a learned relative position bias to attention logits, which generalizes better across window and input sizes.

## Takeaways

Swin showed that injecting locality and hierarchy into transformers yields strong accuracy and efficiency, and it became a common backbone for dense prediction.
