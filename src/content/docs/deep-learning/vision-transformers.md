---
title: Vision Transformers (ViT)
description: How the Transformer architecture was adapted for image understanding, replacing convolutional networks as the state of the art in computer vision.
---

Vision Transformers (ViT) apply the Transformer architecture — originally designed for text — directly to images. Introduced by Google Brain in 2020, ViT demonstrated that a pure Transformer without convolutional inductive biases can match or exceed CNNs on image classification when trained on sufficient data. It has since become the foundation of modern computer vision.

## From Text to Images

The key challenge in applying Transformers to images is that images are 2D grids of pixels — not sequences of tokens. ViT solves this by dividing the image into a grid of fixed-size **patches** (e.g., 16×16 pixels), flattening each patch into a vector, and treating the sequence of patch vectors as tokens.

A learnable **[CLS] token** is prepended to the sequence (borrowed from BERT), and its final representation is used for classification. **Positional embeddings** encode each patch's location in the image.

## Architecture

1. Split image into N patches of size P×P (e.g., a 224×224 image → 196 patches of 16×16).
2. Linearly project each patch to an embedding dimension D.
3. Add positional embeddings.
4. Pass through L standard Transformer encoder layers (multi-head self-attention + MLP).
5. Use the [CLS] token's output for the final prediction.

This is nearly identical to BERT's architecture — just with image patches instead of word tokens.

## Why ViT Outperforms CNNs at Scale

CNNs have strong inductive biases: translation equivariance and local receptive fields. These help on small datasets but can limit what the model learns. ViT has weaker inductive biases — it learns spatial relationships from data, not from architecture. Given enough data (like JFT-300M), this pays off with better representations.

On smaller datasets, CNNs still have an advantage due to their built-in priors. Data augmentation and training tricks (DeiT) close much of this gap.

## Key Variants

- **DeiT (Data-efficient Image Transformers):** Trains ViT on ImageNet-1k alone using knowledge distillation from a CNN teacher.
- **Swin Transformer:** Uses shifted window attention for efficiency — computes attention within local windows rather than globally. State of the art for detection and segmentation.
- **BEiT / MAE (Masked Autoencoders):** Self-supervised pretraining for ViT — mask patches and reconstruct them, analogous to BERT's masked language modeling.
- **DINO / DINOv2:** Self-supervised ViT with excellent zero-shot transfer properties.
- **EVA, SigLIP, InternViT:** Vision encoders used in modern multimodal LLMs.

## Applications

- **Image classification** (ImageNet and beyond)
- **Object detection and segmentation** (Swin-based detectors)
- **Vision-language models:** ViT is the visual encoder in CLIP, LLaVA, GPT-4V, Gemini
- **Medical imaging:** Pathology slides, radiology, ophthalmology
- **Remote sensing:** Satellite and aerial image analysis

## ViT vs. CNN: When to Use Which

| | ViT | CNN |
|---|---|---|
| Large datasets | ✅ Excellent | ✅ Good |
| Small datasets | ⚠️ Needs tricks | ✅ Better |
| Global context | ✅ Self-attention | ❌ Limited |
| Compute efficiency | ⚠️ Quadratic attention | ✅ More efficient |
| Multimodal integration | ✅ Natural | ⚠️ Harder |

For most new research and production vision systems at scale, ViT-based architectures are now the default choice.
