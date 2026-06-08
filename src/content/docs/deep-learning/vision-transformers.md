---
title: Vision Transformers (ViT)
description: How the Transformer architecture was adapted for image understanding, becoming the foundation of modern computer vision.
---

Vision Transformers (ViT) apply the Transformer architecture directly to images. Introduced by Google in 2020, ViT demonstrated that a pure Transformer — without convolutional biases — can match or exceed CNNs on image classification when trained at scale. It is now the backbone of modern computer vision and multimodal AI.

## The Patch Embedding Trick

Transformers expect sequences of tokens. ViT converts an image into a sequence by splitting it into fixed-size **patches** (e.g., 16×16 pixels), flattening each patch into a vector, and projecting it to an embedding dimension. A 224×224 image becomes 196 patch tokens.

A learnable **[CLS] token** is prepended; its final representation is used for classification. Positional embeddings encode each patch's location.

## Architecture

1. Split image → N patches of size P×P.
2. Linear projection of each patch to dimension D.
3. Add positional embeddings.
4. Pass through standard Transformer encoder layers.
5. Use [CLS] token output for prediction.

This is nearly identical to BERT — just with image patches instead of words.

## Key Variants

- **DeiT:** Trains ViT on ImageNet alone using distillation from a CNN teacher — no large private datasets needed.
- **Swin Transformer:** Shifted window attention for efficiency; state of the art for detection and segmentation.
- **MAE (Masked Autoencoders):** Self-supervised ViT pretraining — mask random patches and reconstruct them.
- **DINOv2:** Self-supervised ViT with strong zero-shot transfer for many vision tasks.

## Why It Matters

ViT is the visual encoder in CLIP, LLaVA, GPT-4o, and Gemini. Understanding ViT is essential for understanding how modern multimodal models process images. With sufficient data, it outperforms CNNs because self-attention captures global context that convolutions — limited to local receptive fields — cannot.
