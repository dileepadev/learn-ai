---
title: "Vision Transformers (ViT): Transformers for Computer Vision"
description: "Learn how Vision Transformers apply the transformer architecture to image understanding, why they outperform CNNs at scale, and how modern variants like DINOv2 and SAM work."
---

The transformer architecture, originally designed for text, has become the dominant approach for computer vision as well. **Vision Transformers (ViT)** treat images as sequences of patches and apply standard transformer attention — achieving state-of-the-art results across classification, detection, segmentation, and generation.

## The Core Idea: Images as Patch Sequences

A ViT processes an image by:

1. **Patch extraction**: Divide the image into fixed-size patches (e.g., 16×16 pixels).
2. **Linear projection**: Flatten each patch and project it to the model's embedding dimension.
3. **Position embeddings**: Add learnable position embeddings to encode spatial location.
4. **[CLS] token**: Prepend a learnable classification token whose final representation is used for classification.
5. **Transformer encoder**: Apply standard multi-head self-attention and feed-forward layers.

For a 224×224 image with 16×16 patches, this produces 196 patch tokens — a manageable sequence length for attention.

## Why ViT Outperforms CNNs at Scale

CNNs have strong inductive biases: translation equivariance and local connectivity. These biases help with small datasets but become constraints at scale.

ViTs have weaker inductive biases — they must learn spatial relationships from data. This requires more data to train from scratch, but when pretrained on large datasets (ImageNet-21K, JFT-300M), ViTs outperform CNNs significantly.

The key insight: **scale favors ViTs**. With enough data and compute, the flexibility of attention outweighs the efficiency of convolutional inductive biases.

## Efficient ViT Variants

The quadratic attention cost is a problem for high-resolution images. Solutions include:

- **Swin Transformer**: Applies attention within local windows, with shifted windows for cross-window communication. Linear complexity in image size.
- **DeiT**: Data-efficient ViT training using knowledge distillation from a CNN teacher, enabling strong performance without massive datasets.
- **EfficientViT**: Hardware-aware design with multi-scale attention for efficient inference.

## Self-Supervised ViT Pretraining

### DINO and DINOv2
**DINO** trains ViTs with self-supervised learning using a self-distillation objective. The resulting features have remarkable properties — the attention maps naturally segment objects without any segmentation supervision.

**DINOv2** scales this up with curated data and improved training, producing general-purpose visual features that transfer well to many downstream tasks.

### MAE (Masked Autoencoders)
**MAE** pretrains ViTs by masking 75% of image patches and training the model to reconstruct them. This is highly efficient (only 25% of patches are processed by the encoder) and produces strong representations.

## Segment Anything Model (SAM)

**SAM** from Meta uses a ViT image encoder to produce dense image embeddings, then a lightweight mask decoder that generates segmentation masks from point, box, or text prompts. The ViT encoder runs once per image; the decoder runs in milliseconds for each prompt.

SAM demonstrates how a powerful ViT backbone enables flexible, interactive vision applications.

## ViTs in Multimodal Models

ViTs are the standard vision encoder in multimodal LLMs:
- **CLIP**: Trains a ViT and text encoder jointly with contrastive learning on image-text pairs.
- **LLaVA, InternVL, Qwen-VL**: Use CLIP or SigLIP ViTs as the vision backbone for vision-language models.

The quality of the vision encoder directly determines the visual understanding capability of the downstream multimodal model.
