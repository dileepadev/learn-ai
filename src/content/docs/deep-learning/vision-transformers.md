---
title: Vision Transformers (ViT)
description: Explore how the Transformer architecture was adapted for computer vision — enabling state-of-the-art image recognition, segmentation, and visual representation learning.
---

**Vision Transformers (ViT)** brought the Transformer architecture — originally designed for NLP — directly to computer vision. Introduced by Dosovitskiy et al. (2020), ViT demonstrated that pure self-attention over image patches could match or surpass Convolutional Neural Networks (CNNs) on large-scale image recognition tasks.

## From CNNs to Transformers

CNNs dominated computer vision for nearly a decade due to their built-in **inductive biases**:

- **Translation equivariance** — detecting a feature in one location generalizes to others.
- **Local connectivity** — convolutions operate on local patches.

These biases are useful when data is scarce, but they can become a bottleneck at scale. Transformers have far weaker inductive biases, making them harder to train with limited data — but when trained on large datasets, they learn better global representations.

## How ViT Works

### Patch Embedding

An input image of size $H \times W \times C$ is divided into $N$ non-overlapping patches of size $P \times P$:

$$N = \frac{H \times W}{P^2}$$

Each patch is flattened and linearly projected into a $D$-dimensional embedding — analogous to token embeddings in NLP.

### Adding Positional Embeddings

Since Transformers are permutation-invariant, **positional embeddings** are added to the patch embeddings to encode spatial location. Standard ViT uses 1D learnable positional embeddings (treating patches as a sequence).

### Classification Token

A special learnable `[CLS]` token is prepended to the patch sequence, mirroring BERT. The final hidden state of the `[CLS]` token is used as the image representation for classification.

### Transformer Encoder

The sequence of patch embeddings passes through a standard Transformer encoder: Multi-Head Self-Attention (MHSA) → Layer Norm → Feed-Forward Network (FFN), repeated $L$ times.

```
Image → Patches → Linear Projection → + Positional Embeddings
     → Transformer Encoder (×L) → [CLS] token → Classification Head
```

## Scaling Behavior

Unlike CNNs, ViT models require large-scale pre-training to shine:

| Model | Params | Pre-train Data | ImageNet-1K Top-1 |
|---|---|---|---|
| ViT-B/16 | 86M | ImageNet-21K | 85.8% |
| ViT-L/16 | 307M | ImageNet-21K | 87.8% |
| ViT-H/14 | 632M | JFT-3B | 90.5% |

With sufficient data, ViT outperforms CNN-based models. Without it, CNNs trained on the same data generally win.

## Data-Efficient ViT Variants

Several methods address ViT's data hunger:

### DeiT (Data-efficient Image Transformers)

Facebook AI's DeiT trains ViT-scale models on **ImageNet-1K alone** (no extra data) using:

- **Knowledge distillation** from a CNN teacher.
- A **distillation token** alongside the `[CLS]` token during training.

DeiT matched ViT's performance without requiring the massive JFT or ImageNet-21K datasets.

### Swin Transformer

The **Swin Transformer** reintroduces hierarchical features and local attention windows — combining ViT's strengths with CNN-like inductive biases:

- Operates attention within **local windows** (efficient — linear in image size instead of quadratic).
- **Shifted windows** across layers enable cross-window information flow.
- Produces multi-scale feature maps, making it compatible with dense prediction tasks (detection, segmentation).

Swin became a standard backbone for many vision tasks after outperforming ViT on COCO object detection.

## Self-Supervised ViT: DINO and MAE

### DINO

DINO trains ViT using **self-distillation** — no labels required. A student network learns to match the output of a slowly updating teacher (momentum encoder). The resulting features:

- Exhibit strong semantic segmentation properties *without supervision*.
- Produce attention maps that naturally highlight object boundaries.

### Masked Autoencoders (MAE)

MAE adapts the BERT-style masked pre-training to images. A high fraction (75%) of patches are masked, and the model learns to reconstruct the missing ones. This produces:

- Rich, transferable image representations.
- Highly efficient pre-training (fewer visible patches = less compute per step).

## Dense Prediction: Detection and Segmentation

ViT was initially designed for classification. Adapting it for dense tasks required new architectures:

- **ViTDet** — Uses a plain ViT backbone with feature pyramid construction for object detection.
- **Segment Anything Model (SAM)** — Uses a ViT encoder to enable promptable, zero-shot segmentation of anything in an image.
- **SETR / Segmenter** — Apply ViT encoders with transformer decoders for semantic segmentation.

## ViT vs. CNN — A Summary

| Property | CNN | ViT |
|---|---|---|
| Inductive bias | Strong (local, translation equivariant) | Weak |
| Data efficiency | High | Requires large pre-training |
| Global context | Limited (deep layers only) | Every layer |
| Scalability | Good | Excellent |
| Dense tasks (detection, segmentation) | Excellent (native hierarchies) | Requires adaptation (Swin, MAE, SAM) |

## Practical Considerations

- For **limited data** scenarios, use CNN backbones or DeiT with strong augmentation.
- For **large-scale pre-training**, ViT/Swin are preferred.
- For **dense prediction**, Swin Transformer or ViTDet are strong choices.
- Pre-trained ViT checkpoints (from HuggingFace or timm) can be fine-tuned on downstream tasks with minimal data.

Vision Transformers represent a paradigm shift in computer vision — moving from hand-crafted locality priors toward scalable, data-driven global modeling that increasingly unifies vision and language under a common architectural family.
