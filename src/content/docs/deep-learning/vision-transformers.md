---
title: Vision Transformers
description: Extending the transformer architecture to computer vision — how ViT achieves state-of-the-art image classification by treating images as sequences of patches.
---

**Vision Transformers (ViT)** extend the transformer architecture, originally developed for NLP, to image understanding. Rather than relying on convolutional neural networks (CNNs), ViT treats an image as a sequence of non-overlapping patches and applies the transformer directly — achieving excellent results on image classification and opening new possibilities for vision tasks.

## Why Transformers for Vision?

Traditional CNNs rely on **local receptive fields** and **weight sharing** to extract hierarchical features. While effective, they have limitations:

1. **Locality bias**: Early layers are limited to local patterns; global context requires many layers.
2. **Inductive bias toward images**: Weight sharing and convolutions assume grid structure, but this inductive bias is expensive for training from scratch.
3. **Limited flexibility**: CNNs are specialized for images; transformers are general-purpose sequence processors.

Transformers, by contrast, compute **global attention** between all elements, giving each token (or patch) direct access to all others from the first layer. This suggests they might learn more efficiently on large-scale data.

## Vision Transformer Architecture

### Patch Embedding

The first step converts an image into a sequence of patches:

1. **Split the image** into non-overlapping patches. For a 224×224 image with 16×16 patches, you get 196 patches (14×14 grid).
2. **Flatten and project** each patch to a fixed embedding dimension $d$ using a linear layer.
   - A 16×16×3 patch becomes a vector of size $d$ (e.g., 768).
3. **Add position embeddings** to retain spatial information, similar to positional encoding in NLP transformers.

### Class Token and Attention

Following BERT's approach:

1. **Prepend a learnable [CLS] token** to the sequence of patch embeddings.
2. **Apply transformer layers**: each layer consists of multi-head self-attention and feed-forward sublayers with residual connections.
3. **Classification**: Use the [CLS] token's final representation (after all transformer layers) for classification.

The [CLS] token acts as a "global pooling" mechanism, learning to aggregate information from all patches via attention.

### Architecture Details

A typical ViT-Base has:
- **12 transformer layers**
- **12 attention heads**
- **768 hidden dimensions**
- **Total parameters**: ~86M (comparable to ResNet-50 but with fewer inductive biases)

Larger variants (ViT-Large, ViT-Huge) scale up these dimensions for better performance on large datasets.

## Why ViT Works

The success of ViT depends critically on **scale**:

- **Training data scale**: ViT trained on ImageNet-1K (1.2M images) underperforms ResNet-50. But pre-trained on ImageNet-21K (14M images) or JFT-300M (300M private labeled images), ViT surpasses CNNs.
- **Reduced inductive bias**: Without convolutional structure, the model must learn spatial relationships from data. This hurts small-scale training but becomes an advantage at scale — the model can learn representations not biased by convolutional assumptions.
- **Global attention from the start**: Unlike CNNs, ViT accesses global context immediately, potentially learning better long-range dependencies.

## Learnings from ViT

### Patch Size Matters

- **Larger patches** (e.g., 32×32) reduce the sequence length and computation but lose fine-grained detail.
- **Smaller patches** (e.g., 8×8) retain detail but increase sequence length and attention cost.
- A middle ground (14×14 or 16×16) typically works best.

### Attention Patterns are Interpretable

Analyzing attention weights reveals that:
- **Early layers** focus on low-level patterns (edges, textures) with local attention.
- **Intermediate layers** begin integrating global information.
- **Later layers** attend broadly, using global context for classification.

This mirrors CNN feature hierarchies but emerges directly from the data and attention mechanism.

## Variants and Extensions

### DeiT (Data-Efficient Image Transformers)

**DeiT** (Touvron et al., 2021) achieves competitive performance on ImageNet with standard scale (1.2M images) through:

1. **Knowledge distillation**: Using a CNN teacher (e.g., ResNet) to guide ViT training.
2. **Data augmentation**: RandAugment, Mixup, CutMix to compensate for limited data.
3. **Regularization**: Stochastic depth, layer-scale initialization.

**Key insight**: Inductive biases from CNNs (via distillation) can accelerate vision transformer training.

### Swin Transformer

**Swin** (Liu et al., 2021) introduces **local window attention**:

1. **Windows**: Divide the image into local windows; attention is computed within windows.
2. **Shifted windows**: In alternate layers, shift windows to enable cross-window communication.
3. **Hierarchical structure**: Build a pyramid of features, reducing spatial resolution and increasing channels at deeper layers (like CNNs).

This hybrid approach reduces attention complexity from quadratic to linear in image size while maintaining global receptive fields through window shifts.

### CLIP and Multimodal Vision Transformers

**CLIP** (Radford et al., 2021) uses ViT as the vision encoder in a multimodal model that jointly learns image and text representations. This enables:

- **Zero-shot transfer**: Classify images into arbitrary categories (described in text) without task-specific training.
- **Robustness**: Trained on diverse web data, CLIP generalizes better to distribution shift than supervised ImageNet models.

Vision transformers excel in multimodal settings where the ability to align visual and textual representations is crucial.

## Computational Efficiency

A challenge of ViT is **attention complexity**: self-attention scales as $O(n^2)$ with sequence length $n$.

**Efficiency approaches**:
- **Local attention** (Swin, local window attention): Reduce attention to local neighborhoods.
- **Linear attention**: Approximate attention with linear complexity using kernelized attention or low-rank projections.
- **Sparse attention**: Attend to a carefully chosen subset of tokens rather than all pairs.
- **Hierarchical models**: Aggregate patches at multiple scales, reducing sequence length in deeper layers.

## Comparison with CNNs

| Property | CNN | ViT |
|----------|-----|-----|
| **Inductive Bias** | Locality, weight sharing | None (learns from data) |
| **Scalability** | Good; efficient for small images | Excellent with scale; efficient patches |
| **Interpretability** | Kernels capture low-level features | Attention patterns reveal reasoning |
| **Transfer Learning** | Strong with ImageNet pretraining | Requires larger-scale pretraining |
| **Computational Cost** | Efficient; linear in image resolution | Attention is quadratic in sequence length |

## Impact and Future Directions

ViT has fundamentally changed computer vision research:

1. **Pretraining paradigm**: Large-scale supervised or self-supervised pretraining is now standard for vision, mirroring NLP.
2. **Vision-language models**: ViT enables powerful multimodal systems (CLIP, BLIP, LLaVA).
3. **Self-supervised learning**: ViT works exceptionally well with vision self-supervised methods (MAE, DINO), learning rich representations without labels.
4. **Real-time systems**: Efficient variants enable transformers in resource-constrained settings (mobile, edge).

Vision Transformers represent a fundamental shift from hand-designed inductive biases (convolutions) to learning general-purpose transformers at scale — a trend likely to continue as compute and data grow.
