---
title: Self-Supervised Learning
description: Learning rich representations from unlabeled data — contrastive learning, masked language modeling, and vision pretraining without labels.
---

**Self-supervised learning** trains models on large unlabeled datasets by creating supervision signals from the data itself. Rather than relying on expensive manual labels, the model learns to solve pretext tasks that require understanding the data structure.

This has become the dominant pretraining paradigm for both NLP (BERT, GPT) and computer vision (DINO, SimCLR), enabling models to learn powerful, general-purpose representations.

## The Core Idea

Standard supervised learning requires labels: {(image, class), (text, label), ...}. Self-supervised learning creates auxiliary tasks:

- **NLP**: Predict masked words from context (BERT).
- **Vision**: Predict missing image regions or rotations.
- **Contrastive**: Learn that augmented views of the same sample should have similar embeddings, while different samples diverge.

These pretext tasks are solved without manual annotation, yet the learned representations transfer well to downstream tasks.

## Contrastive Learning

**Contrastive learning** is the most successful self-supervised approach: learn embeddings where similar samples cluster together while dissimilar samples separate.

### SimCLR (Simple Contrastive Learning of Representations)

**SimCLR** (Chen et al., 2020):

1. **Augment** each image with two random crops and augmentations (rotation, color jitter, blur): $x_i, x_j$.
2. **Encode** both through a CNN encoder: $h_i = f(x_i), h_j = f(x_j)$.
3. **Project** to a lower-dimensional space: $z_i = g(h_i)$.
4. **Contrastive loss**: Maximize similarity of $z_i$ and $z_j$, minimize similarity to other samples in the batch:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where sim is cosine similarity and $\tau$ is temperature.

**Result**: Representations learn to cluster augmented views of the same image while separating different images. Remarkably effective — supervised accuracy on ImageNet with frozen ResNet-50 features: ~60% (vs. ~69% with supervised training on ImageNet).

### MoCo (Momentum Contrast)

**MoCo** uses a **momentum-updated** encoder and a **queue** of negative samples:

- Maintain a query encoder (updated normally) and a key encoder (updated via momentum).
- Store a large queue of past embeddings as negatives.

This allows large batches effectively without large GPU memory, improving learning.

### BYOL (Bootstrap Your Own Latent)

Surprisingly, contrastive learning without explicit negatives works:

1. Augment an image two ways: $x$ and $x'$.
2. Encode $x$ with learnable encoder, encode $x'$ with the same encoder (stopped gradients).
3. Minimize MSE between their projections.

Without negative samples, why doesn't the encoder collapse (all embeddings identical)? The stopped gradients prevent trivial solutions. BYOL shows self-supervised learning can work without negatives — challenging prior intuitions.

## Masked Language Modeling (MLM)

**BERT** introduced masked language modeling:

1. Randomly mask 15% of tokens.
2. Predict the masked tokens from context.

$$\mathcal{L}_{\text{MLM}} = -\sum_i \mathbb{1}[\text{token}_i \text{ masked}] \log P(\text{token}_i | \text{context})$$

This forces the model to build bidirectional context understanding. Remarkably effective: BERT pretraining significantly improves downstream NLP tasks with little fine-tuning.

### GPT: Causal Language Modeling

**GPT** uses a simpler self-supervised task: predict the next token:

$$\mathcal{L}_{\text{CLM}} = -\sum_i \log P(\text{token}_{i+1} | \text{token}_1, ..., \text{token}_i)$$

Unidirectional but naturally suited to generation. Foundation of today's large language models.

## Multimodal Self-Supervised Learning

### CLIP

**CLIP** learns from image-text pairs:

1. Encode an image: $u = \text{ImageEncoder}(\text{image})$.
2. Encode text: $v = \text{TextEncoder}(\text{text})$.
3. Contrastive loss: Align image and text embeddings for matching pairs; separate for mismatches.

Learned representations transfer to zero-shot classification: the model can classify images into arbitrary categories (described in text) without seeing any examples.

## Pretraining vs. Fine-Tuning

Self-supervised pretraining has become dominant:

1. **Pretrain** on massive unlabeled data (ImageNet, Common Crawl, web).
2. **Fine-tune** on downstream tasks with limited labels.

This two-stage approach outperforms end-to-end supervised training on small-label-budget tasks. The pretrained representations are general and transferable.

## Representation Quality

**What makes good self-supervised representations?**

- **Invariance**: Robust to augmentations and noise.
- **Equivariance**: Sensitive to meaningful changes.
- **Disentanglement**: Separate semantic factors of variation.

Contrastive learning encourages invariance (augmented views similar). But representations often remain entangled. Research into disentangled and interpretable self-supervised learning is active.

## Downstream Adaptation

How to use self-supervised representations for downstream tasks?

### Linear Evaluation Protocol

Freeze the pretrained encoder; train only a linear classifier on top. This measures representation quality directly without fine-tuning confounds.

### Fine-Tuning

Update the entire model on the downstream task. Often improves performance but risks overfitting on small datasets.

### Few-Shot Learning

Leverage pretrained representations for few-shot tasks. Self-supervised models often outperform supervised baselines.

## Challenges and Trade-offs

### Representation Collapse

Contrastive models can collapse: all samples map to the same embedding (trivial solution). Mechanisms preventing collapse:
- **Large batch sizes** (many negatives).
- **Negative samples from queue** (MoCo).
- **Stopped gradients** (BYOL).

### Computational Cost

Pretraining on massive datasets is expensive. CLIP trained on 400M image-text pairs for 32 TPU v3 days (tens of thousands of dollars). Compute budget is a significant barrier to reproducibility and democratization.

### Task-Representation Mismatch

Representations learned via one pretext task may not transfer well to different downstream tasks. Domain-specific pretraining (medical imaging, drug discovery) remains important.

### Scaling Laws

Self-supervised models benefit enormously from scale (more data, bigger models). This drives consolidation toward well-resourced labs and companies.

## Applications

### Domain-Specific Learning

Self-supervised pretraining on domain-specific unlabeled data (e.g., medical images, molecular structures) improves downstream task performance with limited labels.

### Continual Learning

Self-supervised objectives enable continual learning without catastrophic forgetting — models can learn from streaming data without labeled examples.

### Anomaly Detection

Learn normal data distribution unsupervised; anomalies have lower likelihood or different embeddings.

## Current Research Directions

- **Scaling efficiency**: Reduce computational cost of pretraining.
- **Theoretical understanding**: Why does contrastive learning work? Information-theoretic perspectives.
- **Multimodal alignment**: Extend CLIP-like approaches to audio, 3D, and other modalities.
- **Federated self-supervised learning**: Pretrain across decentralized data without centralization.

Self-supervised learning has fundamentally changed how AI systems are developed, enabling powerful models from unlabeled data — a more practical and scalable paradigm than supervised learning alone.
