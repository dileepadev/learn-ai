---
title: Self-Supervised Learning for Vision
description: Explore how self-supervised learning methods — including SimCLR, MoCo, DINO, MAE, and DINOv2 — enable learning powerful visual representations from unlabeled images, eliminating the need for large annotated datasets.
---

**Self-supervised learning (SSL) for vision** is a family of methods that learn powerful visual representations from unlabeled images by constructing supervision signals from the data itself. Rather than requiring manually annotated labels, SSL methods define **pretext tasks** — auxiliary objectives whose solutions require the model to learn useful, generalizable representations of visual content. The resulting pretrained models transfer extremely well to downstream tasks such as image classification, object detection, and segmentation, often matching or surpassing supervised baselines when fine-tuned with far fewer labeled examples.

SSL for vision has experienced rapid progress since 2020, moving from methods requiring careful negative sampling (contrastive learning) to approaches that learn directly from image reconstructions or feature self-distillation — producing dense visual features that generalize across a remarkably broad range of downstream tasks.

## The Motivation: Supervision Without Labels

ImageNet pretraining — fine-tuning models pretrained on ImageNet's 1.2 million labeled images — was the dominant transfer learning paradigm for computer vision from 2012 through 2020. Its limitations became increasingly apparent:

- ImageNet labels cover a finite set of 1,000 classes — a tiny fraction of visual concepts.
- Labeling large datasets is expensive and limits scaling.
- The Internet contains billions of unlabeled images that carry far more visual diversity than any labeled dataset.

SSL unlocks this vast unlabeled resource. Models pretrained on billions of Instagram photos (DINO), internet crawls (DINOv2), or even the same few hundred million ImageNet images without labels can match or exceed the representation quality of supervised ImageNet pretraining.

## Contrastive Learning

### SimCLR

**SimCLR** (Chen et al., 2020, Google) is the foundational modern contrastive SSL method. Its key insight: two different **augmented views** of the same image should produce similar representations, while views from different images should be dissimilar.

**Training procedure:**

1. For each image $x$ in a batch of $N$ images, generate two augmented views $\tilde{x}_i$ and $\tilde{x}_j$ using random augmentations (random crop, horizontal flip, color jitter, grayscale, Gaussian blur).
2. Encode both views through a shared CNN backbone and projection head: $z_i = g(f(\tilde{x}_i))$, $z_j = g(f(\tilde{x}_j))$.
3. Optimize the **NT-Xent (Normalized Temperature-scaled Cross-Entropy)** contrastive loss:

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau$ is a temperature parameter. Each positive pair $(i, j)$ is contrasted against all $2N - 2$ other views in the batch as negatives.

SimCLR requires **large batch sizes** (4,096–8,192) to provide sufficient negatives for effective contrastive learning — a significant compute requirement.

**Key findings from SimCLR:**

- Data augmentation choice is critical — random crop followed by color jitter is the most impactful combination.
- A nonlinear projection head improves representation quality; the representation before the head (the backbone output) is what is used for downstream tasks.
- Temperature scaling significantly affects training stability and final representation quality.

### MoCo: Momentum Contrast

**MoCo** (He et al., 2020, FAIR) addresses SimCLR's large-batch requirement through a **memory bank** (queue) of negatives and a **momentum encoder**:

- A queue stores the representations of the last $K$ (e.g., 65,536) encoded views — providing many negatives without requiring a large simultaneous batch.
- The **query encoder** is updated by gradient descent; the **key encoder** (momentum encoder) is updated by exponential moving average:

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

The momentum encoder ensures that negatives in the queue are encoded by a slowly changing, consistent encoder — preventing the representation instability that arises if the encoder changes too rapidly.

MoCo **v3** adapts the approach to Vision Transformers (ViTs), achieving strong performance on large-scale SSL benchmarks.

## Non-Contrastive Methods

A significant limitation of contrastive methods is **representation collapse** — without negative pairs, the model could trivially satisfy the objective by mapping all inputs to the same constant representation. Several methods resolve this without negatives.

### BYOL: Bootstrap Your Own Latent

**BYOL** (Grill et al., 2020, DeepMind) eliminates negative pairs entirely using an asymmetric architecture:

- **Online network**: Encoder + projector + **predictor** — updated by gradient descent.
- **Target network**: Encoder + projector (no predictor) — updated by exponential moving average of the online network.

The online network is trained to predict the target network's representations of a different augmented view of the same image. The asymmetry (predictor only in online network) and EMA update of the target prevent collapse.

BYOL's surprising success without negatives sparked significant theoretical investigation into why it doesn't collapse, leading to a richer understanding of what prevents representation collapse.

### SimSiam

**SimSiam** (Chen & He, 2020) pushes simplicity even further — no negatives, no momentum encoder, no large batch:

- Two views encoded by a shared encoder.
- A predictor on one branch predicts the representation of the other branch (stop-gradient applied to the target).

The **stop-gradient** operation is critical — without it, SimSiam collapses. With it, SimSiam learns competitive representations despite being simpler than BYOL.

### Barlow Twins

**Barlow Twins** (Zbontar et al., 2021, FAIR) takes a different approach: rather than matching representations directly, it minimizes the cross-correlation matrix between two views' representations toward the identity matrix:

$$\mathcal{L} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2$$

The first term encourages invariance to augmentation (diagonal elements should be 1); the second term discourages feature redundancy (off-diagonal elements should be 0). This objective inherently decorrelates features, encouraging the model to use the full representation capacity.

## DINO and Vision Transformers

**DINO** (Self-DIstillation with NO labels, Caron et al., 2021, FAIR) applied self-supervised learning to Vision Transformers and produced representations with remarkable emergent properties.

### DINO Architecture

DINO uses a **student-teacher framework** similar to BYOL:

- **Multiple views**: Two global crops (large portions of the image) and several local crops (small portions).
- **Student** processes all views (global and local); **teacher** processes only global views.
- The student is trained to match the teacher's output distribution — using a softmax with a **sharpening** temperature for the teacher.
- The teacher is updated by EMA of the student.

### Emergent Segmentation Properties

DINO-pretrained ViTs produce **self-attention maps that segment objects without any segmentation supervision**. The attention maps of the [CLS] token's attention heads align closely with salient object regions — a property not seen in supervised or contrastive SSL models.

This emergent segmentation capability suggests that DINO's representations capture semantic structure rather than texture statistics — a significant qualitative difference from previous SSL methods.

### DINOv2

**DINOv2** (Oquab et al., 2023, FAIR) scales DINO with several improvements:

- **Curated pretraining data**: A carefully curated dataset of 142 million images (LVD-142M) assembled via retrieval-based deduplication and quality filtering from the internet.
- **Improved objective**: Combines DINO's self-distillation with iBOT's masked image modeling — the teacher predicts both the full image and masked patches.
- **Larger models**: ViT-g (1.1B parameters) as the largest variant.

DINOv2 features transfer to an extraordinary range of tasks with simple linear probes — depth estimation, semantic segmentation, monocular video depth, action recognition, instance retrieval, and fine-grained recognition — without task-specific fine-tuning.

## Masked Image Modeling

### MAE: Masked Autoencoders

**MAE** (He et al., 2021, FAIR) takes a radically different approach — predicting pixel values of masked image patches:

1. Divide the image into a grid of non-overlapping patches (e.g., 16×16 pixels).
2. **Mask** a large fraction (75%) of patches randomly.
3. Encode only the visible patches with a ViT encoder.
4. Decode the full set of patches (visible + masked tokens) with a lightweight decoder, predicting pixel values of masked patches.
5. Optimize reconstruction loss only on masked patches.

The **high masking ratio** is critical — it makes the task challenging enough that the model cannot rely on simple interpolation, forcing it to learn global semantic structure to reconstruct the missing patches.

MAE is computationally efficient because the encoder only processes visible patches (25% of the total) — encoding a full image is 3–4× cheaper than processing all patches.

### BEiT and Discrete Token Prediction

**BEiT** (Bao et al., 2022, Microsoft) predicts discrete **visual tokens** rather than raw pixels for masked patches:

1. A VQ-VAE tokenizer converts image patches into discrete vocabulary tokens.
2. The ViT encoder predicts the tokenizer's output for masked patches.

This formulation mirrors BERT's masked language modeling exactly — applied to vision by treating visual tokens as the "words." The discrete target provides cleaner training signal than noisy pixel values.

**BEiT v2** and **data2vec** extend this approach to predict high-level representation targets (teacher model outputs) rather than pixel values or discrete tokens.

## CLIP and Contrastive Language-Image Pretraining

**CLIP** (Radford et al., 2021, OpenAI) learns visual representations through **image-text contrastive learning** — pairing 400 million image-caption pairs from the internet:

- An image encoder (ViT or ResNet) and text encoder (Transformer) are trained jointly.
- The objective: the representation of an image and its matching caption should be similar; image-caption pairs from different examples should be dissimilar.

CLIP's zero-shot transfer is remarkable: to classify images into arbitrary categories, embed the category names as text ("a photo of a dog") and find the closest text embedding to each image — no fine-tuning required.

CLIP representations have become foundational in multimodal AI, serving as the visual encoder in DALL-E 2, Stable Diffusion, LLaVA, and many other vision-language systems.

## Evaluation Protocols

SSL representations are evaluated through standardized protocols:

- **Linear probe**: Freeze the pretrained backbone; train a single linear classifier on frozen features using labeled data. Tests pure representation quality.
- **k-NN evaluation**: Find the $k$ nearest neighbors in the training set using frozen features; classify by majority vote. Zero learning, tests feature quality directly.
- **Fine-tuning**: Update all backbone weights with labeled data. Tests the representations as initialization for full supervised training.
- **Few-shot evaluation**: Fine-tune with 1–100 labeled examples per class to assess data efficiency.

The gap between linear probe and fine-tuning accuracy reveals how much structured information is stored in the pretrained representation vs. requiring task-specific learning.

## Practical Impact

Self-supervised vision pretraining has made high-quality visual representations accessible without large annotated datasets:

- Medical imaging: SSL on unlabeled radiology images before fine-tuning on small labeled clinical datasets.
- Satellite imagery: SSL on massive unlabeled satellite image archives for land use classification and change detection.
- Industrial inspection: SSL pretraining on product images before fine-tuning anomaly detection with few defect examples.

DINOv2 in particular has become a popular general-purpose visual backbone — deployable across diverse tasks with minimal fine-tuning, analogous to the role BERT plays in NLP.
