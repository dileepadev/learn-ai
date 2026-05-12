---
title: Dataset Distillation
description: Understand dataset distillation — the task of synthesizing a small, highly informative surrogate dataset that enables a model trained on it to match the performance of a model trained on the full large dataset. Covers gradient matching, distribution matching, kernel-based methods (KIP, FRePo), trajectory matching (MTT), factorization methods, and applications in continual learning, neural architecture search, and federated learning.
---

**Dataset distillation** (also called dataset condensation) is the task of synthesizing a small **surrogate dataset** $\mathcal{S}$ of $|\mathcal{S}| \ll |\mathcal{T}|$ examples such that a model trained on $\mathcal{S}$ achieves performance comparable to one trained on the full training set $\mathcal{T}$. While **knowledge distillation** compresses a teacher model into a student model, dataset distillation compresses a training dataset into a compact set of synthetic examples that encode the learning signal of the full dataset.

Introduced by Wang et al. (2018) as a bi-level optimization problem, dataset distillation has grown into an active field driven by applications in privacy-preserving ML, efficient neural architecture search, continual learning without catastrophic forgetting, and understanding what information training data encodes.

## Formulation

Let $\mathcal{T} = \{(x_i, y_i)\}_{i=1}^N$ be the full training dataset and $\mathcal{S} = \{(\tilde{x}_j, \tilde{y}_j)\}_{j=1}^M$ be the synthetic dataset with $M \ll N$.

The goal is to find $\mathcal{S}^*$ such that:

$$\mathcal{S}^* = \arg\min_\mathcal{S} \; \mathbb{E}_{\theta_0} \left[ \mathcal{L}(\theta_{\mathcal{S}}(\mathcal{S}), \mathcal{T}_\text{val}) \right]$$

where $\theta_\mathcal{S}(\mathcal{S})$ are the parameters learned by training on $\mathcal{S}$ from initialization $\theta_0$, and $\mathcal{L}$ is the validation loss on real data.

This is a **bilevel optimization**: the outer objective is validation performance; the inner objective is the training procedure on $\mathcal{S}$. Differentiating through the inner training loop (unrolled gradient computation) is the core computational challenge.

## Gradient Matching (DD / DC)

Wang et al. (2018) propose **dataset distillation via gradient matching**: the synthetic dataset is optimized such that one gradient step on $\mathcal{S}$ produces the same parameter update as many steps on $\mathcal{T}$:

$$\min_\mathcal{S} \left\| \nabla_\theta \mathcal{L}(\theta; \mathcal{S}) - \nabla_\theta \mathcal{L}(\theta; \mathcal{T}) \right\|^2$$

**Dataset Condensation (DC)** (Zhao et al., 2021) extends this with class-conditional gradient matching and a more scalable approximation:

1. For each class, sample a random subset of real examples.
1. Optimize synthetic images to minimize the cosine distance between gradients computed on synthetic data and real data.
1. Repeat across multiple random network initializations to ensure generalization across architectures.

DC achieves 1 image per class (IPC=1) on CIFAR-10 with $\sim$44% accuracy — remarkably close to the $\sim$84% achievable with the full 50k training images.

## Distribution Matching

**Distribution matching** (DM, Zhao & Bilen, 2021) sidesteps gradient computation by matching feature distributions between real and synthetic data in a learned embedding space:

$$\min_\mathcal{S} \sum_{c} \left\| \frac{1}{|\mathcal{T}_c|} \sum_{x \in \mathcal{T}_c} \phi(x) - \frac{1}{|\mathcal{S}_c|} \sum_{\tilde{x} \in \mathcal{S}_c} \phi(\tilde{x}) \right\|^2$$

where $\phi$ is a feature extractor and $\mathcal{T}_c, \mathcal{S}_c$ are real/synthetic examples of class $c$.

This is much cheaper than gradient matching (no unrolled differentiation) and scales to larger datasets and higher resolutions.

## Kernel Inducing Points (KIP)

**KIP** (Kernel Inducing Points, Nguyen et al., 2021) leverages Neural Tangent Kernel (NTK) theory: in the infinite-width, lazy training regime, a neural network is equivalent to a kernel regression with the NTK kernel $K$. Dataset distillation becomes finding support points that approximate the kernel regression solution on the full dataset:

$$\min_\mathcal{S} \mathcal{L}(\hat{f}_\mathcal{S}, \mathcal{T}_\text{val})$$

where $\hat{f}_\mathcal{S}$ is the kernel regression predictor using $\mathcal{S}$ as support. This is a convex optimization in the label space, enabling exact gradients w.r.t. $\mathcal{S}$.

**FRePo** (Feature Regularized Points, Zhou et al., 2022) improves on KIP by using a more expressive feature extractor and regularizing synthetic data to stay close to the real data manifold — achieving state-of-the-art at the time at IPC=1,10,50 on CIFAR-10 and CIFAR-100.

## Matching Training Trajectories (MTT)

**MTT** (Cazenavette et al., 2022) proposes a different objective: instead of matching gradients or distributions, match the **training trajectories** of a model trained on real vs. synthetic data across multiple steps:

$$\min_\mathcal{S} \sum_t \left\| \theta_t^\mathcal{S} - \theta_t^\mathcal{T} \right\|^2$$

Pre-recorded expert training trajectories $\{\theta_t^\mathcal{T}\}$ from real data are stored. Synthetic data is optimized so that short training segments on $\mathcal{S}$ match the corresponding expert trajectory segments.

MTT achieves substantially higher performance than gradient matching at the same IPC budget — especially at higher IPC values (IPC=50) — because trajectory matching captures multi-step learning dynamics, not just single-step gradient alignment.

## Factorization-Based Methods

**HaBa** (Liu et al., 2022) factorizes synthetic images into **bases and hallucination scores**: instead of storing full images, store a compact basis $B \in \mathbb{R}^{k \times d}$ and coefficients $A \in \mathbb{R}^{M \times k}$, so $\tilde{X} = A \cdot B$. This reduces storage and enables optimization in a lower-dimensional space.

**IDC** (Kim et al., 2022) further reduces storage by encoding each synthetic image as a low-resolution real image plus a high-frequency residual, leveraging the observation that distilled images lie near real image manifolds.

## Dataset Distillation at Scale

Scaling dataset distillation to ImageNet-scale has been challenging due to the high resolution ($224 \times 224$) and large class count (1000 classes). Approaches include:

- **Soft label distillation**: use probability distributions over classes (not one-hot labels) as targets, encoding inter-class similarity information in fewer synthetic images.
- **Augmentation-aware distillation**: distill with DiffAug applied consistently to both real and synthetic data, matching augmentation-transformed distributions.
- **Class-subset distillation**: apply distillation per 100-class subset, assembling the full distilled dataset from class-wise solutions.

**SRe$^2$L** (Yin et al., 2023) distills ImageNet by recovering synthetic images from batch normalization statistics stored during training — achieving competitive performance at IPC=50 with 3-4 hours of optimization on 8 GPUs.

## Applications

### Continual Learning

Dataset distillation provides a principled memory replay mechanism for continual learning: instead of storing raw examples from past tasks (memory-intensive), store a distilled set of synthetic images per task. The distilled set is much smaller yet preserves sufficient information to prevent catastrophic forgetting when revisited during future task training.

### Neural Architecture Search (NAS)

NAS requires evaluating many architectures — each requiring full training runs on large datasets. Distilling the dataset to IPC=1 or IPC=10 reduces training time per architecture evaluation by 100-500×, enabling more architectures to be explored in the same compute budget. Distillation-based NAS has shown strong correlation with full-dataset evaluation.

### Federated Learning

In federated learning, clients can share distilled synthetic datasets instead of raw private data — providing strong privacy guarantees (synthetic data doesn't directly reveal individual records) while still enabling useful model updates at the server.

### Benchmarking Data Efficiency

Distilled datasets serve as compact benchmarks: a model that achieves high accuracy on a distilled dataset with IPC=1 must be highly data-efficient — useful for studying what makes architectures and training procedures effective under minimal data.

## Summary

Dataset distillation synthesizes a small surrogate training set that preserves the learning signal of a much larger real dataset. Gradient matching (DC), distribution matching (DM), kernel methods (KIP, FRePo), and trajectory matching (MTT) offer increasingly powerful approaches, each trading off computational cost against distillation quality. Factorization methods reduce storage overhead, while recent work scales distillation to ImageNet resolution via soft labels and BN statistics recovery. Applications span continual learning (compact task memory), NAS (fast architecture evaluation), federated learning (privacy-preserving data sharing), and data-efficient benchmarking — making dataset distillation a versatile tool at the intersection of efficiency, privacy, and understanding of neural network training.
