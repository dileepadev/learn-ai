---
title: Multiple Instance Learning
description: Learn about Multiple Instance Learning (MIL) — a weakly supervised learning paradigm where labels are provided for bags of instances rather than individual instances — with applications in medical imaging, drug discovery, and document classification.
---

Multiple Instance Learning (MIL) is a weakly supervised learning paradigm that addresses settings where labels are available at the **bag level** rather than the **instance level**. A bag is a collection of instances, and the learning signal comes only from the bag label — not from which specific instances inside the bag triggered the label. This models many real-world scenarios where fine-grained annotation is expensive or impossible.

## The Multiple Instance Assumption

The classical MIL assumption (Dietterich et al., 1997) for binary classification states:

- A **positive bag** contains **at least one** positive instance
- A **negative bag** contains **no** positive instances

Formally, a bag $B_i = \{x_{i,1}, x_{i,2}, \ldots, x_{i,n_i}\}$ has label:

$$Y_i = \max_j y_{i,j}$$

where $y_{i,j} \in \{0, 1\}$ is the unknown instance-level label. The learner only observes $Y_i$.

This OR-of-instances assumption captures the intuition that a cancer slide is positive if **any** region contains malignant cells, even if most patches are benign.

## Motivating Applications

### Computational Pathology

Whole-slide images (WSIs) in pathology can contain billions of pixels. Slide-level labels (cancer / no cancer) are available from pathology reports, but annotating which specific regions contain cancer is extremely expensive. MIL treats each slide as a bag and each patch (e.g., $256 \times 256$ pixels) as an instance.

### Drug Discovery

A drug molecule may bind to a target protein in multiple conformations. The molecule (bag) is labeled active if **any** of its low-energy conformations (instances) binds. MIL naturally captures this: labels are provided per molecule, not per conformation.

### Document Classification

A document (bag) is labeled positive if any sentence (instance) is relevant to a topic. Labeling every sentence is impractical; document-level labels are cheap.

### Point Cloud and 3D Recognition

A 3D point cloud (bag) can be labeled by object category, treating each point or local region as an instance. MIL enables learning without dense per-point annotation.

## Classical MIL Algorithms

### mi-SVM and MI-SVM

**Andrews et al. (2003)** adapted SVM for MIL. Two variants:

- **mi-SVM**: Assigns instance labels by solving a mixed-integer program, using SVM margin maximization
- **MI-SVM**: Defines a bag-level margin and maximizes it directly

Both solve NP-hard combinatorial problems, requiring heuristic relaxations in practice.

### Citation-kNN

**Wang & Zucker (2000)** extended $k$-nearest neighbors to MIL using bag-level distance metrics — Hausdorff distance or average distance between instances across bags.

### Diverse Density

**Maron & Lozano-Pérez (1998)** introduced Diverse Density (DD), seeking a concept point in instance feature space that:

1. Is **near** instances from many positive bags (density)
1. Is **far** from all instances in negative bags (diversity)

$$DD(t) = \prod_i \left(1 - \prod_j (1 - P(t | x_{i,j}))\right) \cdot \prod_k \prod_j (1 - P(t | x_{k,j}))$$

DD identifies candidate positive instances, forming the basis for many subsequent methods.

## Deep MIL: Attention-Based Aggregation

Modern MIL leverages deep learning with a two-stage approach:

1. **Feature extraction**: A CNN/ViT encodes each instance $x_{i,j}$ into a representation $h_{i,j} = f_\theta(x_{i,j})$
1. **Aggregation**: A pooling operator combines all instance representations into a bag embedding $z_i = \text{pool}(\{h_{i,j}\})$
1. **Bag-level prediction**: A classifier $g_\phi(z_i)$ predicts the bag label

The pooling operator is the critical design choice.

### Max Pooling MIL

$$z_i = \max_j h_{i,j}$$

Directly implements the classical MIL assumption: the most discriminative instance dominates. Simple but discards all information from non-key instances.

### Mean Pooling MIL

$$z_i = \frac{1}{n_i} \sum_j h_{i,j}$$

Assumes all instances contribute equally. Works when labels are distributed across the bag (e.g., multi-label bags) but fails when only rare instances are relevant.

### Attention MIL (ABMIL)

**Ilse et al. (2018)** introduced attention-based MIL, which learns a **weighted** aggregation:

$$z_i = \sum_j a_{i,j} h_{i,j}$$

where attention weights are:

$$a_{i,j} = \frac{\exp\left(w^T \tanh(V h_{i,j})\right)}{\sum_k \exp\left(w^T \tanh(V h_{i,k})\right)}$$

The attention weights $a_{i,j}$ are interpretable: high-weight instances are the most relevant contributors to the bag prediction, providing a form of **weak supervision localization**.

**Gated attention** adds a gate mechanism:

$$a_{i,j} = \frac{\exp\left(w^T (\tanh(V h_{i,j}) \odot \text{sigmoid}(U h_{i,j}))\right)}{\sum_k \exp(\ldots)}$$

which provides better selectivity and handles both positive and negative evidence.

## Transformer-Based MIL

With transformers, the set of instance embeddings is treated as a sequence, and self-attention models pairwise relationships between instances within a bag:

$$Z = \text{Transformer}(\{h_{i,j}\}_{j=1}^{n_i})$$

$$z_i = \text{pool}(Z)$$

This captures **co-occurrence patterns** between instances — e.g., in pathology, the spatial co-occurrence of multiple cell types might signal a diagnosis that no single patch reveals alone.

**TransMIL (Shao et al., 2021)** and **DSMIL (Li et al., 2021)** are notable examples combining multi-scale feature pyramids with transformer aggregation for pathology WSI classification.

## Training with Pseudo-Labels

A common improvement: alternate between assigning pseudo-labels to instances and updating the model.

1. **Initialization**: Train with bag labels using attention MIL
1. **Instance pseudo-labeling**: Assign instance labels based on attention scores or predicted instance probabilities
1. **Instance-level fine-tuning**: Fine-tune the feature extractor with pseudo-labeled instances
1. **Repeat**: Update pseudo-labels with the improved model

This iterative approach progressively improves instance-level features even without explicit annotation.

## Graph MIL

When instances are not i.i.d. but have spatial structure (e.g., tissue histology with spatial context), Graph Neural Networks can model the bag as a graph:

- **Nodes**: Instance embeddings $h_{i,j}$
- **Edges**: Spatial proximity or feature similarity between instances
- **Aggregation**: GNN message passing followed by global pooling

**PatchGCN (Chen et al., 2021)** and related methods use spatial graphs over WSI patches, incorporating neighborhood context that pure MIL aggregation misses.

## Beyond Binary: Ordinal and Multi-Label MIL

### Ordinal MIL

Cancer grading (e.g., Gleason score 3–5) requires ordinal labels. **Ordinal MIL** extends the classical assumption to: a bag's grade equals the maximum grade among its instances, and ordinal ranking loss replaces binary cross-entropy.

### Multi-Label MIL

When multiple labels are possible per bag (e.g., multiple diseases visible in a slide), per-label attention mechanisms or label-conditioned aggregation can be used, with each label having an independent classifier.

## Interpretability and Instance Localization

A key advantage of MIL over fully supervised models is that interpretability is built into the learning problem: **identifying which instances drive the bag-level prediction is part of the task.**

Attention scores in ABMIL provide soft localization: patches with high attention are candidate regions explaining the prediction. In pathology, this corresponds to localizing suspicious tissue regions without tile-level annotations.

**Limitations**: Attention MIL attention scores are not guaranteed to identify the truly predictive instances; they reflect what the trained model relies on, which may include spurious correlations. Calibration and uncertainty estimation remain open problems.

## Practical Considerations

### Instance Feature Extraction

Pre-trained features (from ImageNet CNNs or pathology-specific models like CONCH, UNI, PLIP) are commonly used as frozen feature extractors. Fine-tuning end-to-end is expensive due to bag sizes (thousands of patches per slide) and often destabilizes training.

### Bag Size Variability

Bags can contain anywhere from a handful to tens of thousands of instances. Attention mechanisms handle variable-size bags naturally. For transformers, memory scales quadratically with instances — requiring sparse attention or instance sub-sampling.

### Class Imbalance

In pathology, most patches in a positive slide are benign. Instance-level class imbalance is severe. Sampling strategies (top-$k$ hard instances, attention-guided sampling) help focus training on discriminative regions.

## Comparison with Related Paradigms

| Setting | Labels available | Key assumption |
| --- | --- | --- |
| Fully supervised | Per-instance labels | All instances labeled |
| Multiple instance learning | Per-bag labels only | Bag label = $\max$ instance label |
| Semi-supervised | Some instances labeled | Labeled + unlabeled instances |
| Self-supervised | No labels | Learn from data structure |
| Weakly supervised (general) | Noisy or coarse labels | Various noise models |

MIL occupies a unique niche: richer than fully unsupervised, but more scalable than fully supervised annotation.

## Summary

Multiple Instance Learning is a natural framework for problems where fine-grained labels are unavailable but bag-level supervision is accessible. The key design decisions are:

- **Transition matrix / pooling operator**: max, mean, attention, or graph-based aggregation
- **Feature extraction**: frozen vs. fine-tuned encoders
- **Training objective**: bag-level cross-entropy, ELBO, or auxiliary instance-level pseudo-labels

Attention-based MIL (ABMIL) is the dominant modern approach, combining learned weighted aggregation with interpretable instance localization. Its adoption in computational pathology has enabled large-scale analysis of whole-slide images from clinically available diagnostic labels alone — without the need for expensive region-level annotation.
