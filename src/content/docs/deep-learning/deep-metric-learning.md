---
title: Deep Metric Learning
description: Master deep metric learning — the framework for training neural networks to produce geometrically meaningful embeddings — covering contrastive loss, triplet loss, hard mining strategies, angular margin losses (ArcFace, CosFace), prototypical networks, and applications in face recognition, image retrieval, and few-shot learning.
---

Standard classification models assign inputs to fixed categories. But many real-world problems require measuring **similarity** between inputs rather than assigning them to a closed set of classes: finding duplicate product listings, verifying whether two face photos show the same person, or classifying novel species never seen during training. **Deep metric learning** addresses this by training neural networks to produce embedding vectors where geometric distance reflects semantic similarity — close vectors for similar inputs, distant vectors for dissimilar ones.

## The Metric Learning Objective

Given a neural encoder $f_\theta: \mathcal{X} \to \mathbb{R}^d$, metric learning optimizes $\theta$ so that the learned distance:

$$D_\theta(\mathbf{x}_i, \mathbf{x}_j) = \|f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j)\|_2$$

satisfies:

- $D_\theta(\mathbf{x}_i, \mathbf{x}_j) \ll \varepsilon$ when $\mathbf{x}_i$ and $\mathbf{x}_j$ belong to the same class
- $D_\theta(\mathbf{x}_i, \mathbf{x}_j) > m$ when they belong to different classes (margin $m > 0$)

Embeddings are typically L2-normalized to a unit hypersphere: $f_\theta(\mathbf{x}) / \|f_\theta(\mathbf{x})\|_2 \in \mathbb{S}^{d-1}$.

## Contrastive Loss

Contrastive loss (Hadsell et al., 2006) operates on **pairs** $(x_i, x_j)$ labeled as positive ($y=1$, same class) or negative ($y=0$, different class):

$$\mathcal{L}_{\text{contrastive}} = y \cdot D^2 + (1 - y) \cdot \max(m - D, 0)^2$$

where $D = D_\theta(\mathbf{x}_i, \mathbf{x}_j)$. The loss pulls positive pairs together and pushes negative pairs apart up to margin $m$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # emb1, emb2: (B, d) — L2-normalized embeddings
        # labels: (B,) — 1 for same class, 0 for different class
        distances = F.pairwise_distance(emb1, emb2)
        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        return (positive_loss + negative_loss).mean()
```

A key weakness of contrastive loss is that pair construction is quadratic in dataset size, and random negative pairs are usually easy — providing weak gradients.

## Triplet Loss

Triplet loss (Schroff et al., 2015 — FaceNet) operates on **triplets** $(a, p, n)$: an anchor sample $a$, a positive sample $p$ (same class as $a$), and a negative sample $n$ (different class):

$$\mathcal{L}_{\text{triplet}} = \max\!\left(D(a, p) - D(a, n) + \alpha, \, 0\right)$$

The loss drives $D(a, p) + \alpha < D(a, n)$: the positive must be at least margin $\alpha$ closer than the negative.

```python
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()
```

## Hard Mining

Random triplets are nearly always easy (the negative is already far away), producing near-zero loss and no learning signal. **Hard mining** selects the most informative triplets from a mini-batch.

### Batch Hard Mining

For each anchor in a mini-batch, select:

- **Hardest positive**: the positive with the largest distance to the anchor
- **Hardest negative**: the negative with the smallest distance to the anchor

```python
def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    # embeddings: (B, d), labels: (B,)
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)   # (B, B)

    # Masks for same/different class pairs
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
    diff_class = ~same_class
    diag_mask = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    same_class = same_class & ~diag_mask   # exclude self-pairs

    # Hardest positive: max distance among same-class pairs
    hardest_pos = (pairwise_dist * same_class.float()).max(dim=1).values

    # Hardest negative: min distance among different-class pairs
    max_dist = pairwise_dist.max()
    hardest_neg = (pairwise_dist + max_dist * (~diff_class).float()).min(dim=1).values

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()
```

Batch hard mining requires batches structured so each class has multiple samples — a **balanced sampler** that draws $P$ classes × $K$ samples per class is standard (e.g., $P=16$, $K=4$).

### Semi-Hard Mining

Semi-hard negatives are closer than the margin but farther than the positive: $D(a,p) < D(a,n) < D(a,p) + \alpha$. These provide a gradient signal without the instability of very hard negatives.

## Angular Margin Losses

Face recognition research produced a family of **angular margin** losses that operate on the hypersphere rather than Euclidean space. These consistently outperform triplet loss on large-scale recognition benchmarks.

### SphereFace, CosFace, ArcFace

All three modify the classification cross-entropy by introducing an angular margin $m$ to the target class:

| Method | Modified logit for class $y$ |
| --- | --- |
| SphereFace | $s \cdot \cos(m\,\theta_y)$ |
| CosFace | $s \cdot (\cos\theta_y - m)$ |
| ArcFace | $s \cdot \cos(\theta_y + m)$ |

where $\theta_y$ is the angle between the L2-normalized embedding and the $y$-th class weight vector, and $s$ is a scaling factor.

**ArcFace** (Deng et al., 2019) is the most widely adopted:

```python
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute fixed values
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.threshold = torch.cos(torch.tensor(torch.pi - m))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and weight vectors to unit sphere
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(self.weight, dim=1)

        # Cosine similarity: (B, num_classes)
        cosine = torch.mm(emb_norm, w_norm.t())

        # ArcFace: add angular margin to target class
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        cos_theta_plus_m = cosine * self.cos_m - sine * self.sin_m

        # Only apply margin to target class; use cosine for others
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * cos_theta_plus_m + (1.0 - one_hot) * cosine
        logits = logits * self.s

        return F.cross_entropy(logits, labels)
```

ArcFace's key property: adding a fixed angular margin $m$ to $\theta_y$ requires the network to produce embeddings that are $m$ radians tighter around class centers — creating a uniform geodesic gap between all class boundaries on the hypersphere.

## Prototypical Networks

Prototypical Networks (Snell et al., 2017) use metric learning for **few-shot classification**. During each episode, the model computes a class prototype — the mean embedding of the support examples for each class — and classifies query examples by nearest prototype:

$$p(y = k \mid \mathbf{x}) = \frac{\exp\!\left(-D(f_\theta(\mathbf{x}), \mathbf{c}_k)\right)}{\sum_{k'} \exp\!\left(-D(f_\theta(\mathbf{x}), \mathbf{c}_{k'})\right)}$$

where $\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(\mathbf{x}_i, y_i) \in S_k} f_\theta(\mathbf{x}_i)$ is the prototype of class $k$.

```python
def prototypical_loss(
    support_embeddings: torch.Tensor,   # (n_way * n_shot, d)
    query_embeddings: torch.Tensor,     # (n_way * n_query, d)
    n_way: int,
    n_shot: int,
) -> tuple[torch.Tensor, float]:
    # Compute class prototypes
    support = support_embeddings.view(n_way, n_shot, -1)
    prototypes = support.mean(dim=1)   # (n_way, d)

    # Negative squared Euclidean distance to each prototype
    dists = torch.cdist(query_embeddings, prototypes).pow(2)   # (n_query_total, n_way)
    log_probs = F.log_softmax(-dists, dim=1)

    # Labels: query i belongs to class i // n_query
    n_query = query_embeddings.shape[0] // n_way
    labels = torch.arange(n_way, device=query_embeddings.device).repeat_interleave(n_query)

    loss = F.nll_loss(log_probs, labels)
    accuracy = (log_probs.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy
```

## Unified Embedding Training Pipeline

```python
from torch.utils.data import DataLoader, Sampler
import random


class BalancedBatchSampler(Sampler):
    """Samples P classes × K instances per batch."""

    def __init__(self, labels, n_classes: int = 16, n_per_class: int = 4):
        self.labels = labels
        self.n_classes = n_classes
        self.n_per_class = n_per_class

        self.class_indices = {}
        for idx, label in enumerate(labels):
            self.class_indices.setdefault(label, []).append(idx)

    def __iter__(self):
        classes = random.sample(list(self.class_indices.keys()), self.n_classes)
        batch = []
        for cls in classes:
            indices = random.choices(self.class_indices[cls], k=self.n_per_class)
            batch.extend(indices)
        yield batch

    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_per_class)
```

## Applications

### Face Recognition

Large-scale face recognition (ArcFace on MS-Celeb-1M) produces embeddings that generalize to millions of identities not seen during training. Verification threshold is set on a validation set to balance false accept/reject rates.

### Image and Product Search

E-commerce search engines embed product images and query images into the same space. Approximate nearest neighbor search (FAISS, ScaNN) retrieves visually similar products in milliseconds across catalogs of hundreds of millions of items.

### Person Re-Identification

Surveillance cameras across different viewpoints capture the same person under different lighting, angles, and occlusion. Metric learning models identify that two crops from different cameras show the same individual, enabling tracking across a camera network.

### Medical Imaging

Few-shot learning with metric learning enables rare disease classification from a handful of labeled examples per condition — critical in radiology where annotated cases for uncommon conditions are scarce.

## Summary

Deep metric learning trains encoders that map inputs to a geometrically structured embedding space:

- **Contrastive and triplet loss** provide the foundational pairwise/triplet objectives, with hard mining strategies essential for efficiency
- **ArcFace and angular margin losses** set the state of the art for large-scale recognition by enforcing uniform angular margins between classes on the unit hypersphere
- **Prototypical networks** extend metric learning to few-shot classification through class prototype computation
- **Balanced samplers** ensure mini-batches contain multiple instances per class — a prerequisite for informative loss computation

The key insight shared across all these methods is the same: learn a representation space where the geometry of distances mirrors the structure of semantic similarity, enabling zero-shot generalization to new classes and efficient large-scale retrieval.
