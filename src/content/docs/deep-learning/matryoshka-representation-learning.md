---
title: Matryoshka Representation Learning (MRL)
description: Explore Matryoshka Representation Learning (MRL), a technique that trains neural networks to encode information at multiple granularities, allowing adaptive embedding size and massive search savings.
---

**Matryoshka Representation Learning (MRL)** is a representation learning technique developed by researchers at Google and the University of Washington. Inspired by nested Russian Matryoshka dolls, MRL trains embedding models to store information at multiple granularities within a single, high-dimensional vector.

By structuring embeddings such that their early dimensions (e.g., first 64 or 128 elements) represent a highly dense summary of the concept, while later dimensions add finer detail, MRL allows systems to dynamically adjust embedding sizes depending on computational and memory budgets.

---

## The Embedding Bottleneck

In modern vector databases and search engines:
- Storing high-dimensional embeddings (e.g., $d = 1536$ or $3072$ for text-embedding-3) requires massive amounts of memory (RAM/GPU VRAM).
- Computing cosine similarities or Euclidean distances scales linearly with embedding dimension $d$.
- To reduce costs, engineers often use quantization (FP16 $\to$ INT8) or train separate, smaller models. However, training distinct models for every deployment constraint is expensive.

MRL solves this by training **one model** whose embeddings can be sliced at run-time to any desired dimension without a significant loss in accuracy.

---

## How MRL Works

Given a neural network that outputs a high-dimensional vector $z \in \mathbb{R}^D$, we define a set of target dimensions $\mathcal{M} = \{d_1, d_2, \dots, d_K\}$ such that:

$$d_1 < d_2 < \dots < d_K = D$$

For example, $\mathcal{M} = \{64, 128, 256, 512, 1024\}$.

During training, we append a separate classification head (or use a separate contrastive loss component) for each slice of the embedding $z_{1:d}$. The total loss is a weighted sum of the losses calculated across all target dimensions:

$$\mathcal{L}_{\text{MRL}} = \sum_{d \in \mathcal{M}} c_d \cdot \mathcal{L}(z_{1:d})$$

Where:
- $z_{1:d}$ is the sub-vector containing the first $d$ elements of $z$.
- $\mathcal{L}$ is the task-specific loss function (e.g., cross-entropy or InfoNCE).
- $c_d$ is a weight scaling factor (often set to $1/d$ or simply $1.0$).

```
Full Embedding Vector (D = 1024):
[ x1, x2, x3, ... x64 | x65 ... x128 | x129 ... x256 | x257 ... x1024 ]
  \________________/    \__________/    \__________/    \____________/
       Dim 64             Dim 128         Dim 256          Dim 1024
       (Coarse)                                            (Highly Detailed)
```

By optimization, the network learns to compress the most important classification features into the earliest dimensions, leaving finer, nuanced features for the latter dimensions.

---

## Slicing Embeddings in Practice

Once trained, you can reduce embedding sizes by simply slicing the vector. No re-training or post-processing is required:

```python
import numpy as np

# A full Matryoshka embedding (dim = 1024)
embedding_1024 = get_matryoshka_embedding("Deep learning and artificial intelligence")

# Slice down to dim = 128
embedding_128 = embedding_1024[:128]

# Normalize to ensure correct cosine similarity calculations
embedding_128 = embedding_128 / np.linalg.norm(embedding_128)
```

---

## Benefits of MRL

1. **Adaptive Retrieval (Funnel Search):** You can perform coarse search on the first 64 dimensions to filter down to the top-1000 items, and then compute the similarity using the full 1024 dimensions on only those 1000 items. This cuts down vector database operations by up to 14x.
2. **Dynamic Compression:** A single model can serve mobile devices (using 128 dimensions to save bandwidth) and cloud servers (using 1024 dimensions for maximum accuracy).
3. **Negligible Performance Loss:** Slicing a 1024-dimensional Matryoshka embedding down to 256 dimensions often preserves over 95% of the original representation's retrieval performance.

---

## Implementing MRL with Sentence Transformers

The `sentence-transformers` library offers native support for training MRL models via `MatryoshkaLoss`.

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.losses import MatryoshkaLoss
from torch.utils.data import DataLoader

# 1. Load a base model
model = SentenceTransformer("bert-base-uncased")

# 2. Configure base loss (e.g., MultipleNegativesRankingLoss for retrieval)
base_loss = losses.MultipleNegativesRankingLoss(model)

# 3. Wrap in MatryoshkaLoss and specify target dimensions
mrl_loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[64, 128, 256, 512, 768]
)

# 4. Train the model using the wrapped loss
model.fit(
    train_objectives=[(train_dataloader, mrl_loss)],
    epochs=1,
    warmup_steps=100
)
```
