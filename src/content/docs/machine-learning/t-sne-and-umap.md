---
title: t-SNE and UMAP - Visualizing High-Dimensional Data
description: Learn how t-SNE and UMAP embed high-dimensional data in two dimensions, how to tune them, and the common mistakes in interpreting their plots.
---

t-SNE and UMAP are nonlinear dimensionality reduction methods used mainly to visualize embeddings, clusters, and learned representations in 2D or 3D.

## t-SNE

t-distributed Stochastic Neighbor Embedding converts pairwise distances into probabilities:

1. In high dimensions, each point's neighbors get probabilities from a Gaussian whose width is set by **perplexity**
2. In the low-dimensional map, similarities use a heavy-tailed Student-t distribution
3. Points are moved to minimize the KL divergence between the two distributions

The heavy tails let moderately distant points sit farther apart, relieving the "crowding problem" and producing well-separated clusters.

## UMAP

Uniform Manifold Approximation and Projection builds a weighted k-nearest-neighbor graph (a fuzzy topological representation) and optimizes a low-dimensional layout to have a similar graph structure. It is typically faster than t-SNE and tends to preserve more global structure.

## Usage

```python
from sklearn.manifold import TSNE
import umap

X_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)
```

## Key Hyperparameters

| Method | Parameter | Effect |
|--------|-----------|--------|
| t-SNE | `perplexity` | Effective neighborhood size (5-50 typical) |
| UMAP | `n_neighbors` | Local versus global focus |
| UMAP | `min_dist` | How tightly points pack together |

## Interpretation Pitfalls

- **Cluster sizes and distances between clusters are not reliable** in t-SNE; do not read meaning into gaps or relative size
- Different seeds and parameters give different layouts; check stability
- Apparent clusters can be artifacts of low perplexity
- Neither method gives a faithful global distance map
- Use them for exploration, not as evidence; confirm with quantitative analysis in the original space

## Practical Advice

Reduce very high dimensions first with PCA (to 30-50 components) for speed and noise reduction, standardize features where appropriate, and color points by known labels or metadata to sanity-check the structure.
