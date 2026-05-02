---
title: Dimensionality Reduction
description: Learn dimensionality reduction — the family of techniques that project high-dimensional data into lower-dimensional representations while preserving structure. Covers PCA with full derivation, kernel PCA for nonlinear structure, t-SNE for cluster visualization, UMAP for scalable topology-preserving embeddings, and autoencoders for deep representation learning.
---

**Dimensionality reduction** transforms data from a high-dimensional space to a lower-dimensional one while preserving as much meaningful structure as possible. High-dimensional data is ubiquitous in machine learning — images are millions of pixels, gene expression arrays measure 20,000+ genes, word embeddings sit in 768+ dimensions — yet the true degrees of freedom are often far smaller. Dimensionality reduction exploits this **intrinsic dimensionality** to enable visualization, noise reduction, compression, and faster downstream learning.

The fundamental challenge is deciding *which* structure to preserve: global covariance (PCA), local distances (t-SNE), topological connectivity (UMAP), or reconstruction fidelity (autoencoders). No single method dominates; the right choice depends on the task.

## Principal Component Analysis (PCA)

PCA finds the orthogonal directions of maximum variance in the data — the axes along which the data spreads most. Projecting onto the top $k$ of these **principal components** gives the best rank-$k$ linear approximation to the data in terms of mean squared reconstruction error.

### Derivation

Given centered data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ (zero mean), PCA solves:

$$\mathbf{W}^* = \arg\max_{\mathbf{W} \in \mathbb{R}^{d \times k},\ \mathbf{W}^\top \mathbf{W} = \mathbf{I}} \text{Var}(\mathbf{X}\mathbf{W}) = \arg\max \text{tr}(\mathbf{W}^\top \mathbf{X}^\top \mathbf{X} \mathbf{W})$$

The solution is the top $k$ eigenvectors of the covariance matrix $\mathbf{C} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}$, equivalently the right singular vectors of $\mathbf{X}$ via SVD:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

The principal components are the columns of $\mathbf{V}$; the projected data is $\mathbf{Z} = \mathbf{X}\mathbf{V}_k = \mathbf{U}_k \mathbf{\Sigma}_k$.

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

def pca_from_scratch(X: np.ndarray, n_components: int) -> dict:
    """
    PCA via SVD — equivalent to sklearn's PCA but explicit.
    
    X: (n_samples, n_features) — will be centered
    Returns: projected data, explained variance ratios, components
    """
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # SVD: X = U Σ V^T
    # Use full_matrices=False for economy SVD (faster for tall matrices)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Top k principal components (rows of Vt)
    components = Vt[:n_components]           # (k, d) — each row is a PC
    
    # Project data
    Z = X_centered @ components.T           # (n, k)
    
    # Explained variance
    total_variance = (S ** 2).sum()
    explained_variance = (S[:n_components] ** 2) / (len(X) - 1)
    explained_variance_ratio = (S[:n_components] ** 2) / total_variance
    
    return {
        "Z": Z,                              # projected data
        "components": components,            # principal components
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": np.cumsum(explained_variance_ratio)
    }


# PCA on handwritten digits (64-dimensional → 2D visualization)
digits = load_digits()
X = StandardScaler().fit_transform(digits.data.astype(float))

result = pca_from_scratch(X, n_components=10)
print(f"Top 2 PCs explain: {result['cumulative_variance'][1]:.1%} of variance")
print(f"Top 10 PCs explain: {result['cumulative_variance'][9]:.1%} of variance")


def choose_n_components(explained_ratios: np.ndarray,
                        threshold: float = 0.95) -> int:
    """Return minimum number of PCs to explain `threshold` of variance."""
    cumsum = np.cumsum(explained_ratios)
    return int(np.searchsorted(cumsum, threshold) + 1)
```

### Limitations of PCA

PCA captures only **linear** structure. If data lies on a curved manifold (e.g., a Swiss roll, concentric circles), PCA will project points that are far apart on the manifold but close in 3D space into nearby low-dimensional positions — destroying the true structure.

## Kernel PCA

Kernel PCA applies PCA in a high-dimensional feature space $\phi(\mathbf{x})$ implicitly defined by a kernel function $k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$. This allows capturing nonlinear structure without explicitly computing the high-dimensional features.

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
import numpy as np

# Two concentric circles — PCA fails; kernel PCA succeeds
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)

# Linear PCA: projects both circles onto a line — indistinguishable
from sklearn.decomposition import PCA
Z_pca = PCA(n_components=2).fit_transform(X)

# Kernel PCA with RBF kernel: separates circles in 2D
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
Z_kpca = kpca.fit_transform(X)

# Other useful kernels:
# kernel="poly", degree=3      — polynomial kernel
# kernel="cosine"               — cosine similarity (good for text)
# kernel="sigmoid"              — sigmoid/tanh kernel

# Note: kernel PCA requires storing the full kernel matrix O(n^2)
# — expensive for large datasets
```

## t-SNE: Visualization of Cluster Structure

**t-Distributed Stochastic Neighbor Embedding** (van der Maaten & Hinton, 2008) is a nonlinear dimensionality reduction technique optimized for **visualizing cluster structure** in 2D or 3D. It preserves local neighborhood structure: points that are nearby in high dimensions are mapped nearby in low dimensions.

### How t-SNE Works

1. **High-dimensional similarities**: Compute pairwise similarities as conditional probabilities:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

Symmetrize: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

1. **Low-dimensional similarities**: Use a heavy-tailed Student-t distribution (t-distribution with 1 degree of freedom) to prevent the "crowding problem":

$$q_{ij} = \frac{(1 + \|z_i - z_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|z_k - z_l\|^2)^{-1}}$$

1. **Optimize** the KL divergence $\text{KL}(P \| Q)$ between the two similarity distributions via gradient descent on the low-dimensional positions $\{z_i\}$.

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np

def visualize_tsne(X: np.ndarray, labels: np.ndarray,
                   perplexity: float = 30.0) -> np.ndarray:
    """
    Project high-dimensional X to 2D with t-SNE.
    
    Key hyperparameters:
    - perplexity (5–50): effective number of neighbors to preserve.
      Lower = more focus on local structure. Typical: 30.
    - n_iter (1000+): more iterations = better convergence.
    - learning_rate ("auto" or 10–1000): too high → diffuse, too low → compressed.
    """
    X_scaled = StandardScaler().fit_transform(X)
    
    # Optional: reduce to 50 dims with PCA first (recommended for d >> 50)
    from sklearn.decomposition import PCA
    if X.shape[1] > 50:
        X_scaled = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        n_iter=1000,
        random_state=42
    )
    return tsne.fit_transform(X_scaled)

# t-SNE caveats:
# 1. Distances between clusters are NOT meaningful — only local structure is.
# 2. Cluster sizes in t-SNE do not reflect true cluster sizes.
# 3. t-SNE is stochastic — different runs produce different layouts.
# 4. Does not support out-of-sample projection (must refit for new points).
# 5. O(n^2) memory/time; use approximate methods (Barnes-Hut) for n > 10,000.
```

## UMAP: Scalable Topology-Preserving Embeddings

**Uniform Manifold Approximation and Projection** (McInnes et al., 2018) addresses t-SNE's key weaknesses: it is faster (O(n log n)), better preserves global structure, and supports out-of-sample projection.

UMAP is grounded in topological data analysis: it constructs a **fuzzy topological representation** (a weighted graph) of the high-dimensional data and optimizes a low-dimensional representation to have the same fuzzy topology.

```python
import umap
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# Basic UMAP projection
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,        # balances local vs. global structure (5–50)
    min_dist=0.1,          # minimum distance between points in embedding (0.0–0.99)
    metric="euclidean",    # or "cosine" for text embeddings, "manhattan", etc.
    random_state=42
)
Z = reducer.fit_transform(X)

# Out-of-sample projection (unlike t-SNE!)
X_new = np.random.randn(10, 64)
Z_new = reducer.transform(X_new)

# Supervised UMAP: use labels to guide separation
reducer_supervised = umap.UMAP(n_neighbors=15, min_dist=0.1)
Z_supervised = reducer_supervised.fit_transform(X, y=y)  # clusters much cleaner!

# UMAP for dimensionality reduction (not just visualization)
# e.g., reduce 768-dim BERT embeddings to 50 dims before clustering
reducer_50d = umap.UMAP(n_components=50, n_neighbors=30, min_dist=0.0)
X_50d = reducer_50d.fit_transform(X)
```

## Autoencoders for Nonlinear Dimensionality Reduction

Deep autoencoders learn a nonlinear embedding by training an encoder-decoder pair to reconstruct the input:

```python
import torch
import torch.nn as nn

class AutoencoderDR(nn.Module):
    """
    Nonlinear dimensionality reduction via a deep autoencoder.
    The encoder output (bottleneck) is the low-dimensional embedding.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        # Encoder: input_dim → hidden_dims → latent_dim
        enc_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        enc_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent_dim → hidden_dims (reversed) → input_dim
        dec_layers = []
        dec_dims = [latent_dim] + list(reversed(hidden_dims))
        for i in range(len(dec_dims) - 1):
            dec_layers += [nn.Linear(dec_dims[i], dec_dims[i+1]), nn.ReLU()]
        dec_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def train_autoencoder(X: np.ndarray, latent_dim: int = 2,
                       n_epochs: int = 100) -> AutoencoderDR:
    model = AutoencoderDR(
        input_dim=X.shape[1],
        hidden_dims=[256, 128],
        latent_dim=latent_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    for epoch in range(n_epochs):
        x_hat, _ = model(X_tensor)
        loss = nn.functional.mse_loss(x_hat, X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, Z = model(X_tensor)
    return model
```

## Choosing a Method

| Method | Preserves | Speed | Out-of-sample | Best for |
|---|---|---|---|---|
| **PCA** | Global variance | Fast | Yes | Preprocessing, noise reduction |
| **Kernel PCA** | Nonlinear global | Moderate | Yes | Nonlinear data, moderate N |
| **t-SNE** | Local neighborhoods | Slow (O(n²)) | No | 2D/3D visualization |
| **UMAP** | Local + global topology | Fast (O(n log n)) | Yes | Visualization + preprocessing |
| **Autoencoder** | Reconstruction | Slow (train) | Yes | Complex data, task-specific |
| **LDA** | Class separability | Fast | Yes | Supervised classification |

A practical workflow: use PCA to reduce to 50–100 dimensions, then apply UMAP or t-SNE for 2D visualization. Use UMAP (not t-SNE) when you need the 2D coordinates for downstream tasks like clustering, since UMAP better preserves global relationships between clusters.
