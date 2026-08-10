---
title: "Persistent Homology in Machine Learning"
description: An accessible introduction to persistent homology — a topological data analysis tool — and its applications in machine learning for shape analysis, neural network topology, graph learning, and data characterization.
---

Machine learning is largely a geometry-and-statistics enterprise. We think about data in terms of distances, densities, manifolds, and probability distributions. But there is another lens through which data can be analyzed: **topology** — the study of shape properties that are preserved under continuous deformation.

**Persistent homology** is the central tool of Topological Data Analysis (TDA), a field that applies algebraic topology to data science and machine learning. It extracts multi-scale shape features from data — connected components, loops, voids, and higher-dimensional holes — that are invisible to purely geometric or statistical approaches.

## Why Topology Adds Something New

Consider two datasets:

- **Dataset A:** 500 points sampled from a circle
- **Dataset B:** 500 points sampled from a figure-8 (two intersecting circles)

These datasets have similar statistical summaries (similar variance, similar nearest-neighbor distance distributions). A standard manifold learning algorithm might embed both as 1D curves. But topologically, they are fundamentally different:

- The circle has one 1-dimensional loop (one independent cycle)
- The figure-8 has two 1-dimensional loops

Persistent homology captures this difference as a topological invariant — a feature that encodes the fundamental "shape" of the data regardless of its coordinate representation.

## Homology: A Brief Introduction

Homology is an algebraic way to count holes of different dimensions:

- **$H_0$ (0-dimensional homology):** Connected components — how many separate clusters/islands?
- **$H_1$ (1-dimensional homology):** 1D holes — how many independent loops or cycles?
- **$H_2$ (2-dimensional homology):** 2D voids — how many enclosed hollow regions?
- **$H_k$ (k-dimensional homology):** k-dimensional holes

For a circle (the boundary of a disk):
- $H_0$ = 1 connected component
- $H_1$ = 1 loop (the circle itself)

For a torus (donut surface):
- $H_0$ = 1 connected component
- $H_1$ = 2 independent loops (around the tube and through the hole)
- $H_2$ = 1 enclosed void

## From Point Clouds to Homology: The Vietoris-Rips Complex

Given $n$ points in some metric space, how do we compute homology? We build a **simplicial complex** — a combinatorial structure made of points (0-simplices), edges (1-simplices), triangles (2-simplices), and higher-dimensional analogues.

The **Vietoris-Rips complex** $\text{VR}(X, \epsilon)$ at scale $\epsilon$ connects points within distance $\epsilon$:
- Add a vertex for each point in $X$
- Add an edge between points $x_i, x_j$ if $d(x_i, x_j) \leq \epsilon$
- Add a triangle for every triple of mutually connected points
- Continue for higher dimensions

The problem: the resulting homology depends entirely on the choice of $\epsilon$. Too small: the complex is disconnected, capturing only isolated points. Too large: everything collapses into a single clump.

## Persistence: Varying Scale Continuously

**Persistent homology** solves the scale-choice problem by computing homology at **all scales simultaneously** and tracking how topological features appear and disappear:

1. Start with $\epsilon = 0$: just the isolated points
2. Gradually increase $\epsilon$
3. At each value, new topological features are **born** (appear) or **die** (disappear as they are filled in)
4. Record each feature's birth time $b$ and death time $d$ as a pair $(b, d)$

The collection of all $(b, d)$ pairs is called the **persistence diagram** (or **barcode** when displayed as intervals).

Features with high persistence (large $d - b$) are topologically significant — they represent genuine shape structure. Features with low persistence are noise.

### The Persistence Barcode

A barcode visualizes the lifespan of each topological feature:

```
Scale ε →  0    0.2   0.4   0.6   0.8   1.0

H_0 (components):
  Component 1: |--------------------------------|  (born at 0, dies at ∞ = the main cluster)
  Component 2: |-----|                            (born at 0, dies at 0.15 = merges into 1)
  Component 3: |---------|                        (born at 0, dies at 0.22 = merges into 1)

H_1 (loops):
  Loop 1:            |-------------|              (born at 0.35, dies at 0.71 = significant)
  Loop 2:                  |--|                   (born at 0.45, dies at 0.52 = noise)
```

The long bar in H_1 reveals a real loop in the data. The short bar is likely noise.

## Computing Persistent Homology in Python

The **Giotto-TDA** and **Ripser** libraries provide efficient persistent homology computation:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from ripser import ripser
from persim import plot_diagrams

# Generate a circle dataset
X, _ = make_circles(n_samples=200, noise=0.05, factor=0.3)

# Compute persistent homology (up to dimension 1)
diagrams = ripser(X, maxdim=1)["dgms"]

# Plot persistence diagrams
plot_diagrams(diagrams, show=True)

# H_0 diagram: birth-death pairs for connected components
h0_diagram = diagrams[0]  # shape (n_components, 2)

# H_1 diagram: birth-death pairs for loops
h1_diagram = diagrams[1]  # shape (n_loops, 2)

# Extract persistence (lifetime) of each feature
h1_persistence = h1_diagram[:, 1] - h1_diagram[:, 0]
print(f"Most persistent loop: {h1_persistence.max():.3f}")
```

```python
# Using Giotto-TDA for ML pipelines
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceEntropy

# Point cloud: (n_samples, n_points, n_dimensions)
point_cloud = X[np.newaxis, :, :]  # Add batch dimension

# Compute persistence
vr = VietorisRipsPersistence(homology_dimensions=[0, 1])
diagrams = vr.fit_transform(point_cloud)

# Convert to fixed-size feature vectors for ML
betti_curve = BettiCurve()
features = betti_curve.fit_transform(diagrams)  # Fixed-length vector

entropy = PersistenceEntropy()
entropy_features = entropy.fit_transform(diagrams)
```

## Topological Features for Machine Learning

To use persistent homology in a machine learning pipeline, we need to convert persistence diagrams to fixed-size vectors (since diagrams have variable numbers of points):

### 1. Persistence Images

The **persistence image** (Adams et al., 2017) converts a persistence diagram to a fixed-size 2D grid by placing Gaussian kernels at each $(b, d)$ point:

$$I(\mathbf{u}) = \sum_{(b,d) \in \text{diagram}} w(b,d) \cdot \phi_\sigma(\mathbf{u} - (b, d-b))$$

Where $w(b,d)$ is a persistence-weighting function and $\phi_\sigma$ is a Gaussian kernel. The resulting image can be flattened into a feature vector for any standard ML classifier.

### 2. Betti Numbers and Betti Curves

The **Betti number** $\beta_k(\epsilon)$ is the count of $k$-dimensional holes at scale $\epsilon$. The **Betti curve** traces $\beta_k$ as $\epsilon$ varies — a compact 1D signature of topological structure at each scale.

### 3. Persistence Entropy

$$E = -\sum_{(b,d)} \frac{d-b}{\mathcal{L}} \log \frac{d-b}{\mathcal{L}}$$

Where $\mathcal{L} = \sum_{(b,d)} (d-b)$ is total persistence. Entropy measures the complexity and uniformity of topological feature lifetimes.

### 4. Topological Vector Summaries

Simple scalar summaries: total persistence, mean persistence, max persistence, number of features above a threshold — can capture meaningful topological signals with minimal computational overhead.

## Applications in Machine Learning

### Shape Analysis and 3D Data

Persistent homology is particularly powerful for 3D point clouds:

- **Medical imaging:** Characterizing tumor shape morphology from CT/MRI scans; topological features outperform purely geometric features for distinguishing malignant from benign lesions in some datasets
- **Materials science:** Characterizing the pore structure of porous materials from micro-CT scans for properties like permeability and thermal conductivity
- **Protein structure:** Topological features of protein binding pockets improve drug binding site prediction

### Neural Network Topology

Persistent homology has been used to analyze neural networks themselves:

- **Weight space topology:** Analyzing the topological structure of weight matrices to study network complexity and correlate it with generalization
- **Activation topology:** Studying the topology of the manifold traced by hidden activations across a dataset — topological properties of activation manifolds correlate with representation quality
- **Loss landscape topology:** Characterizing the connectivity of minima in loss landscapes to understand generalization and optimization dynamics

```python
# Analyzing the topology of neural network hidden representations
import torch
from ripser import ripser

def compute_activation_topology(model, data_loader, layer_name):
    activations = []
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda m, i, o: activations.append(o.detach().cpu().numpy())
    )

    with torch.no_grad():
        for batch in data_loader:
            model(batch)

    hook.remove()
    activation_matrix = np.concatenate(activations, axis=0)

    # Reduce dimensionality before computing persistence
    from sklearn.decomposition import PCA
    activation_pca = PCA(n_components=50).fit_transform(activation_matrix)

    diagrams = ripser(activation_pca, maxdim=1)["dgms"]
    return diagrams
```

### Graph Learning with Persistent Homology

For graph neural networks, persistent homology applied to graph structure provides complementary features to spectral and spatial GNN methods:

- **Graph filtration:** Building a persistence diagram from the graph's edge weights (sorted in order) captures multi-scale connectivity structure
- **Topological graph classification:** Combining standard GNN node/edge features with persistent homology graph features improves classification on molecular property prediction benchmarks

### Time Series Analysis

For time series, the **Takens delay embedding** converts a scalar time series to a point cloud in delay-coordinate space, revealing attractor topology:

```python
def time_delay_embedding(series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    """Convert 1D time series to delay-coordinate point cloud."""
    n = len(series) - (dimension - 1) * delay
    embedded = np.array([series[i:i + dimension * delay:delay] for i in range(n)])
    return embedded  # (n, dimension)

# Chaotic time series (Lorenz, ECG, seismic data) have distinctive attractor topologies
embedding = time_delay_embedding(ecg_signal, dimension=3, delay=10)
diagrams = ripser(embedding, maxdim=2)["dgms"]
```

Persistent homology on delay embeddings distinguishes periodic, quasi-periodic, and chaotic signals — useful for heartbeat classification, seismic event detection, and anomaly detection in sensor data.

## Computational Complexity and Practical Limitations

Persistent homology computation on $n$ points has worst-case complexity $O(n^3)$ in both time and space, making it impractical for very large datasets without approximation:

- **Landmark-based methods:** Subsample $m \ll n$ landmark points, compute persistence on the landmark complex
- **Sparse Rips:** Use approximate nearest-neighbor graphs instead of the full Vietoris-Rips complex
- **Cubical homology:** For image data, compute homology on the pixel grid directly — far more efficient than point cloud methods

## The Bottleneck Distance and Stability

A key theoretical property of persistent homology is the **stability theorem**: small perturbations to the input data produce at most equally small perturbations to the persistence diagram, as measured by the bottleneck distance:

$$d_B(D_1, D_2) = \inf_{\gamma: D_1 \to D_2} \sup_{x \in D_1} \|x - \gamma(x)\|_\infty$$

This stability makes persistent homology a robust feature extractor: it does not amplify noise.

## Persistent Homology as a Regularizer

Beyond feature extraction, persistent homology has been used as a **regularization term** in neural network training — penalizing topological complexity in learned representations or encouraging specific topological structure in generative models. This is an active research frontier connecting deep learning optimization with algebraic topology.

Topological data analysis represents a genuinely different perspective on machine learning data, one that captures shape and connectivity information inaccessible to purely metric or probabilistic approaches. As efficient implementations mature and the theoretical foundations deepen, persistent homology is transitioning from a specialized academic tool to a practical component of the ML practitioner's toolkit.
