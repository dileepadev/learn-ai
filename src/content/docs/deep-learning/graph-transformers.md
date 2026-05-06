---
title: Graph Transformers
description: A comprehensive guide to Graph Transformers, which combine the expressiveness of self-attention with graph structure to overcome the limitations of traditional message-passing neural networks.
---

# Graph Transformers

**Graph Transformers** integrate the global self-attention mechanism of Transformers with the structural inductive biases of Graph Neural Networks (GNNs). While standard message-passing GNNs are limited by local neighborhood aggregation, Graph Transformers allow every node to attend to every other node — capturing long-range dependencies that are critical in molecular graphs, knowledge graphs, and social networks.

## Limitations of Message-Passing GNNs

Classical GNNs (GCN, GAT, MPNN) face two fundamental bottlenecks:

### Over-Smoothing

After $k$ layers of message passing, node representations converge toward indistinguishable vectors as neighborhood sizes grow exponentially:

$$h_v^{(k)} \to \text{const} \quad \text{as } k \to \infty$$

This limits effective depth to roughly 3–5 layers on most graph tasks.

### Over-Squashing

Information from exponentially many nodes must be compressed into a fixed-size vector as it traverses graph bottlenecks — narrow paths or bridges where many shortest routes converge. Formally, the Jacobian $\partial h_v^{(k)} / \partial h_u^{(0)}$ decays exponentially with the shortest path length $d(u, v)$.

Graph Transformers address both by computing attention globally, bypassing the path-length constraint.

## Architecture Overview

A Graph Transformer block replaces (or augments) message passing with multi-head self-attention over all node pairs:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top + B}{\sqrt{d_k}}\right) V$$

where $B \in \mathbb{R}^{N \times N}$ is a structural bias matrix encoding graph topology (edge weights, shortest path distances, etc.).

## Graph Positional Encodings

Unlike sequence Transformers with absolute positions, graphs lack a canonical node ordering. Graph Transformers use learnable structural encodings:

### Laplacian Positional Encoding (LPE)

Eigenvectors of the graph Laplacian $L = D - A$ provide smooth, structure-aware node positions:

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def laplacian_pe(adj: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute k smallest non-trivial Laplacian eigenvectors."""
    n = adj.shape[0]
    d = adj.sum(axis=1)
    D = sp.diags(d)
    A = sp.csr_matrix(adj)
    L = D - A

    # k+1 eigenvectors; skip trivial constant eigenvector (eigenvalue 0)
    eigenvalues, eigenvectors = eigsh(L.astype(float), k=k + 1, which="SM")
    # Sort by eigenvalue, skip first (constant)
    idx = np.argsort(eigenvalues)[1: k + 1]
    return eigenvectors[:, idx]   # (N, k)
```

### Random Walk Positional Encoding (RWPE)

Landing probabilities of $k$-step random walks encode structural roles:

$$\text{RWPE}_v = [p_{vv}^{(1)}, p_{vv}^{(2)}, \ldots, p_{vv}^{(k)}]$$

where $p_{vv}^{(t)} = (A D^{-1})^t_{vv}$ is the probability of returning to $v$ in $t$ steps.

## Key Model Variants

### Graphormer (Microsoft, 2021)

Graphormer introduced structural biases directly into the attention matrix:

- **Centrality encoding**: degree-based node features added to input embeddings
- **Spatial encoding**: shortest path distance $d(u,v)$ as a learnable scalar bias $b_{d(u,v)}$ in the attention logits
- **Edge encoding**: edge features along shortest paths averaged into attention

```python
import torch
import torch.nn as nn


class GraphormerAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_dist: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        # Learnable spatial bias per head per distance bucket
        self.spatial_bias = nn.Embedding(max_dist + 1, num_heads)

    def forward(self, x: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.d_k ** 0.5

        # Add spatial bias: (B, N, N, H) -> (B, H, N, N)
        spatial = self.spatial_bias(dist_matrix.clamp(max=20))  # (B, N, N, H)
        attn = attn + spatial.permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)
```

### GPS — General, Powerful, Scalable Graph Transformer

GPS (Rampásek et al., 2022) combines local MPNN layers with global attention in parallel:

```python
class GPSLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # Local: standard GNN message passing
        from torch_geometric.nn import GINEConv
        self.local_mpnn = GINEConv(nn.Linear(d_model, d_model))
        # Global: full self-attention
        self.global_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Local MPNN
        h_local = self.local_mpnn(x, edge_index, edge_attr)

        # Global attention (pad to dense batch)
        from torch_geometric.utils import to_dense_batch
        x_dense, mask = to_dense_batch(x, batch)
        h_global, _ = self.global_attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
        h_global = h_global[mask]

        # Combine
        h = self.norm1(x + h_local + h_global)
        h = self.norm2(h + self.ff(h))
        return h
```

### SAN — Spectral Attention Network

SAN uses Laplacian eigenvectors as keys/queries, allowing attention to be grounded in the spectral geometry of the graph.

## Scalability Challenges

Full self-attention is $O(N^2)$ in nodes — prohibitive for large graphs:

| Method | Complexity | Approach |
|---|---|---|
| Full attention (Graphormer) | $O(N^2)$ | All pairs |
| GPS | $O(N^2 + mN)$ | Global + local |
| Exphormer | $O(N \log N)$ | Sparse virtual nodes |
| NAGphormer | $O(N k)$ | Hop-aware tokenization |
| NodeFormer | $O(N \log N)$ | Kernelized attention |

For molecular graphs (typically $N < 100$), full attention is feasible. For social networks or knowledge graphs ($N \sim 10^6$), sparse or hierarchical variants are necessary.

## Applications and Benchmarks

| Domain | Task | Dataset | Top GT Model |
|---|---|---|---|
| Drug discovery | Molecular property | PCQM4Mv2 | Graphormer |
| Biochemistry | Protein interaction | STRING | GPS |
| NLP | Knowledge graph completion | FB15k-237 | SAT |
| Chemistry | Reaction prediction | USPTO | GT-RXN |
| Code analysis | Bug detection | CodeNet | GraphTrans |

## Summary

Graph Transformers overcome the local view of message-passing GNNs by enabling global, structure-aware attention over graph nodes. Positional encodings based on Laplacian eigenvectors or random walks give nodes a structural identity, while spatial biases in attention scores incorporate topology without sacrificing the expressiveness of self-attention. The GPS framework's combination of local MPNN and global attention is now a dominant paradigm, achieving state-of-the-art on molecular benchmarks while remaining modular and scalable.
