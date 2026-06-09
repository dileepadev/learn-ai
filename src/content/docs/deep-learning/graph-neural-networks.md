---
title: Graph Neural Networks
description: An introduction to Graph Neural Networks (GNNs) — deep learning models that operate directly on graph-structured data.
---

Graph Neural Networks (GNNs) are a class of deep learning models designed to work with graph-structured data — data where entities (nodes) have relationships (edges) between them. Unlike images or text, graphs have irregular structure and variable connectivity, making standard neural networks unsuitable. GNNs solve this by learning to aggregate information from a node's local neighborhood.

## Why Graphs?

Many real-world problems are naturally represented as graphs:
- **Social networks:** Users as nodes, friendships as edges.
- **Molecules:** Atoms as nodes, chemical bonds as edges.
- **Knowledge graphs:** Entities as nodes, relations as edges.
- **Citation networks:** Papers as nodes, citations as edges.
- **Road networks:** Intersections as nodes, roads as edges.

## The Core Idea: Message Passing

Most GNNs are built on **message passing**: at each layer, every node aggregates feature information from its neighbors, updates its own representation, and passes the result to the next layer.

```
hᵥ⁽ˡ⁺¹⁾ = UPDATE(hᵥ⁽ˡ⁾, AGGREGATE({hᵤ⁽ˡ⁾ : u ∈ N(v)}))
```

After k layers, each node's representation captures information from its k-hop neighborhood.

## Common GNN Variants

- **GCN (Graph Convolutional Network):** Averages neighbor features with spectral-based convolutions. Simple and widely used.
- **GraphSAGE:** Samples a fixed number of neighbors and uses learnable aggregation functions (mean, LSTM, pooling). Scales to large graphs.
- **GAT (Graph Attention Network):** Assigns learned attention weights to neighbors instead of uniform averaging — more expressive.
- **GIN (Graph Isomorphism Network):** Theoretically most expressive; uses sum aggregation to distinguish graph structures.
- **Graph Transformer:** Applies Transformer-style attention to graph structures, combining local message passing with global attention.

## Types of Tasks

- **Node classification:** Classify individual nodes (e.g., categorize users in a social network).
- **Edge prediction / link prediction:** Predict missing or future edges (e.g., recommend friends, predict protein interactions).
- **Graph classification:** Classify entire graphs (e.g., predict if a molecule is toxic).
- **Graph generation:** Generate new graphs with desired properties (e.g., drug-like molecules).

## Applications

- **Drug discovery:** Predicting molecular properties and generating novel drug candidates (used in AlphaFold's structure module).
- **Recommendation systems:** Pinterest's PinSage, Uber Eats recommendations.
- **Fraud detection:** Catching fraud rings by analyzing transaction graphs.
- **Traffic forecasting:** Google Maps uses GNNs to predict travel times.
- **Physics simulation:** Modeling particle interactions for scientific computing.

## Libraries and Tools

- **PyTorch Geometric (PyG):** The most popular GNN library, with implementations of dozens of GNN variants.
- **DGL (Deep Graph Library):** Flexible, supports multiple backends (PyTorch, TensorFlow).
- **NetworkX:** Graph manipulation and analysis (not for deep learning but commonly used alongside GNN tools).
