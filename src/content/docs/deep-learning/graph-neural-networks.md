---
title: Graph Neural Networks
description: An introduction to GNNs — deep learning models that operate on graph-structured data like molecules, social networks, and knowledge graphs.
---

Graph Neural Networks (GNNs) are neural networks designed to work with graph-structured data, where entities are nodes and relationships are edges. Standard neural architectures assume regular inputs like grids (images) or sequences (text); GNNs handle the irregular connectivity of graphs.

## Core Idea: Message Passing

Most GNNs use a **message passing** scheme. At each layer, every node collects feature information from its neighbors, aggregates it, and updates its own representation:

```
hᵥ⁽ˡ⁺¹⁾ = UPDATE(hᵥ⁽ˡ⁾, AGGREGATE({hᵤ⁽ˡ⁾ : u ∈ N(v)}))
```

After k layers, each node's embedding captures information from its k-hop neighborhood.

## Common Variants

- **GCN:** Averages neighbor features with normalized convolutions. Simple and widely used.
- **GraphSAGE:** Samples neighbors and aggregates with learnable functions. Scales to large graphs.
- **GAT:** Uses learned attention weights over neighbors instead of uniform averaging.
- **GIN:** Sum aggregation — theoretically most expressive for distinguishing graph structures.

## Task Types

- **Node classification:** Predict a label for each node (e.g., classify users in a network).
- **Link prediction:** Predict missing edges (e.g., friend recommendations, drug-target interactions).
- **Graph classification:** Classify entire graphs (e.g., predict molecular toxicity).

## Applications

- **Drug discovery:** Predicting molecular properties; used in AlphaFold's structure module.
- **Recommendation systems:** Pinterest's PinSage, Uber Eats.
- **Fraud detection:** Identifying fraud rings via transaction graph analysis.
- **Traffic forecasting:** Google Maps uses GNNs for travel time prediction.

## Getting Started

**PyTorch Geometric (PyG)** is the most popular library, with implementations of dozens of GNN architectures and benchmark datasets built in.
