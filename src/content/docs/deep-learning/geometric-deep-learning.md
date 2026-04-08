---
title: "Geometric Deep Learning: Logic on Graphs and Manifolds"
description: "An introduction to the field of Geometric Deep Learning, focusing on symmetry, invariance, and non-Euclidean data."
---

Traditional deep learning excels on grids (images) and sequences (text). **Geometric Deep Learning (GDL)** generalizes these techniques to data that doesn't fit into a flat structure, such as graphs, social networks, and 3D shapes.

## The Core Principles

The "Erlangen Program" of Deep Learning suggests that architectures should be defined by the symmetries of the data they process:

- **Invariance**: The model's output remains the same even if the input is transformed (e.g., rotating a 3D molecule shouldn't change its predicted properties).
- **Equivariance**: If the input is transformed, the model's internal representations are transformed in a predictable way.

## Key Architectures

### 1. Graph Neural Networks (GNNs)

These models operate on nodes and edges, using "message passing" to update the state of a node based on its neighbors.

### 2. Spherical CNNs

Designed for data mapped onto a sphere, such as global weather patterns or omnidirectional 180/360-degree video.

## Why It Matters

GDL is critical for:

- **Drug Discovery**: Molecules are naturally represented as graphs.
- **Computer Vision**: Understanding 3D point clouds in autonomous driving.
- **Social Analysis**: Modeling influence and community detection in large networks.
