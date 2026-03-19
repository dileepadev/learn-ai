---
title: "Attention Mechanisms Explained"
description: "A concise explanation of attention, scaled dot-product, and multi-head attention."
date: "2026-03-19"
tags: ["deep-learning", "transformers", "attention"]
---

Attention mechanisms let models focus on relevant parts of input when computing representations. They are central to modern architectures like Transformers.

## Scaled dot-product attention

Given queries (Q), keys (K), and values (V):

1. Compute scores: Q K^T / sqrt(d_k)
2. Apply softmax to get attention weights
3. Multiply weights by V to get the attended output

Scaling by sqrt(d_k) stabilizes gradients for large dimensions.

## Multi-head attention

- Split Q, K, V into multiple heads, apply attention in parallel, then concatenate and project.
- Benefits: allows the model to attend to different subspaces and capture diverse relations.

## Why attention matters

- It provides flexible context aggregation across tokens.
- It is permutation-invariant and efficient with optimized matrix operations.

## Practical notes

- Attention can be memory-heavy for long sequences; use sparse or locality-sensitive variants for long-context tasks.
- Relative positional encodings help generalization on longer sequences.

Next steps: experiment with small multi-head modules to see interpretability of attention maps.
