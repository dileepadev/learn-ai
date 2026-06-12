---
title: Attention Mechanisms
description: A guide to understanding attention mechanisms — the key innovation that powers modern transformers and large language models.
---

Attention mechanisms allow neural networks to dynamically focus on the most relevant parts of an input when producing an output. Originally introduced to improve sequence-to-sequence models in NLP, attention is now the foundation of Transformers and essentially all modern large language models (LLMs).

## The Problem Attention Solves

Early sequence models (RNNs, LSTMs) compressed the entire input into a single fixed-size context vector. This created a bottleneck — long sequences caused information to be lost or diluted. Attention lets the model selectively look back at all input tokens when generating each output token.

## How Attention Works

The core mechanism computes a weighted sum over values, where the weights reflect how relevant each value is to the current query.

Given:
- **Query (Q):** What the current position is "looking for"
- **Key (K):** What each input position "offers"
- **Value (V):** The actual content to aggregate

The attention output is computed as:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

The dot product of Q and K measures relevance; dividing by √d_k prevents gradient vanishing for large dimensions; softmax converts scores into a probability distribution over positions.

## Self-Attention

In **self-attention**, the queries, keys, and values all come from the same sequence. This allows each token to attend to every other token in the same sequence, capturing long-range dependencies directly — something RNNs struggle with.

## Multi-Head Attention

Instead of a single attention function, **multi-head attention** runs several attention operations in parallel with different learned projections:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ
```

Each head can learn to attend to different aspects (e.g., one head focuses on syntax, another on semantics). This is the key building block of the Transformer architecture.

## Types of Attention

- **Scaled Dot-Product Attention:** The standard form described above.
- **Additive Attention (Bahdanau):** Uses a feed-forward network to compute compatibility — the original attention mechanism from 2015.
- **Cross-Attention:** Queries come from one sequence (e.g., decoder), keys and values from another (e.g., encoder output). Used in encoder-decoder Transformers.
- **Causal / Masked Attention:** Each position can only attend to previous positions. Essential for autoregressive language models.
- **Sparse Attention:** Limits attention to a subset of positions to reduce the O(n²) cost. Examples: Longformer, BigBird.

## Computational Complexity

Standard attention scales as **O(n²)** in sequence length, which becomes expensive for very long contexts. This drives research into efficient alternatives like:
- **Flash Attention:** Reorders computation for memory efficiency without changing results.
- **Linear Attention:** Approximates softmax attention in linear time.

## Why Attention Matters

Attention is the mechanism that makes Transformers so powerful:
- Captures global context regardless of distance in the sequence.
- Parallelizable during training (unlike RNNs).
- Interpretable — attention weights can provide insights into model behavior.
- Generalizes beyond NLP to vision (ViT), audio, and multimodal models.
