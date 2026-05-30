---
title: "Rotary Positional Embeddings (RoPE)"
description: "Learn how Rotary Positional Embeddings encode position information in LLMs, why they enable better length generalization than absolute embeddings, and how they power most modern transformer models."
---

Transformers have no built-in sense of order — without positional encodings, "the cat sat on the mat" and "the mat sat on the cat" look identical. **Rotary Positional Embeddings (RoPE)** are the dominant approach for encoding position in modern LLMs, used in LLaMA, Mistral, Gemma, Qwen, and most other open-weight models.

## The Problem with Absolute Positional Embeddings

The original transformer added a fixed or learned vector to each token embedding based on its absolute position. This has two problems:

1. **No relative position information**: The model must learn that position 5 and position 6 are adjacent, rather than having this encoded directly.
2. **Poor length generalization**: Models trained on sequences up to length N struggle to generalize to longer sequences at inference time.

## How RoPE Works

RoPE encodes position by **rotating** the query and key vectors in the attention mechanism. For a token at position m, its query vector q is rotated by angle mθ, and for a token at position n, its key vector k is rotated by angle nθ.

When computing the dot product q·k (the attention score), the rotation difference (m-n)θ remains — encoding the **relative distance** between the two tokens, not their absolute positions.

Mathematically, for a 2D vector:

```
RoPE(x, m) = [x₁cos(mθ) - x₂sin(mθ), x₁sin(mθ) + x₂cos(mθ)]
```

For higher dimensions, this is applied to pairs of dimensions with different base frequencies.

## Key Properties

- **Relative position awareness**: Attention scores naturally depend on the distance between tokens.
- **Decaying influence**: The rotation frequencies are chosen so that distant tokens have less correlated representations.
- **No additional parameters**: RoPE is a deterministic function, not a learned embedding table.
- **Efficient implementation**: Can be applied as element-wise operations on Q and K.

## Context Length Extension

A major research area is extending RoPE beyond the training context length. Techniques include:

- **Position Interpolation**: Scale down position indices to fit within the trained range, then fine-tune.
- **YaRN (Yet another RoPE extensioN)**: Applies different scaling to different frequency components, preserving high-frequency information.
- **LongRoPE**: Progressively extends context by adjusting RoPE scaling factors.

These techniques allow models trained on 4K tokens to be extended to 128K or more with relatively little fine-tuning.

## RoPE vs. ALiBi

**ALiBi** (Attention with Linear Biases) is an alternative that adds a linear bias to attention scores based on distance. It generalizes better to longer sequences out of the box but doesn't support position interpolation as cleanly as RoPE. Most new models have converged on RoPE due to its flexibility and strong empirical performance.
