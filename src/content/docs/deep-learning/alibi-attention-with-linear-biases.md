---
title: ALiBi - Attention with Linear Biases for Length Extrapolation
description: Learn how ALiBi replaces positional embeddings with a distance-proportional penalty on attention scores, enabling transformers to extrapolate to longer sequences.
---

ALiBi (Attention with Linear Biases) is a positional method that adds no position embeddings to the input. Instead, it penalizes attention scores in proportion to the distance between query and key tokens. Its main selling point is that models trained on short sequences can run on much longer ones at inference.

## The Mechanism

For a query at position i attending to a key at position j (j <= i in a causal model), the attention logit becomes:

```text
score(i, j) = (q_i . k_j) / sqrt(d)  -  m * (i - j)
```

where m is a fixed, head-specific slope. The farther a token is, the larger the penalty, giving a built-in recency bias.

## Head-Specific Slopes

Slopes follow a geometric sequence so that different heads attend at different ranges. For 8 heads:

```text
m = 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256
```

Heads with large slopes focus locally; heads with small slopes can look far back. The slopes are fixed, not learned.

## Implementation Sketch

```python
import torch

def alibi_bias(n_heads, seq_len):
    slopes = torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])
    pos = torch.arange(seq_len)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0)   # i - j
    return -slopes[:, None, None] * dist                  # (heads, seq, seq)

# scores = scores + alibi_bias(n_heads, seq_len)
```

## Advantages

- No positional embedding parameters
- Strong train-short, test-long behavior
- Slightly lower memory and faster training than sinusoidal positions in the original reports

## Limitations

- Strong recency bias may hurt tasks requiring precise long-range retrieval
- Most recent LLMs favor rotary embeddings with scaling methods, so ALiBi is less common in newer models, though it appears in models such as BLOOM and MPT
