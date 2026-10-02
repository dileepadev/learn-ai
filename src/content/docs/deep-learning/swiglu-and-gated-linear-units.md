---
title: SwiGLU and Gated Linear Units in Transformer Feed-Forward Layers
description: Learn how gated linear units, especially SwiGLU, replace the standard transformer MLP in modern LLMs and why gating improves quality at matched compute.
---

Most modern large language models (LLaMA, Mistral, PaLM and others) replace the original transformer feed-forward network with a gated variant, usually SwiGLU. The change is small in code but consistently improves quality.

## The Standard FFN

```text
FFN(x) = W2 * activation(W1 * x)
```

with a hidden size usually 4x the model dimension.

## Gated Linear Units

A GLU splits the computation into two parallel projections, one acting as a gate:

```text
GLU(x) = (W1 * x)  (elementwise*)  sigmoid(W2 * x)
```

The gate lets the network control, per feature, how much information passes through, which acts like a learned, input-dependent filter.

## Variants

| Variant | Gate activation |
|---------|-----------------|
| GLU | sigmoid |
| ReGLU | ReLU |
| GeGLU | GELU |
| SwiGLU | Swish / SiLU |

SwiGLU in a transformer block:

```text
FFN_SwiGLU(x) = W_down * ( SiLU(W_gate * x)  (elementwise*)  (W_up * x) )
```

## Implementation

```python
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

## Keeping Parameter Counts Fair

A gated FFN has three weight matrices instead of two. To match the parameters of a standard 4x FFN, models shrink the hidden width to about 8/3 of the model dimension (often rounded to a hardware-friendly multiple).

## Why It Works

The original paper that proposed these variants offered little theoretical explanation and attributed the gains to the architecture empirically. A common intuition is that multiplicative interactions increase expressiveness beyond a single pointwise nonlinearity.
