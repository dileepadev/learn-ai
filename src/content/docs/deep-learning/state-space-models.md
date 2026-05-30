---
title: "State Space Models: Mamba and Beyond"
description: "Learn how state space models (SSMs) like Mamba offer a compelling alternative to transformers for sequence modeling, with linear-time inference and strong long-range dependency handling."
---

Transformers dominate sequence modeling, but their quadratic attention complexity is a fundamental bottleneck for very long sequences. **State Space Models (SSMs)** offer a mathematically elegant alternative with linear-time inference and competitive performance.

## What is a State Space Model?

SSMs are inspired by classical control theory. They model a sequence by maintaining a hidden state that evolves over time:

```
h'(t) = Ah(t) + Bx(t)   # state update
y(t)  = Ch(t) + Dx(t)   # output
```

Where:
- `x(t)` is the input at time t
- `h(t)` is the hidden state
- `A, B, C, D` are learned matrices

This is a continuous-time formulation. For discrete sequences, it's discretized using a timescale parameter Δ.

## S4: Structured State Spaces

The **S4** model (Structured State Space for Sequences) made SSMs practical for deep learning by:

1. Parameterizing A as a diagonal-plus-low-rank matrix for efficient computation.
2. Computing the SSM as a convolution during training (parallelizable).
3. Using the recurrent form during inference (constant memory, linear time).

S4 showed strong results on the Long Range Arena benchmark, handling sequences of thousands of tokens efficiently.

## Mamba: Selective State Spaces

The key limitation of S4 is that A, B, C are fixed for all inputs — the model can't selectively focus on or ignore parts of the input. **Mamba** solves this with **selective state spaces**:

- B, C, and Δ become functions of the input x.
- The model learns to selectively remember or forget information based on content.
- A hardware-aware parallel scan algorithm makes this efficient on GPUs.

Mamba achieves transformer-level performance on language modeling while scaling linearly with sequence length.

## Mamba-2 and Hybrid Architectures

**Mamba-2** reformulates the selective SSM as a structured matrix multiplication, enabling better theoretical understanding and faster GPU kernels.

**Hybrid models** (e.g., Jamba, Zamba) interleave Mamba layers with attention layers, getting the best of both: efficient long-range processing from SSM layers and precise retrieval from attention layers.

## SSMs vs. Transformers

| Property | Transformer | SSM (Mamba) |
|---|---|---|
| Training complexity | O(L²) | O(L) |
| Inference memory | O(L) KV cache | O(1) state |
| Recall from long context | Strong | Weaker |
| Throughput at long sequences | Poor | Excellent |

## Current Status

SSMs are a serious research direction but haven't yet displaced transformers at the frontier. The main challenge is that attention's ability to do exact lookup over the full context is hard to replicate with a fixed-size state. Hybrid architectures are currently the most promising path forward.
