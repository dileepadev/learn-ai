---
title: State Space Models and Mamba
description: Learn how State Space Models (SSMs) and the Mamba architecture challenge the Transformer's dominance by offering linear-time sequence modeling with selective state compression.
---

State Space Models (SSMs) are a family of sequence models rooted in classical control theory that have re-emerged as serious competitors to Transformers for long-sequence tasks. The **Mamba** architecture (2023) introduced selective state spaces, enabling SSMs to match Transformer quality while scaling to million-length sequences with linear time and memory complexity.

## What Is a State Space Model?

An SSM maps an input sequence $x(t)$ to an output $y(t)$ through a hidden state $h(t)$, governed by a system of differential equations:

$$\dot{h}(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
$$y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$

where:

- $\mathbf{A}$ is the state transition matrix (how the hidden state evolves)
- $\mathbf{B}$ is the input projection matrix
- $\mathbf{C}$ is the output projection matrix
- $\mathbf{D}$ is the feedthrough (skip connection)

For discrete sequences, this is discretized using a timescale parameter $\Delta$ into:

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t$$
$$y_t = \mathbf{C} h_t$$

## The S4 Model: Structured State Spaces

The S4 model (Gu et al., 2021) made SSMs practical by parameterizing $\mathbf{A}$ as a **HiPPO matrix** — a structured matrix designed to optimally memorize input history via orthogonal polynomial projections. This enabled extremely long-range dependency capture while remaining computationally efficient.

S4 could process sequences of length 16,000+ on tasks where Transformers struggled, particularly in **Long-Range Arena** benchmarks.

## The Core Problem with Original SSMs

Traditional SSMs have **time-invariant** parameters: $\mathbf{A}$, $\mathbf{B}$, and $\mathbf{C}$ are fixed for all positions. This means the model applies the same compression to every input regardless of content — it cannot selectively remember important tokens or forget irrelevant ones.

This is fundamentally different from attention, which is **content-aware**.

## Mamba: Selective State Spaces

Mamba (Gu & Dao, 2023) solves this with a **selection mechanism**: the SSM parameters $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ become **functions of the input**:

$$\mathbf{B}_t = \text{Linear}(x_t), \quad \mathbf{C}_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

Now the model can dynamically decide what to store in state and what to ignore — similar to how a human reader skims irrelevant sentences while retaining key facts.

## Complexity Comparison

| Model | Training Complexity | Inference Complexity | Memory |
|---|---|---|---|
| Transformer | $O(L^2 D)$ | $O(L^2 D)$ | $O(L^2)$ |
| S4/S6 (SSM) | $O(L D N)$ | $O(D N)$ per step | $O(D N)$ |
| Mamba | $O(L D N)$ | $O(D N)$ per step | $O(D N)$ |

where $L$ = sequence length, $D$ = model dimension, $N$ = state dimension.

Mamba's inference is **recurrent** (constant per-step cost), but its training uses a **parallel scan algorithm** for efficiency — getting the best of both worlds.

## Hardware-Aware Implementation

Mamba uses a **hardware-aware parallel scan** (similar to FlashAttention's philosophy) that keeps intermediate states in SRAM rather than HBM, dramatically reducing memory bandwidth. This makes Mamba 5x faster than Transformers at sequence lengths of 2K+ tokens.

## Mamba Architecture

A Mamba block replaces the Transformer's attention sublayer with a **Selective SSM** sublayer:

```
Input → LayerNorm → [Linear projection → SSM → Gate] → Linear → Output
                                     ↑
                            (Selective: B, C, Δ from input)
```

Multiple Mamba blocks are stacked, with the FFN sublayer retained from standard Transformers.

## Mamba 2 and Hybrid Architectures

**Mamba 2** (2024) reformulates the SSM as a structured matrix multiplication, enabling even faster training and establishing a theoretical connection to attention mechanisms — both can be seen as special cases of a **State Space Duality (SSD)** framework.

**Hybrid models** like **Jamba** (AI21 Labs) and **Zamba** alternate Mamba and Transformer blocks, capturing Mamba's efficiency for long contexts while retaining Transformer's strong in-context learning for short sequences.

## Where Mamba Shines

- **Genomics:** DNA sequence modeling with sequences up to 1M base pairs
- **Audio:** Raw waveform modeling (SampleMamba)
- **Video:** Long-video understanding without quadratic attention cost
- **Code:** File-level code completion with long context
- **Time Series:** Multi-variate forecasting with long historical windows

## Current Limitations

- **Recall-intensive tasks:** Mamba underperforms Transformers on tasks requiring precise lookup of specific earlier tokens (e.g., retrieval, multi-hop Q&A)
- **In-context learning:** Transformers still have an edge on few-shot learning benchmarks
- **Ecosystem:** Fewer optimized kernels, less tooling support compared to Transformer infrastructure

## Further Reading

- Gu et al. (2021), *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*
