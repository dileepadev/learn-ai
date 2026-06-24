---
title: Mamba and Selective State Space Models
description: A comprehensive guide to Mamba and selective state space models — covering the limitations of Transformers that motivated their design, the S4/S6 architecture, the selection mechanism, hardware-aware algorithms, and how Mamba competes with Transformers on language modeling.
---

**Mamba** is a deep learning architecture based on **selective state space models (SSMs)** that achieves Transformer-level performance on language modeling while scaling **linearly** with sequence length — compared to the quadratic complexity of standard attention. Introduced by Gu and Dao in December 2023, Mamba quickly became one of the most influential architecture innovations in years, sparking a wave of follow-up work across language, vision, and multimodal domains.

## The Quadratic Attention Problem

The Transformer's self-attention mechanism is its greatest strength and its most significant bottleneck. Computing attention over a sequence of length $L$ requires:

- **Time complexity**: $O(L^2)$ — every token attends to every other token
- **Memory complexity**: $O(L^2)$ — the full attention matrix must be materialized

For short sequences (< 2K tokens), this is acceptable. For long sequences (100K+ tokens needed for book-length context, genomics, or long videos), quadratic scaling becomes prohibitively expensive.

Alternatives pursued over the years — sparse attention, linear attention, sliding window attention — either sacrifice global context, compromise on expressivity, or require complex approximations. **State space models** offer a fundamentally different approach rooted in classical signal processing.

## State Space Models: Mathematical Foundation

A **linear time-invariant (LTI) state space model** maps an input sequence $x(t)$ to an output sequence $y(t)$ through a hidden state $h(t)$:

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

where:
- $A \in \mathbb{R}^{N \times N}$ is the **state transition matrix** (how the hidden state evolves)
- $B \in \mathbb{R}^{N \times 1}$ is the **input projection matrix**
- $C \in \mathbb{R}^{1 \times N}$ is the **output projection matrix**
- $N$ is the **state dimension**

In continuous time this is a differential equation; discretized for use in neural networks with step size $\Delta$:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

where $\bar{A}$ and $\bar{B}$ are discretized versions of $A$ and $B$.

This formulation enables two equivalent computational modes:
- **Recurrent mode**: Process sequentially, maintaining $h_t$ — $O(L)$ time and $O(1)$ memory (constant hidden state)
- **Convolutional mode**: Unroll into a global convolution kernel — enables parallel training via FFT

## S4: Structured State Spaces

**S4** (Structured State Space for Sequences, Gu et al., 2021) was the breakthrough that made deep SSMs practical. The fundamental challenge: learning $A$ naively is unstable. S4 constrains $A$ to a **diagonal plus low-rank (DPLR)** structure, specifically the **HiPPO matrix** — designed to optimally memorize history by projecting past inputs onto orthogonal polynomial bases.

The HiPPO initialization gives SSMs a powerful inductive bias for long-range dependencies, far outperforming vanilla RNNs on long-sequence tasks. S4 achieved state-of-the-art results on the **Long Range Arena (LRA)** benchmark, demonstrating sequence modeling at lengths of 4K–16K tokens with linear complexity.

**Limitation of S4**: The LTI property means parameters $A$, $B$, $C$ are **fixed regardless of the input content**. The model processes all inputs identically — it cannot selectively focus on relevant tokens or ignore irrelevant ones.

## The Selection Mechanism: Mamba's Core Innovation

Mamba's key insight: the inability of S4 to selectively process inputs is what prevents it from reaching Transformer-level language modeling performance. The **selection mechanism** makes $B$, $C$, and $\Delta$ **input-dependent**:

$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

This seemingly simple change has profound consequences:

- **$\Delta_t$ controls time step size**: A large $\Delta$ compresses the state (forgets the past), a small $\Delta$ preserves it. The model can dynamically choose to remember or ignore context based on the current input.
- **Selective copying**: When given a task like "copy the token that follows the word 'color'", the model can learn to set $\Delta$ large for irrelevant tokens (erasing them from state) and small for relevant ones (preserving them).
- **Context-dependent filtering**: Unlike attention, which explicitly computes similarities, Mamba implicitly routes information through the state based on learned input-dependent transitions.

The selectivity makes $A$ effectively input-dependent too (since $\bar{A} = \exp(\Delta A)$ and $\Delta$ is input-dependent), even though $A$ itself is fixed — a subtle but important point.

**The price of selection**: Input-dependent parameters break the LTI property, which means the model **cannot be computed as a global convolution** anymore. The fast parallel training mode disappears.

## Hardware-Aware Parallel Scan

To recover training efficiency despite losing the convolutional mode, Mamba uses a **parallel scan** (prefix scan) algorithm implemented with **hardware-aware kernel fusion** on GPUs.

The parallel scan computes all hidden states $\{h_1, h_2, \ldots, h_L\}$ in $O(\log L)$ parallel time using an associative reduction — similar to parallel prefix sums. Gu and Dao implemented this as a custom **CUDA kernel** that:

1. **Avoids materializing** intermediate states in HBM (high-bandwidth memory) — the bottleneck for memory-bound operations
2. **Fuses** the scan, input projection, and activation into a single kernel
3. Uses **recomputation** during the backward pass to avoid storing activations (trading compute for memory)

This hardware-aware design is why Mamba achieves **5× higher throughput** than Transformer at long sequences despite the theoretical equivalence. The implementation insight — that memory bandwidth, not FLOPs, is the real bottleneck on modern GPUs — is a recurring theme in efficient deep learning (cf. Flash Attention).

## Mamba Architecture

The Mamba block replaces the Transformer's attention sublayer with an **SSM sublayer** while retaining the overall residual block structure:

```
Input → LayerNorm → [SSM Block] → residual add
                 ↓
         Linear (expand 2×)
              ↙        ↘
         SSM path    Gate path
              ↘        ↙
          element-wise multiply
               ↓
         Linear (project back)
```

**SSM Block internals**:
1. Expand input from $D$ to $E = 2D$ dimensions
2. Split into two branches
3. One branch: apply 1D depthwise convolution, then SiLU activation, then SSM
4. Other branch: SiLU gate
5. Multiply branches (gated activation)
6. Project back to $D$ dimensions

The depthwise convolution adds local context before the SSM, improving performance on tasks requiring position-aware local patterns.

## Performance: Language Modeling

On language modeling (the core benchmark for LLMs), Mamba at 1.4B parameters trained on 300B tokens:

- **Matches Transformer++ (optimized Transformer) at equal parameters/tokens** on perplexity
- **Scales better at inference**: Mamba generates tokens at 5× the throughput of Transformers at sequence length 2K, improving further as context grows
- **Linear memory**: Mamba's state is constant-size regardless of context — eliminates the KV cache that grows linearly in Transformers

For **in-context learning**, Mamba performs comparably to Transformers on most benchmarks but shows a relative weakness on tasks requiring precise retrieval from very early context — reflecting a fundamental difference in how SSMs vs. attention maintain long-range information.

## Mamba-2 and the SSD Framework

**Mamba-2** (Dao and Gu, 2024) reformulates the Mamba selective SSM as a **Structured State Space Duality (SSD)**, revealing a mathematical connection between SSMs and **linear attention**:

- Under certain parameterizations, the Mamba SSM is equivalent to a form of linear attention with a specific kernel
- This unification enables 2-8× faster training than Mamba-1 by using **tensor cores** (matrix multiplication hardware) more efficiently
- SSD allows mixing SSM layers with attention layers, yielding hybrid models that combine the strengths of both

**Hybrid architectures** (interleaving Mamba and attention layers) have become a dominant approach, as a small number of full attention layers (for precise retrieval) combined with many Mamba layers (for efficient long-context processing) often outperforms pure Mamba or pure Transformer models.

## Mamba Beyond Language

### Vision Mamba

**VMamba** and **Vision Mamba (Vim)** apply Mamba to image modeling by serializing 2D image patches into 1D sequences using various scan orders (row, column, diagonal, spiral). Challenges include:

- Images lack natural left-to-right ordering — multiple scan directions are combined
- Inductive biases for spatial locality must be recovered through architecture choices

Vision Mamba models achieve competitive results with vision Transformers (ViT) at lower computational cost for high-resolution images.

### Mamba for Genomics

**Caduceus** and **HyenaDNA** apply SSM architectures to DNA sequences, which can span millions of base pairs — far beyond Transformer context limits. Mamba's linear scaling makes genome-scale modeling tractable for the first time.

### Mamba in Audio

Audio signals at 44.1kHz require modeling sequences of tens of thousands of tokens per second. Mamba-based audio models process raw waveforms efficiently where Transformers would require aggressive downsampling.

## Comparison: Mamba vs. Transformer

| Property | Transformer | Mamba |
|---|---|---|
| Time complexity (training) | $O(L^2)$ | $O(L)$ |
| Memory (KV cache at inference) | $O(L)$ | $O(1)$ |
| Inference throughput | Decreases with context | Constant |
| In-context retrieval | Excellent | Good (weaker on precise recall) |
| Long-range dependencies | Quadratic cost | Linear cost |
| Hardware efficiency | Well-optimized (FlashAttention) | Custom CUDA kernels required |
| Interpretability | Attention weights are inspectable | Hidden state is opaque |

## Current Landscape (2025)

The initial Mamba-vs-Transformer debate has largely settled into a recognition that both architectures have distinct strengths, and **hybrid models** combining selective SSMs with attention are often optimal:

- **Jamba** (AI21 Labs): Mixture-of-experts Mamba-Transformer hybrid
- **Zamba** (Zyphra): Compressed hybrid with shared attention layers
- **Falcon Mamba 7B**: First Mamba-based model to match Llama-scale Transformer quality
- **NVIDIA's research** on SSM-Transformer hybrids for long-context reasoning

Mamba demonstrated that the Transformer's architectural dominance since 2017 was not inevitable — and opened the field to a new generation of architectures that challenge the attention-is-all-you-need paradigm.
