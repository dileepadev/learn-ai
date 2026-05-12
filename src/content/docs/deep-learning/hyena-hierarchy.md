---
title: Hyena Hierarchy
description: Understand the Hyena hierarchy — a subquadratic drop-in replacement for attention in deep learning that uses long convolutions and data-controlled gating to capture long-range dependencies without the quadratic cost of self-attention. Covers the Hyena recurrence, implicit parameterization of convolution filters, H3 and Hyena-ViT variants, state space connection, throughput benchmarks, and comparisons with Mamba and linear attention.
---

**Hyena** (Poli et al., 2023) is a subquadratic sequence model designed as an attention-free alternative to Transformer self-attention, capable of processing sequences of arbitrary length with $O(N \log N)$ computational cost per layer (via FFT-based long convolutions) instead of $O(N^2)$. Hyena replaces the query-key-value attention mechanism with a sequence of **data-controlled long convolutions** — element-wise gated by projections of the input — enabling recurrence-like expressivity without the quadratic memory and compute bottleneck of full attention.

## Motivation: The Quadratic Bottleneck

Self-attention computes pairwise token interactions:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

with cost $O(N^2 d)$ in sequence length $N$. For language models at $N = 8192$ tokens, attention already dominates compute; at $N = 100k$ tokens (long documents, genomics, audio), it becomes prohibitively expensive.

Linear attention, sparse attention (Longformer, BigBird), and state space models (S4, Mamba) are alternatives. Hyena occupies a distinct position: a **dense convolutional model** that maintains $O(N \log N)$ complexity while matching or approaching attention quality through long implicit convolution filters and multiplicative data-gating.

## The Hyena Operator

The Hyena operator processes an input sequence $u \in \mathbb{R}^{N \times d}$ through three components:

### 1. Linear Projections

Project the input into $p+1$ sequences:

$$v, z_1, \ldots, z_p = \text{Linear}(u)$$

$v$ is the "value" and $z_1, \ldots, z_p$ are "gating" projections.

### 2. Long Convolution Filters

Learn $p$ long convolution filters $h^{(1)}, \ldots, h^{(p)} \in \mathbb{R}^N$ — one filter per gating projection. These are **implicit filters**: rather than storing $N$ learnable parameters per filter (which would be parameter-heavy and hard to optimize), Hyena parameterizes filters via a small neural network:

$$h^{(k)}(t) = \text{MLP}_k(t) \cdot w(t)$$

where $t \in \{0, 1, \ldots, N-1\}$ are timestep indices, $\text{MLP}_k$ is a small positional MLP (sine activations), and $w(t)$ is a learned window function that decays toward zero at large lags — ensuring the filter has finite effective support.

### 3. Hyena Recurrence

Apply the Hyena recurrence iteratively:

$$y^{(0)} = v$$
$$y^{(k)} = z_k \cdot (h^{(k)} * y^{(k-1)}), \quad k = 1, \ldots, p$$
$$\text{Hyena}(u) = y^{(p)}$$

where $*$ denotes causal (or bidirectional) convolution and $\cdot$ is element-wise multiplication.

Each iteration: convolve the current sequence with a learned filter, then gate element-wise by the next projection. Stacking $p$ such operations creates a deep mixture of convolutions with multiplicative interactions — enabling the model to condition filter outputs on the current input (data control), analogous to how attention weights in transformers are input-dependent.

**Efficiency**: each $h^{(k)} * y^{(k-1)}$ is computed via FFT in $O(N \log N)$. With $p = 2$ (default), a single Hyena layer costs $O(N \log N)$ — matching Fast Fourier Transform complexity.

## Relationship to H3

**H3** (Fu et al., 2023) precedes Hyena and proposes a two-stage sequence model:

1. A **shift SSM** (state space model) that propagates information forward across positions.
1. A **multiplicative gate** conditioned on a diag-SSM transformation of the input.

H3 can be viewed as a special case of the Hyena hierarchy with $p=2$ and SSM-parameterized convolution filters instead of implicit MLP filters. Both aim to capture long-range dependencies with data-dependent gating, but Hyena's implicit filter parameterization is more flexible and doesn't require SSM-specific initialization procedures.

## Implicit Convolution Filter Properties

The implicit MLP parameterization of filters is a key design choice:

- **Sinusoidal positional encodings**: using $\sin(2\pi k t / N)$ features enables filters with oscillatory structure at multiple frequencies — capturing both short-range (high frequency) and long-range (low frequency) dependencies.
- **Exponential decay window**: multiplying by $e^{-\alpha t}$ prevents filter coefficients at large lags from being freely learnable (which leads to instability) — encouraging the model to learn compact, well-structured filters.
- **Length generalization**: because filters are parameterized by a function of relative position (not a lookup table), Hyena naturally generalizes to sequence lengths longer than those seen during training — unlike learned absolute positional embeddings.

## Performance vs. Attention

On language modeling benchmarks (The Pile, OpenWebText), Hyena with order $p=2$ trained at 125M-1.3B parameters achieves:

- **Comparable perplexity** to GPT-3 at 125M parameters with the same number of training tokens.
- **2-5× faster throughput** at sequence lengths $N \geq 4096$ due to $O(N \log N)$ vs. $O(N^2)$ attention cost.
- **Persistent gap** at small sequence lengths ($N \leq 2048$): attention models slightly outperform Hyena, likely because softmax attention's inductive bias (local normaliation, content-based retrieval) aids in short-context tasks.

At long sequences ($N = 64k$), Hyena is dramatically faster than standard attention — enabling training on long-context documents that would be memory-prohibitive with full attention.

## Hyena for DNA and Genomics

Hyena's $O(N \log N)$ scaling makes it particularly attractive for **genomic sequence modeling** — where sequences can be $10^4$-$10^6$ base pairs long and long-range regulatory interactions (enhancers, insulators) span thousands of bases:

**HyenaDNA** (Nguyen et al., 2023) trains a Hyena-based language model on the human genome at single-nucleotide resolution — processing sequences up to 1 million tokens in a single forward pass. Pretrained HyenaDNA achieves state-of-the-art on GenomicBenchmarks (chromatin accessibility, epigenetic marks, gene expression) — tasks requiring integration of information across hundreds of thousands of base pairs that vanilla Transformers cannot process without windowing.

## Comparison with Mamba

**Mamba** (Gu & Dao, 2023) is another recent attention-free architecture combining selective state space models with hardware-efficient parallel scan algorithms. Comparison:

| Property | Hyena | Mamba |
| --- | --- | --- |
| Core operation | Long FFT convolution + gating | Selective SSM (parallel scan) |
| Complexity | $O(N \log N)$ | $O(N)$ |
| Data-controlled | Yes (multiplicative gating) | Yes (input-dependent SSM params) |
| Filter expressivity | Implicit MLP (flexible) | SSM transition matrices |
| Length generalization | Natural (continuous filter) | Requires position-aware training |
| Genomics performance | Excellent (HyenaDNA) | Competitive |

Mamba achieves $O(N)$ complexity (strictly better than $O(N \log N)$) via hardware-efficient scan, and has matched or outperformed Hyena on many language benchmarks. However, Hyena's FFT-based approach has simpler implementation and strong theoretical grounding in the theory of long convolutions.

## Summary

Hyena is a subquadratic sequence model that replaces attention with a hierarchy of data-controlled long convolutions, achieving $O(N \log N)$ per-layer complexity. Implicit MLP-parameterized filters enable length generalization and capture multi-scale temporal structure. Stacked multiplicative gating creates input-dependent filtering analogous to attention's content-based token interaction. Hyena matches GPT-level perplexity at 125M-1.3B parameters while offering significant throughput advantages at long sequence lengths. Its most compelling application is genomics (HyenaDNA), where single-nucleotide resolution modeling of megabase sequences reveals long-range regulatory interactions inaccessible to attention-based models. Hyena, H3, and Mamba collectively represent a class of attention-free architectures that may eventually challenge Transformer dominance for long-sequence tasks.
