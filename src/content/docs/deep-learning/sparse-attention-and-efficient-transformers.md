---
title: Sparse Attention and Efficient Transformers
description: Explore sparse attention mechanisms that scale Transformers beyond quadratic complexity — covering Longformer, BigBird, sliding window attention, global tokens, random attention, and linear attention approximations for long-sequence modeling.
---

Standard scaled dot-product attention computes pairwise interactions between all $n$ tokens in a sequence, giving $O(n^2)$ time and memory complexity. For a sequence of 4,096 tokens with a 512-dimensional model, the attention matrix alone occupies ~67 MB per head — and this scales with the square of sequence length. Handling documents of 100K+ tokens requires fundamentally rethinking the attention computation.

**Sparse attention** replaces full pairwise attention with structured patterns that each token attends to only $k \ll n$ other tokens, reducing complexity to $O(n \cdot k)$ while preserving the ability to capture long-range dependencies through multi-hop information flow.

## Full Attention: The Quadratic Baseline

For a sequence of $n$ tokens with queries $Q$, keys $K$, and values $V$ (each $\in \mathbb{R}^{n \times d}$):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

The attention matrix $A \in \mathbb{R}^{n \times n}$ stores $n^2$ scores. For $n = 16{,}384$ (a common long-document length), this is 268M entries — 1 GB per head at float32. This is the bottleneck that sparse attention targets.

## Sparse Attention Patterns

### Local / Sliding Window Attention

The most natural sparsity pattern: each token attends to the $w$ tokens on each side within a sliding window of width $2w + 1$. This captures local context efficiently.

$$A_{ij} = \begin{cases} \text{score}(i, j) & \text{if } |i - j| \leq w \\ -\infty & \text{otherwise} \end{cases}$$

Complexity: $O(n \cdot w)$. With a window size $w = 512$ on a sequence of $n = 16{,}384$, this is 32× cheaper than full attention.

Sliding window attention alone cannot connect tokens that are more than $w$ positions apart in a single layer. With $L$ layers, information can propagate up to $L \cdot w$ positions — so stacking layers provides long-range receptive fields similar to dilated convolutions.

### Dilated Sliding Window

Augments the sliding window with **dilation** (gaps): token $i$ attends to positions $i - d, i - 2d, \ldots, i - wd$ and $i + d, i + 2d, \ldots, i + wd$ for dilation factor $d$. This increases the effective context window by $d$× at the same compute cost.

Different attention heads can use different dilation factors, allowing the model to simultaneously capture multiple scales of context.

### Global Tokens

A small number of tokens (often special tokens like `[CLS]`) are designated as **global tokens** that attend to and are attended by every other token:

- Global tokens attend to all $n$ positions: $O(g \cdot n)$ operations.
- All positions attend to global tokens: another $O(g \cdot n)$.
- Local attention handles the rest: $O(n \cdot w)$.

Total: $O(n \cdot (w + g))$. Global tokens serve as "communication hubs" through which distant tokens can exchange information without direct full pairwise attention.

### Random Attention

Each token attends to $r$ randomly selected positions in addition to its local window. Random connections provide short expected path lengths between any two tokens (similar to small-world networks), enabling efficient information routing:

$$\mathbb{E}[\text{path length}(i, j)] = O(\log n / r)$$

### Stride / Block-Sparse Patterns

Tokens attend to every $s$-th token globally (stride attention), or the sequence is divided into blocks and attention is computed densely within each block plus between selected block pairs (block-sparse). These patterns are cache-friendly on GPU hardware.

## Longformer

**Longformer** (Beltagy et al., 2020) combines three attention patterns:

1. Sliding window attention for local context (most tokens).
1. Dilated sliding window for select heads.
1. Global attention for task-specific tokens (e.g., `[CLS]` for classification, question tokens for QA).

This combination scales to sequences of 4,096–16,384 tokens with linear memory. Longformer was pretrained on long documents (books, scientific papers) and fine-tuned on tasks requiring long-document understanding: evidence retrieval, legal document analysis, scientific QA.

The implementation uses a custom CUDA kernel that implements the sliding window as a strided memory access, making it efficient in practice (not just in theory).

## BigBird

**BigBird** (Zaheer et al., 2020) uses three pattern types simultaneously for every token:

- **Local attention:** window of size $w$ tokens on each side.
- **Global attention:** $g$ special global tokens (either prepended sentinel tokens or task-specific tokens).
- **Random attention:** each token attends to $r$ randomly sampled positions.

BigBird proves theoretically that this combination is a **universal approximator of sequence functions** — it can simulate any function computable by a full-attention transformer, given sufficient global and random connections. This theoretical guarantee distinguishes BigBird from heuristic sparse patterns.

BigBird achieves strong results on genomics tasks (DNA sequence classification at length 4K–16K) where full attention is computationally prohibitive.

## Reformer

**Reformer** (Kitaev et al., 2020) reduces attention complexity using **locality-sensitive hashing (LSH)**:

1. Project queries and keys onto random vectors.
1. Use LSH to bucket queries and keys into $b$ hash buckets.
1. Within each bucket, compute full attention (attending to nearby vectors in embedding space).

The key insight: a query only needs to attend to keys with high inner products, and LSH efficiently finds those keys without computing all $n^2$ pairs. Complexity: $O(n \log n)$.

Reformer also uses **reversible residual layers** (RevNet) that allow backpropagation without storing all intermediate activations, reducing memory from $O(Ln)$ to $O(n)$ across $L$ layers.

## Linear Attention

**Linear attention** approximates the softmax attention kernel using a feature map $\phi$:

$$\text{softmax}(q \cdot k / \sqrt{d}) \approx \phi(q)^\top \phi(k)$$

For suitable $\phi$ (e.g., the positive random features used in Performer):

$$\text{Attention}(Q, K, V) \approx \frac{\phi(Q)\left(\phi(K)^\top V\right)}{\phi(Q)\left(\phi(K)^\top \mathbf{1}\right)}$$

By computing $\phi(K)^\top V \in \mathbb{R}^{r \times d}$ first (the inner bracket), attention reduces to $O(nr)$ operations — fully linear in sequence length. The trade-off is a loss in representation quality compared to exact softmax, particularly for tasks requiring sharp attention on specific tokens.

### Performer

**Performer** (Choromanski et al., 2020) uses **FAVOR+** (Fast Attention Via positive Orthogonal Random features) to construct the approximating feature map $\phi$:

$$\phi(x) = \frac{1}{\sqrt{m}} \exp\!\left(-\frac{\|x\|^2}{2}\right) \left[\exp(\omega_1^\top x), \ldots, \exp(\omega_m^\top x)\right]$$

where $\omega_1, \ldots, \omega_m$ are orthogonal random vectors. FAVOR+ guarantees unbiased estimation of the softmax kernel with low variance.

## Comparison of Efficient Transformer Methods

| Method | Complexity | Pattern Type | Long-Range | Exact? |
| --- | --- | --- | --- | --- |
| Full Attention | $O(n^2)$ | Dense | Yes | Yes |
| Longformer | $O(n \cdot w)$ | Local + global | Via global tokens | Yes |
| BigBird | $O(n \cdot (w+g+r))$ | Local + global + random | Yes (theoretically) | Yes |
| Reformer | $O(n \log n)$ | LSH buckets | Approximate | Approx |
| Performer | $O(n \cdot r)$ | Linear kernel | Yes | Approx |
| Flash Attention | $O(n^2)$ | Dense (IO-aware) | Yes | Yes |

Note: Flash Attention is not a sparse method — it computes full attention but uses tiling and recomputation to reduce memory bandwidth. It can be combined with sparse patterns.

## Practical Impact and Shift to Full Context

As hardware improved and FlashAttention made full $O(n^2)$ attention practical at lengths up to 128K–1M tokens, the emphasis shifted from sparse approximations to:

- **FlashAttention-2/3:** IO-aware full attention kernels that achieve near-hardware-limit throughput.
- **Sliding window + full hybrid (Gemma, Mistral):** Alternate sliding-window and full-attention layers — local context most of the time, global context every $k$ layers.
- **Ring Attention:** Distribute the attention matrix across multiple GPUs for sequences that don't fit on a single device.

Sparse attention patterns remain important for:

- Very long DNA/protein sequences (>32K tokens) where even FlashAttention is memory-limited.
- Edge/mobile deployment with strict memory budgets.
- Document-level tasks with clear local structure (legal documents, code files).

## Summary

Sparse attention replaces the $O(n^2)$ quadratic bottleneck of full attention with structured patterns — sliding windows, global tokens, random connections, and LSH bucketing — that reduce complexity to $O(n \log n)$ or $O(n)$ while preserving long-range communication through multi-hop routing. Longformer and BigBird pioneered practical sparse transformers for long documents; linear attention methods like Performer enable true $O(n)$ scaling at the cost of approximation quality. As hardware advances have partially alleviated the quadratic cost via FlashAttention, modern systems increasingly use hybrid architectures that combine local and full attention layers.
