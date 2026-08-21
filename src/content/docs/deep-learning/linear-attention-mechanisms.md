---
title: "Linear Attention Mechanisms"
description: Understand linear attention and sub-quadratic transformer variants — how kernel methods, feature maps, and recurrent reformulations break the O(n²) barrier of standard self-attention to enable efficient sequence modeling at scale.
---

The transformer's central operation — scaled dot-product attention — is widely regarded as one of the most important algorithmic innovations in modern deep learning. But it carries a steep computational cost: for a sequence of length $n$, attention requires $O(n^2)$ time and memory. For most NLP tasks with sequences of hundreds to a few thousand tokens, this is acceptable. For tasks requiring very long contexts — genomics, audio, long documents, video — it becomes a hard bottleneck.

Linear attention is the family of approaches that rewrite or approximate standard attention to achieve $O(n)$ or near-linear complexity, unlocking efficient processing of arbitrarily long sequences.

## Standard Self-Attention: The Quadratic Baseline

In scaled dot-product attention, queries $Q$, keys $K$, and values $V$ are matrices of shape $(n, d)$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

The bottleneck is the $n \times n$ attention matrix $QK^\top$. For $n = 16{,}384$ tokens, this matrix has $\approx 268$ million entries. Storing it in float16 requires ~536 MB per attention layer per batch element. With multiple layers and large batch sizes, memory becomes the first constraint to break.

The softmax normalization ensures the rows of the attention matrix sum to one — a key property that makes attention behave like a soft retrieval operation. Linear attention methods must preserve (or approximate) this property while avoiding the explicit $n \times n$ matrix.

## The Kernel Trick for Attention

The key mathematical insight behind linear attention: the softmax in standard attention can be approximated as a kernel function. If we write:

$$\text{softmax}(q_i^\top k_j / \sqrt{d}) \approx \phi(q_i)^\top \phi(k_j)$$

where $\phi: \mathbb{R}^d \to \mathbb{R}^r$ is a feature map, then the attention output for query $q_i$ becomes:

$$o_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}$$

The numerator can be rewritten by associativity of matrix multiplication:

$$o_i = \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j) v_j^\top \right)}{\phi(q_i)^\top \left(\sum_j \phi(k_j)\right)}$$

Let $S = \sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{r \times d}$ and $z = \sum_j \phi(k_j) \in \mathbb{R}^r$. Then:

$$o_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}$$

**The critical observation:** $S$ and $z$ are sums over all keys and values — they can be computed once in $O(n)$ time and reused for every query. The total complexity drops from $O(n^2 d)$ to $O(n r d)$, which is linear in $n$ when $r \ll n$.

### Choosing the Feature Map $\phi$

The choice of $\phi$ determines how well the approximation matches true softmax attention:

**Random Fourier Features (Performer):** The Performer (Choromanski et al., 2020) uses random Fourier features to approximate the RBF kernel. By drawing random frequencies $\omega \sim \mathcal{N}(0, I)$:

$$\phi(x) = \frac{1}{\sqrt{r}} \left[\exp\!\left(\omega_1^\top x\right), \ldots, \exp\!\left(\omega_r^\top x\right)\right]$$

This provides an unbiased approximation: $\mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k)$. The Performer achieves near-linear complexity with provable approximation quality controlled by $r$.

**ELU-based features (Linear Transformer):** The simplest practical feature map (Katharopoulos et al., 2020):

$$\phi(x) = \text{elu}(x) + 1$$

This avoids random projections entirely. It's faster but offers no approximation guarantee — the dot product of ELU-mapped features is not close to the softmax kernel in general. Empirically, it works reasonably for tasks that don't require fine-grained selective attention.

**Positive orthogonal random features (FAVOR+):** An improved version of the Performer that uses positive and orthogonal random features, reducing variance and improving approximation quality without increasing $r$.

## Recurrent Reformulation

A complementary perspective on linear attention: any linear attention model can be rewritten as a recurrent neural network. This is important because it means the model can process sequences token by token with $O(1)$ state, rather than attending over all previous tokens.

Define the running state $S_t = \sum_{j \leq t} \phi(k_j) v_j^\top$. At each step:

```
S_t = S_{t-1} + φ(k_t) * v_t^T   # rank-1 update
z_t = z_{t-1} + φ(k_t)            # scalar update
o_t = φ(q_t)^T * S_t / (φ(q_t)^T * z_t)
```

This is a recurrence with matrix-valued hidden state. During training, the parallel form computes $S$ efficiently over the full sequence. During inference, the recurrent form allows constant-time per-token generation — a significant advantage over standard attention which must recompute or cache all previous key-value pairs.

This duality — **parallel training, recurrent inference** — is a defining characteristic of the broader class of efficient sequence models, including Mamba and RWKV.

## Gating and Selective State Updates

Pure linear attention has a limitation: every key-value pair contributes equally to the running state, regardless of relevance. Standard attention's softmax provides implicit selectivity — it concentrates attention on a small number of highly relevant positions. Without gating, linear attention treats all positions as equally important.

Modern linear attention models add **gating mechanisms** to address this:

**Gated Linear Attention (GLA):** Introduces a data-dependent gate $G_t \in (0, 1)^{d \times d}$ applied to the state:

$$S_t = G_t \odot S_{t-1} + \phi(k_t) v_t^\top$$

The gate controls how much of the previous state is retained versus replaced. When $G_t = 1$ everywhere, the model behaves like standard linear attention. When $G_t = 0$, the state is fully overwritten, creating a "forget" mechanism analogous to LSTM cells.

**RetNet (Retention Network):** Uses a fixed exponential decay $\gamma^{t-j}$ that discounts older tokens. This provides a simple inductive bias — recent context matters more — without learned gating.

**HGRN2 and Hawk/Griffin:** Recent architectures that combine linear recurrences with local attention windows, achieving strong empirical performance on language benchmarks while maintaining near-linear complexity.

## Sub-quadratic Attention: Beyond Pure Linearity

Not all efficient attention methods are purely linear. Several achieve sub-quadratic complexity by approximating the full attention matrix more carefully:

**Sparse attention:** Restricts each query to attend only to a subset of keys — nearby positions (local/sliding window), strided positions, or globally designated "landmark" tokens. The attention matrix is sparse, reducing computation proportionally.

- **Longformer** (Beltagy et al., 2020): Sliding window of size $w$ plus global tokens. Complexity $O(n \cdot w)$.
- **BigBird** (Zaheer et al., 2020): Combines random, local, and global attention. Theoretical guarantees that sparse attention is a universal approximator.
- **Longformer and BigBird comparison:**

| Model | Pattern | Complexity | Relative Context |
|-------|---------|-----------|-----------------|
| Full attention | All pairs | $O(n^2)$ | Full |
| Sliding window | Local only | $O(nw)$ | Limited long-range |
| Longformer | Window + global | $O(n \cdot (w + g))$ | Good |
| BigBird | Random + local + global | $O(n)$ | Theoretical full |

**Low-rank attention:** Approximates the $n \times n$ attention matrix with a product of lower-rank matrices. Linformer (Wang et al., 2020) projects keys and values to a fixed dimension $k \ll n$ before computing attention, reducing complexity to $O(nk)$.

**FlashAttention (not linear, but memory-efficient):** It's worth noting that FlashAttention (Dao et al., 2022, 2023) doesn't reduce theoretical complexity but reorders computation to be IO-efficient, dramatically reducing memory bandwidth requirements. It's often the practical choice for sequences up to ~100K tokens.

## Practical Tradeoffs

The choice between linear and standard attention involves real tradeoffs:

**Expressiveness.** Standard softmax attention has a theoretically stronger inductive bias for selective retrieval. Tasks requiring precise lookup of a specific token from a long context (e.g., "what was the value mentioned 10,000 tokens ago?") are harder for pure linear models. Gated variants mitigate but don't fully eliminate this gap.

**Training stability.** Linear attention models can exhibit training instabilities if the feature map produces zero denominators or the state grows without bound. Careful normalization (layer norm on the state, small initialization) is important.

**Performance on benchmarks.** On standard language modeling benchmarks (perplexity on The Pile, LAMBADA, etc.), recent linear attention models approach but typically don't match transformer baselines of the same parameter count. The gap has narrowed significantly in 2023-2024 with architectures like RWKV-v6, Mamba-2, and GLA.

**Inference throughput.** For generation tasks, the recurrent form of linear attention is dramatically faster than cached attention at long sequence lengths. A model generating token $n=100{,}000$ with linear attention accesses a fixed-size state; standard attention must attend over 99,999 cached key-value pairs.

## State Space Models as Linear Attention

State space models (SSMs) — particularly S4 and Mamba — are closely related to linear attention. Both can be viewed as instances of the general formulation:

$$h_t = A_t h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t$$

where $h_t$ is the hidden state, $x_t$ is the input, and $A_t$, $B_t$, $C_t$ are (possibly input-dependent) matrices.

The key differences are in how the matrices $A_t$, $B_t$, $C_t$ are parameterized:
- **S4**: Diagonal structured $A$ with no input dependence — very efficient, strong theoretical guarantees for long-range dependencies
- **Mamba**: Input-dependent $A$, $B$, $C$ via selective scan — adds selectivity at modest cost
- **GLA / RetNet / RWKV**: Factored representations equivalent to linear attention with specific feature maps

The unification of SSMs and linear attention under the "linear RNN" framework (Dao & Gu, 2024) provides a clean conceptual foundation for understanding this entire class of models.

## When to Use Linear Attention

Linear attention variants are particularly well-suited for:

- **Very long sequences** (>32K tokens) where quadratic memory is prohibitive
- **Streaming or real-time inference** where the recurrent form enables token-by-token processing
- **Edge/mobile deployment** where memory constraints are tight
- **Tasks with strong local dependencies** where the inability to do precise long-range lookup is less costly

Standard softmax attention remains preferable for:

- **Retrieval-intensive tasks** where finding a specific token in a long context is essential
- **Moderate-length sequences** where FlashAttention makes quadratic complexity practical
- **Tasks with available compute** where absolute performance matters more than efficiency

The field is rapidly evolving. As linear attention architectures close the performance gap with transformers on standard benchmarks, the case for their use in production systems grows stronger.
