---
title: Rotary Positional Embeddings
description: Learn how Rotary Positional Embeddings (RoPE) encode token position by rotating query and key vectors in complex space — covering the mathematical derivation, relative attention formulation, long-context extensions (YaRN, LongRoPE, dynamic NTK scaling), and why RoPE has become the dominant positional encoding in modern large language models.
---

Positional encodings allow transformer models to distinguish tokens by their position in a sequence. Without positional information, the self-attention mechanism is permutation-invariant — it treats the input as an unordered set of tokens. **Rotary Positional Embeddings (RoPE)**, introduced by Su et al. (2021), encode position by rotating query and key vectors in complex number space. Unlike sinusoidal or learned absolute positional embeddings, RoPE encodes **relative** positions directly in the attention score computation, with no explicit position embedding vector added to the token representation.

RoPE is used in LLaMA, Mistral, Falcon, Gemma, Qwen, and most other leading open-weight LLMs as of 2024–2025, having displaced earlier approaches due to its combination of mathematical elegance, training stability, and long-context extrapolation properties.

## Background: Why Positional Encodings Matter

Self-attention computes a score between query $q_m$ at position $m$ and key $k_n$ at position $n$:

$$\text{score}(m, n) = q_m^\top k_n$$

For a model to understand sequence structure, this score should depend on the **relative offset** $m - n$, not merely the content of the tokens. Absolute positional encodings (APE) add a position-dependent vector to each token embedding before computing attention — the position information then leaks into the attention scores indirectly through the dot product. This is indirect and does not naturally encode the relative relationship.

**Relative positional encodings** (Shaw et al., 2018; Raffel et al., 2020 T5 bias) modify attention scores to depend explicitly on the offset $m - n$, but require additional learned or fixed bias terms that add computational overhead.

RoPE achieves relative positional encoding by choosing a specific transformation $f(x, m)$ of token representations that makes the dot product $\langle f(q, m), f(k, n) \rangle$ depend only on $q$, $k$, and $m - n$.

## Mathematical Derivation

### Setup in 2D

Consider a 2-dimensional query/key space. Encode position $m$ as a rotation by angle $m\theta$ in the 2D plane:

$$f(x, m) = \begin{pmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_1 \sin(m\theta) + x_2 \cos(m\theta) \end{pmatrix} = R_m x$$

where $R_m$ is the $2 \times 2$ rotation matrix for angle $m\theta$. The inner product between rotated query $f(q, m)$ and rotated key $f(k, n)$ is:

$$\langle f(q, m), f(k, n) \rangle = \langle R_m q, R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k$$

This depends only on the **relative position** $n - m$ — exactly the property we want. The rotation matrices cancel algebraically, encoding relative position without any explicit position vector.

### Extension to $d$ Dimensions

For a $d$-dimensional hidden state, the rotation is applied independently to each pair of adjacent dimensions $(x_{2i-1}, x_{2i})$ for $i = 1, \ldots, d/2$. Each pair uses a different base frequency $\theta_i$:

$$\theta_i = \text{base}^{-2(i-1)/d}$$

where $\text{base} = 10000$ (the original RoPE value, borrowed from sinusoidal encoding). The full rotation matrix $R_m^{(d)}$ is block-diagonal:

$$R_m^{(d)} = \begin{pmatrix} R_{m\theta_1} & & \\ & \ddots & \\ & & R_{m\theta_{d/2}} \end{pmatrix}$$

The attention score between position $m$ and position $n$ decomposes across dimension pairs:

$$q_m^\top k_n = \sum_{i=1}^{d/2} \left( q_{2i-1} k_{2i-1} \cos((m-n)\theta_i) + q_{2i} k_{2i} \cos((m-n)\theta_i) + \text{cross terms} \right)$$

Different frequency pairs $\theta_i$ encode positional information at different scales — low-index pairs (small $i$) have high frequency and distinguish nearby tokens; high-index pairs (large $i$) have low frequency and capture coarse position relationships.

### Complex Number Formulation

RoPE can be expressed compactly using complex numbers. Treat each dimension pair as a complex number $z_i = x_{2i-1} + i \cdot x_{2i}$. Applying RoPE multiplies each complex pair by the phase factor $e^{im\theta_i}$:

$$\tilde{z}_i = z_i \cdot e^{im\theta_i}$$

The attention score is then:

$$\text{Re}\left(\sum_{i=1}^{d/2} \tilde{q}_i^* \cdot \tilde{k}_i\right) = \text{Re}\left(\sum_{i=1}^{d/2} q_i^* \cdot k_i \cdot e^{i(n-m)\theta_i}\right)$$

which depends only on the relative offset $n - m$, confirming the relative encoding property.

## Efficient Implementation

In practice, RoPE is applied efficiently without constructing the full block-diagonal rotation matrix. For a query or key vector $x \in \mathbb{R}^d$, the rotation is computed as:

```python
def apply_rope(x, cos, sin):
    # x: [batch, heads, seq_len, d_head]
    # cos, sin: [seq_len, d_head] — precomputed for current positions
    x1 = x[..., :x.shape[-1] // 2]   # Even dimensions
    x2 = x[..., x.shape[-1] // 2:]   # Odd dimensions
    # Rotate: apply the 2x2 rotation to each pair
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
```

The precomputed `cos` and `sin` tensors have shape `[seq_len, d_head]` and are cached for the maximum expected sequence length. Applying RoPE costs two element-wise multiplications and one addition — negligible relative to the attention computation.

## Long-Context Extensions

RoPE was originally designed for sequences of up to 2048 tokens (LLaMA-1) or 4096 tokens (LLaMA-2). Extending to longer contexts requires modifying the rotational frequencies, as the model has not seen position indices beyond its training context during training.

### Position Interpolation (PI)

**Position Interpolation** (Chen et al., 2023) scales the position indices by a factor $L_{\text{train}} / L_{\text{target}}$:

$$m \rightarrow m \cdot \frac{L_{\text{train}}}{L_{\text{target}}}$$

This maps a 32768-token context into the original [0, 4096] position range. The model sees familiar position values and generalizes reasonably well after a small amount of fine-tuning on long-context data. The tradeoff is reduced resolution at short distances — nearby tokens receive very similar positional encodings.

### NTK-Aware Scaling

**Neural Tangent Kernel (NTK)-aware scaling** modifies the base frequency rather than scaling position indices. Increasing the base from 10000 to a larger value (e.g., 500000) stretches the rotational period of each frequency component:

$$\theta_i = \text{base\_new}^{-2(i-1)/d}, \quad \text{base\_new} = \text{base} \cdot \left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{d/(d-2)}$$

This preserves short-range positional resolution (high-frequency components are unchanged) while extending long-range capacity (low-frequency components rotate more slowly). NTK-aware scaling was discovered empirically in the community and works surprisingly well without any fine-tuning.

### YaRN (Yet Another RoPE Extension)

**YaRN** (Peng et al., 2023) combines frequency interpolation with an attention scale correction. It applies different scaling strategies to different frequency components:

- High-frequency dimensions: no change (preserves short-range locality).
- Mid-frequency dimensions: position interpolation.
- Low-frequency dimensions: no interpolation (already captures long range).

YaRN also adjusts the attention temperature (scaling factor before softmax) to compensate for the reduced attention score magnitudes that occur at long distances after interpolation. LLaMA-3 uses a variant of YaRN for its 128K context window.

### LongRoPE

**LongRoPE** (Ding et al., 2024) searches for optimal per-dimension non-uniform scaling factors rather than applying a uniform formula. It uses an evolutionary search algorithm to find the scaling coefficients that minimize perplexity on long-context text after fine-tuning. LongRoPE enables context lengths of up to 2M tokens in Phi-3-mini-128K.

## RoPE vs. Other Positional Encodings

| Method | Relative PE | Extrapolates | Memory overhead | Used in |
| --- | --- | --- | --- | --- |
| Sinusoidal (absolute) | No | Partially | None | Original Transformer |
| Learned absolute | No | No | $O(L \cdot d)$ | BERT, GPT-2 |
| ALiBi | Yes (bias) | Yes (linear bias) | None | MPT, BLOOM |
| T5 relative bias | Yes (bucketed) | Partially | $O(L^2 / \text{buckets})$ | T5, Flan |
| **RoPE** | **Yes** | **With scaling** | **None** | **LLaMA, Mistral, Gemma** |

### RoPE vs. ALiBi

**ALiBi** (Press et al., 2022) adds a position-dependent linear bias to attention scores: $-|m - n| \cdot s$ where $s$ is a per-head slope. It extrapolates well by construction (the linear bias naturally penalizes distant tokens) but does not model complex positional structure that RoPE captures through sinusoidal components. RoPE has empirically outperformed ALiBi on most benchmarks when paired with long-context fine-tuning.

## Multi-Head Attention Compatibility

RoPE is applied separately to each attention head before computing the dot product. Different heads share the same rotational frequencies (same $\theta_i$ values) but rotate different portions of the query/key vectors. This is fully compatible with grouped query attention (GQA) and multi-query attention (MQA) used in Mistral, LLaMA-3, and Gemma — the rotation is applied to both query and key vectors regardless of the number of key-value heads.

## Training Stability

RoPE's stability advantages over absolute positional encodings arise from several properties:

- **No learnable parameters**: RoPE requires no position embedding matrix, removing one source of optimization instability and eliminating the possibility of position embedding collapse.
- **Bounded magnitude**: rotations preserve vector norms exactly, preventing position-induced explosion or vanishing of attention scores.
- **Smooth frequency spectrum**: the logarithmically spaced frequencies $\theta_i = \text{base}^{-2i/d}$ provide smooth coverage of positional scales without frequency aliasing.

## Summary

Rotary Positional Embeddings encode token position by rotating query and key vectors in complex space, ensuring that attention scores depend only on the relative offset between positions. The key insight is that rotations compose multiplicatively — rotating position $m$ and position $n$ separately and then computing the dot product is equivalent to applying a single rotation by $n - m$, giving exact relative position encoding at zero additional memory cost. Extensions including NTK-aware scaling, YaRN, and LongRoPE enable context windows of 128K–2M tokens with fine-tuning. RoPE's combination of mathematical elegance, parameter-free design, and long-context scalability has made it the standard positional encoding across virtually all modern open-weight large language models.
