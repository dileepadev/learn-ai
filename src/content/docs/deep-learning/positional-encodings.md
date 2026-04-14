---
title: Positional Encodings in Transformers
description: Understand how Transformer models represent sequence order through positional encodings — from the original sinusoidal scheme to modern Rotary Position Embeddings (RoPE), ALiBi, and context extension techniques like YaRN.
---

Transformers have no inherent sense of order — self-attention treats its inputs as a **set**, not a sequence. Without positional information, "the cat ate the fish" and "the fish ate the cat" would produce identical attention patterns. Positional encodings solve this by injecting order information into the model's representations.

## Why Order Matters

The meaning of a sentence is fundamentally order-dependent. In NLP, position encodes:
- **Syntactic roles:** Subject before verb before object (in English)
- **Temporal relationships:** Earlier events precede later ones
- **Proximity:** Nearby tokens are more likely to be semantically related

Self-attention's permutation invariance is a feature for set-structured data but a bug for sequential text, requiring an explicit positional mechanism.

## Absolute Positional Encodings

### Sinusoidal Encoding (Vaswani et al., 2017)
The original Transformer paper used a fixed, non-learned encoding based on sine and cosine functions at different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Here $pos$ is the token position, $d$ is the model dimension, and $i$ indexes the dimension. Each dimension oscillates at a different wavelength, from $2\pi$ to $10000 \cdot 2\pi$.

**Key properties:**
- Each position has a unique encoding vector
- The dot product of two position encodings only depends on their relative offset — enabling the model to learn relative position from absolute encodings
- Can extrapolate beyond training length (theoretically)

### Learned Absolute Embeddings
BERT and many subsequent models replaced the fixed sinusoidal scheme with **learned position embeddings** — one trainable vector per position, added to the token embedding:

$$h_t = \text{TokenEmbed}(x_t) + \text{PosEmbed}(t)$$

**Advantage:** The model can optimize positional information end-to-end.

**Disadvantage:** Cannot generalize beyond the maximum training length — position 513 has no learned embedding if training used at most 512 positions.

## Relative Positional Encodings

Rather than encoding absolute positions, relative encodings encode the **distance between token pairs**. This naturally generalizes — position 100 relative to position 90 is the same as position 20 relative to position 10.

### Transformer-XL / T5 Relative Biases
T5 adds learned scalar biases to attention logits based on the relative offset between query and key positions. Offsets are bucketed (nearby positions get distinct biases; distant positions share a bucket):

$$\text{Attention}_{ij} = \frac{q_i k_j^T}{\sqrt{d}} + b_{i-j}$$

### ALiBi: Attention with Linear Biases
ALiBi (Press et al., 2021) subtracts a fixed, non-learned linear penalty proportional to distance from attention logits:

$$\text{Attention}_{ij} = q_i k_j^T - m \cdot |i - j|$$

where $m$ is a head-specific slope (fixed geometric progression across heads).

**Key properties:**
- Zero parameters — no learned embeddings or biases
- Excellent length generalization: ALiBi models trained at 1024 tokens extrapolate to 2048+ tokens at inference time with minimal perplexity degradation
- Simple to implement and tune

## Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) has become the dominant positional encoding in modern LLMs (Llama, Mistral, Llama 3, Falcon, Qwen, etc.). It is elegant, efficient, and generalizes well.

### The Core Idea
RoPE encodes position by **rotating** query and key vectors in 2D subspaces by an angle proportional to the position:

For a 2D case:
$$R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$$

For a $d$-dimensional vector, RoPE splits the dimensions into $d/2$ pairs and applies separate 2D rotations with different base frequencies:

$$q_m' = R_{\theta_m} q_m, \quad k_n' = R_{\theta_n} k_n$$

The dot product of rotated query and key then only depends on their relative position $m - n$:

$$\langle q_m', k_n' \rangle = \langle R_{\theta_m} q, R_{\theta_n} k \rangle = f(q, k, m-n)$$

**Why this is powerful:**
- Absolute positions are encoded (each position gets a unique rotation)
- The attention score only depends on **relative** positions (rotation difference cancels out the absolute components)
- Computationally efficient — applied element-wise, no additional parameters

### RoPE Implementation

```python
def apply_rotary_emb(q, k, cos, sin):
    # q, k: (batch, seq_len, heads, head_dim)
    # Interleave the rotation across adjacent dimension pairs
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_rope = q * cos + rotate_half(q) * sin
    k_rope = k * cos + rotate_half(k) * sin
    return q_rope, k_rope
```

## Context Length Extension Techniques

A major challenge for deployed LLMs: models trained on context length $L$ degrade when given inputs longer than $L$ at inference time.

### Position Interpolation (PI)
Instead of extrapolating to unseen positions, **compress** the position indices to fit within the training range:

$$\theta' = \theta \cdot \frac{L_\text{train}}{L_\text{test}}$$

Results in slightly lower performance than native training at that length, but enables 4–8x context extension with minimal fine-tuning.

### YaRN: Yet Another RoPE Extensionn Method
YaRN (Peng et al., 2023) applies **frequency-dependent scaling** — high-frequency (short-range) RoPE dimensions are extrapolated rather than interpolated, while low-frequency (long-range) dimensions are interpolated. A temperature scaling factor adjusts attention logits to compensate for the changed distribution.

YaRN achieves near-native performance at extended context lengths (up to 128K) with as little as 400 fine-tuning steps.

### LongRoPE
LongRoPE (Ding et al., 2024) applies a non-uniform rescaling per RoPE dimension, computed by searching for the optimal per-dimension stretch factor on a small calibration dataset. Used in Phi-3 and other Microsoft models to extend context to 128K+.

## Comparison Summary

| Method | Learned | Relative | Extrapolates | Used By |
|---|---|---|---|---|
| Sinusoidal | No | Partially | Theoretically | Original Transformer |
| Learned Absolute | Yes | No | No | BERT, GPT-2 |
| T5 Relative Bias | Yes | Yes | Moderate | T5, Flan-T5 |
| ALiBi | No | Yes | Yes | MPT, BLOOM |
| RoPE | No | Yes | Moderate | Llama, Mistral, Qwen |
| RoPE + YaRN | No | Yes | Yes (extended) | Llama variants |

## Positional Encodings Beyond Text

- **Vision Transformers (ViT):** 2D sinusoidal or learned 2D absolute position embeddings for image patches
- **Video Transformers:** 3D positional encodings (height, width, time) for video patches
- **Graph Transformers:** Structural positional encodings based on graph topology (eigenvalues of Laplacian)
- **Point clouds:** Continuous 3D coordinates encoded as Fourier features

## Further Reading

- Vaswani et al. (2017), *Attention Is All You Need* — sinusoidal PE
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)*
- Press et al. (2021), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)*
- Peng et al. (2023), *YaRN: Efficient Context Window Extension of Large Language Models*
