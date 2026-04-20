---
title: Positional Encodings in Transformers
description: A technical deep dive into how transformers represent token position — covering absolute and relative encodings, Rotary Position Embeddings (RoPE), ALiBi, YaRN, and how positional schemes determine a model's ability to generalize to longer contexts.
---

**Positional encodings** are a critical component of transformer architecture. Because self-attention is inherently **permutation-invariant** — treating a sequence of tokens as a set with no inherent order — position information must be injected explicitly. How this is done profoundly affects a model's context length, generalization ability, and computational properties.

## Why Position Matters

Without positional encoding, a transformer would produce identical output regardless of whether "The cat sat on the mat" or "Mat the on sat cat the" was given as input. The attention mechanism treats all tokens as an unordered bag. Positional encodings break this symmetry by giving each token a representation of where it sits in the sequence.

## Absolute Positional Encodings

The original transformer (Vaswani et al., 2017) used **absolute positional encodings** — fixed vectors added to each token's embedding before it enters the transformer layers.

### Sinusoidal Encodings

The original paper used deterministic sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where $pos$ is the token position and $i$ is the dimension index. Different dimensions encode position at different frequencies — low-frequency dimensions capture coarse position; high-frequency dimensions capture fine-grained position.

**Properties:**

- Deterministic, no learned parameters.
- Can extrapolate to positions unseen during training in theory, but in practice degrades beyond training length.
- Relative positions can be derived from dot products of two position encodings.

### Learned Absolute Encodings

**BERT** and **GPT-2** replaced sinusoidal encodings with **learned embedding tables** — one learnable vector per position. The model learns what positional representation is most useful from data.

**Properties:**

- More flexible — adapts to the specific distribution of positional dependencies in training data.
- Hard limit at the maximum training length; cannot represent positions beyond the embedding table size.
- GPT-4, LLaMA, and most modern models have moved away from this scheme.

## Relative Positional Encodings

Rather than encoding absolute position, **relative positional encodings** encode the offset between pairs of tokens. This is more aligned with how linguistic structure works — what matters is often *how far apart* two tokens are, not their absolute positions in the sequence.

### T5 Relative Bias

T5 (Raffel et al., 2020) introduced learned relative position biases added to attention logits:

$$a_{ij} = q_i^\top k_j + b(i - j)$$

Where $b(i-j)$ is a learned scalar bias depending only on the relative distance. Distances beyond a threshold are bucketed (log-scale binning), sharing parameters.

**Advantage**: Allows some generalization beyond training length since the same relative biases apply regardless of absolute position.

## Rotary Position Embeddings (RoPE)

**RoPE** (Su et al., 2021) is the positional encoding scheme used by **LLaMA**, **Mistral**, **Qwen**, **Gemma**, and most modern open-source LLMs. It encodes position by rotating the query and key vectors before computing attention.

### The Core Idea

Instead of adding a position vector to the token embedding, RoPE multiplies the query/key vectors by a rotation matrix that depends on the absolute position:

$$f(x, m) = x e^{im\theta}$$

In practice, this rotates pairs of dimensions by an angle proportional to position $m$ and dimension-specific frequency $\theta$.

**The key property**: The inner product $\langle f(q, m), f(k, n) \rangle$ depends only on $q$, $k$, and the *relative* offset $m - n$. RoPE achieves relative position sensitivity through an absolute position operation.

### Properties of RoPE

- **Relative position encoding in disguise**: Attention scores automatically encode relative distance.
- **Decays with distance**: Attention scores between distant tokens naturally decay as the rotation angle difference grows — matching linguistic intuition.
- **No additional parameters**: Position is encoded without a learned embedding table.
- **Generalizes to longer sequences** (with extensions): Frequencies can be rescaled to extend to longer contexts.

## ALiBi: Attention with Linear Biases

**ALiBi** (Press et al., 2021) takes a different approach: don't encode position in the embeddings at all. Instead, add a **linear penalty** to attention logits based on the distance between tokens:

$$a_{ij} = q_i^\top k_j - m \cdot (i - j)$$

Where $m$ is a head-specific fixed slope. Closer tokens are penalized less; distant tokens are increasingly penalized.

**Advantages:**

- Extremely simple — no learned parameters, no embedding modifications.
- **Excellent length extrapolation**: Models trained with ALiBi on 1024 tokens often generalize well to 4096+ tokens at inference time.
- Used in **BLOOM**, **MPT**, and several multilingual models.

**Disadvantages:**

- The fixed linear decay may not be optimal for all tasks — long-range dependencies in documents or code are penalized even when they are semantically important.
- Performance on very long contexts can degrade compared to RoPE-based models with context extension techniques.

## Extending Context Windows: YaRN and Beyond

A major challenge with models trained on fixed context lengths is **extrapolating to longer sequences** at inference time. Several techniques extend RoPE-based models:

### Position Interpolation

**Linear position interpolation** (Chen et al., 2023) rescales position indices to fit within the original training range:

$$m' = m \cdot \frac{L_{\text{train}}}{L_{\text{new}}}$$

This compresses position representations that were trained on 4096 tokens into a 32768-token window by running positions 0–32768 through the original 0–4096 range. With fine-tuning on long sequences, this achieves substantial context extension.

### NTK-Aware Scaling

NTK-aware scaling addresses the limitation that naive interpolation over-compresses high-frequency RoPE dimensions (which encode fine-grained position differences) while under-using low-frequency dimensions. It distributes the context extension more evenly across frequency bands.

### YaRN

**YaRN** (Yet Another RoPE extensioN, Peng et al., 2023) combines NTK-aware scaling with a **temperature adjustment** to the attention softmax:

$$a_{ij} = \frac{q_i^\top k_j}{\sqrt{d} \cdot t}$$

The temperature factor $t > 1$ reduces attention entropy, compensating for the distributional shift at longer contexts. YaRN is used by **Mistral** models and several others to extend context from 4K to 32K–128K tokens with minimal fine-tuning.

### RoPE + Fine-Tuning (LLaMA Long)

Meta fine-tuned LLaMA 2 models on long-context data with rescaled RoPE to produce **LLaMA 2 Long** — demonstrating that a combination of architecture modification and data works better than architecture alone.

## Comparison at a Glance

| Method | Type | Length Extrapolation | Parameters | Used By |
| --- | --- | --- | --- | --- |
| Sinusoidal | Absolute | Poor | None | Original Transformer |
| Learned Absolute | Absolute | None | Table size | BERT, GPT-2 |
| T5 Relative | Relative | Moderate | Small | T5, UL2 |
| RoPE | Relative (via rotation) | Moderate (extendable) | None | LLaMA, Mistral, Qwen |
| ALiBi | Relative (via bias) | Good | None | BLOOM, MPT |

## Positional Encoding and Context Length

The choice of positional encoding is one of the primary determinants of a model's effective context length:

- Models with **learned absolute encodings** are hard-limited to their training context.
- Models with **RoPE + context extension** (YaRN, NTK) can generalize to 2–8× their training context with fine-tuning, and often somewhat beyond without.
- Models with **ALiBi** exhibit the smoothest length extrapolation of the classic methods.
- **No positional encoding** (e.g., some state-space models like Mamba) sidestep the problem entirely by using recurrent dynamics rather than attention.

Understanding positional encodings is essential for working with transformer-based LLMs — it directly governs what sequence lengths models can handle and how performance degrades as context grows beyond training distribution.
