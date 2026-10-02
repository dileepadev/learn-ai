---
title: Multi-Head Latent Attention - Low-Rank KV Compression
description: Learn how Multi-head Latent Attention (MLA), introduced in DeepSeek models, compresses keys and values into a low-rank latent to cut KV cache size without giving up quality.
---

Multi-head Latent Attention (MLA) is an attention variant introduced in the DeepSeek-V2 model. It reduces KV cache memory by caching a small compressed latent vector per token instead of full per-head keys and values.

## The Core Idea

Standard attention caches keys and values for every head. MLA projects the hidden state down to a low-dimensional latent, caches only that, and reconstructs keys and values with up-projection matrices when needed:

```text
c_t = W_down * h_t            # compressed latent, dim d_c  (cached)
k_t = W_uk * c_t              # reconstructed keys
v_t = W_uv * c_t              # reconstructed values
```

Since d_c is much smaller than (heads x head_dim), the cache shrinks substantially.

## Absorbing the Up-Projections

At inference, the up-projection matrices can be merged into other matrices:

- `W_uk` can be absorbed into the query projection, so attention scores are computed directly against the cached latent
- `W_uv` can be absorbed into the output projection

This avoids materializing full keys and values during decoding.

## Handling Rotary Positions

Rotary embeddings depend on position and do not commute with the absorbed matrices. MLA therefore uses a **decoupled RoPE** design: a small separate set of query/key dimensions carries the rotary position, and it is cached alongside the latent.

## Comparison

| Method | Cache per token (relative) | Quality |
|--------|---------------------------|---------|
| MHA | Largest | Baseline |
| GQA | Reduced by sharing heads | Slightly below MHA |
| MLA | Much smaller via low-rank latent | Reported on par with or better than MHA |

## Trade-offs

MLA adds architectural complexity and extra matrix operations, and its efficiency depends on well-optimized kernels. The payoff is a much smaller cache, which enables long contexts and high-throughput serving.
