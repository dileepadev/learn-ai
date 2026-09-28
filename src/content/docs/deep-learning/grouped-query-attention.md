---
title: Grouped-Query Attention - Shrinking the KV Cache
description: Learn how multi-query and grouped-query attention share key/value heads across query heads to reduce KV cache memory and speed up LLM decoding.
---

During autoregressive generation, each new token attends to cached keys and values from all previous tokens. This KV cache dominates memory and bandwidth for long contexts. Grouped-query attention (GQA) shrinks it by sharing key/value heads.

## Three Attention Layouts

| Method | Query heads | KV heads |
|--------|-------------|----------|
| Multi-head attention (MHA) | H | H |
| Multi-query attention (MQA) | H | 1 |
| Grouped-query attention (GQA) | H | G (1 < G < H) |

In GQA, the H query heads are divided into G groups, and all query heads in a group use the same key and value head.

## Cache Size Savings

Per token and per layer, the KV cache stores `2 * kv_heads * head_dim` values. With 32 query heads and 8 KV heads, the cache is 4x smaller than MHA. That means longer contexts or larger batches in the same memory, and less data to read each decode step (decoding is typically memory-bandwidth bound).

## Why Not Just MQA?

MQA gives the maximum savings but can degrade quality and cause training instability. GQA recovers most of MHA's quality while keeping most of MQA's speed.

## Implementation Sketch

```python
import torch

def gqa(q, k, v, n_groups):
    # q: (B, H, T, D)   k, v: (B, G, T, D)
    B, H, T, D = q.shape
    repeat = H // n_groups
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    att = (q @ k.transpose(-2, -1)) / D ** 0.5
    return att.softmax(-1) @ v
```

## Uptraining

An existing MHA checkpoint can be converted by mean-pooling KV heads within each group and then continuing training for a small fraction of the original compute.

## Adoption

GQA is used in many open models, including the LLaMA 2 70B and LLaMA 3 families and Mistral 7B.
