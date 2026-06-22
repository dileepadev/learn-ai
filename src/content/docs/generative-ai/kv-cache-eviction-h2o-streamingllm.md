---
title: "KV Cache Eviction: H2O and StreamingLLM"
description: Explore KV cache eviction strategies like H2O and StreamingLLM that compress context state by discarding non-essential tokens while preserving long-term coherence.
---

When generating long-form text, Large Language Models store intermediate Key and Value vectors for all past tokens to speed up inference—a mechanism known as the **KV Cache**. However, as sequence length scales, the memory occupied by the KV cache grows linearly $\mathcal{O}(N)$ and quickly exceeds the GPU VRAM limit.

**KV Cache Eviction** is a class of techniques designed to keep the cache footprint constant or bounded. By intelligently discarding non-essential tokens from the cache, strategies like **H2O (Heavy-Hitter Oracle)** and **StreamingLLM** enable LLMs to process infinite text streams or massive contexts with limited, constant memory.

---

## The Bottleneck: Active Context Memory

For an 8-billion parameter model (like Llama 3) with a sequence length of 32,000 tokens:
- Storing the model weights requires ~16 GB of memory.
- Storing the KV cache for a single batch element requires roughly **16 GB** of VRAM.
- At batch size 4, the KV cache footprint ($64\text{ GB}$) dwarfs the model size itself.

KV cache eviction algorithms tackle this by keeping only a small fraction of tokens in the cache at any time.

---

## StreamingLLM: Retaining Attention Sinks

A naive approach to limiting the KV cache is a sliding window: keep only the most recent $L$ tokens (e.g., the last 1,000 tokens). However, if you discard the early tokens, the perplexity of the model spikes dramatically, causing it to generate gibberish.

The researchers behind **StreamingLLM** identified why: **Attention Sinks**.

During pre-training, softmax normalizes attention weights across the sequence. Because the model must assign attention weights even when no tokens are relevant, it dumps a massive amount of attention weight onto the very first tokens (e.g., the first 1 to 4 tokens in the prompt). If these initial tokens are evicted from the KV cache, the attention score distribution breaks.

```
StreamingLLM Cache Structure:
[ Token 0 | Token 1 | Token 2 ] <--- Attention Sinks (Never Evicted)
           +
[ Token 1000 | Token 1001 ... | Token 1050 ] <--- Sliding Window (Recent Context)
```

By keeping just **4 initial tokens (attention sinks)** and a sliding window of the **recent $N$ tokens**, StreamingLLM maintains normal model behavior over sequences of up to 4 million tokens, using a constant memory footprint.

---

## H2O: Heavy-Hitter Oracle

While StreamingLLM preserves the attention sinks and local context, it loses access to middle-range details (e.g., facts mentioned 2,000 tokens ago).

**H2O (Heavy-Hitter Oracle)** solves this by identifying and retaining **Heavy Hitters**—tokens that receive the highest cumulative attention scores from subsequent tokens.

### How H2O Evicts Tokens
1. **Track Attention Scores:** The system maintains a running accumulator of the attention scores assigned to each token in the KV cache:
   
   $$S_i = \sum_{t > i} \text{Attention}(t \to i)$$

2. **Rank Tokens:** Tokens with high $S_i$ are categorized as "Heavy Hitters" (key nouns, concepts, or facts that the model repeatedly references).
3. **Eviction:** When the KV cache exceeds its budget, the system evicts the token with the *lowest* attention score accumulation, preserving both the attention sinks, local sliding window context, and the long-term heavy-hitter tokens.

---

## Comparison: StreamingLLM vs. H2O

| Aspect | StreamingLLM | H2O (Heavy-Hitter Oracle) |
|---|---|---|
| **Cache Composition** | 4 initial tokens + sliding window | Attention sinks + sliding window + heavy hitters |
| **Eviction Metric** | Position-based (oldest in window) | Dynamic attention weight accumulator |
| **Middle-Context Recall** | Poor (fully discarded) | High (preserves frequently cited facts) |
| **Computation Cost** | Extremely low (fixed slices) | Low (requires tracking scalar scores) |
| **Best Used For** | Infinite stream generation, chat | Retrieval, synthesis, multi-turn reasoning |

---

## Code Concept: Simulating H2O Eviction

Below is a Python demonstration of how H2O tracks heavy hitters and evicts the least-attended KV tensors.

```python
import torch

class H2OKVCache:
    def __init__(self, max_budget=1024, sink_size=4):
        self.max_budget = max_budget
        self.sink_size = sink_size
        
        # KV storage
        self.k_cache = None # [layers, batch, heads, seq_len, head_dim]
        self.v_cache = None
        
        # Attention score accumulator per token position
        self.accumulated_scores = []

    def update_scores(self, new_attention_weights):
        # new_attention_weights: [seq_len, current_cache_len]
        # Sum attention weights received by each cache index
        step_scores = new_attention_weights.sum(dim=0).tolist()
        
        # Update running totals
        for idx, score in enumerate(step_scores):
            if idx < len(self.accumulated_scores):
                self.accumulated_scores[idx] += score
            else:
                self.accumulated_scores.append(score)

    def evict_if_needed(self):
        cache_len = self.k_cache.shape[3]
        if cache_len <= self.max_budget:
            return
            
        # Determine how many tokens to evict
        evict_count = cache_len - self.max_budget
        
        # Find candidates for eviction (excluding sinks)
        candidates = list(range(self.sink_size, cache_len))
        
        # Sort candidates by accumulated attention scores
        candidates.sort(key=lambda idx: self.accumulated_scores[idx])
        
        # Select the lowest scoring indices to remove
        evict_indices = set(candidates[:evict_count])
        keep_indices = [i for i in range(cache_len) if i not in evict_indices]
        
        # Slice the K and V tensors
        self.k_cache = self.k_cache[:, :, :, keep_indices, :]
        self.v_cache = self.v_cache[:, :, :, keep_indices, :]
        
        # Update score tracking list
        self.accumulated_scores = [self.accumulated_scores[i] for i in keep_indices]
```
