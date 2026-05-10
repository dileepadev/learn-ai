---
title: Medusa, EAGLE, and Tree-Based Decoding
description: Understand self-speculative decoding methods — Medusa multi-head drafting, EAGLE feature-level speculation, Jacobi decoding, and lookahead decoding — that accelerate LLM inference by generating multiple tokens per forward pass without a separate draft model.
---

Standard autoregressive decoding generates one token per forward pass of the full model: each new token requires a complete matrix multiplication through every layer. For a 70B-parameter model, a single forward pass takes ~20ms on one A100 GPU, capping generation at ~50 tokens/second regardless of how fast the hardware is. **Self-speculative decoding** methods break this one-token-per-pass limitation by generating multiple candidate tokens in parallel during a single or few forward passes, then verifying them efficiently.

This distinguishes them from **draft-model speculative decoding** (e.g., SpecDecoding, Medusa-1's original motivation), which requires a separate smaller model. Self-speculative methods use only the target model itself — no external draft model — making them simpler to deploy.

## The Speculation-Verification Framework

All speculative decoding methods share the same structure:

1. **Draft phase:** Cheaply generate $k$ candidate tokens $\hat{t}_1, \ldots, \hat{t}_k$.
1. **Verify phase:** Run the full model in a single batched forward pass to score all $k$ candidates simultaneously.
1. **Accept/reject:** Keep all tokens that match what the full model would have generated; reject the first mismatch and restart.

The speedup comes from the verify phase: scoring $k$ candidates in one pass is only marginally more expensive than scoring 1 token (due to the KV cache), yet $k$ tokens are potentially accepted.

The **expected acceptance length** $\mathbb{E}[\tau]$ determines the speedup. If each token is accepted independently with probability $\alpha$:

$$\mathbb{E}[\tau] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

For $\alpha = 0.8$ and $k = 5$, $\mathbb{E}[\tau] \approx 3.6$ — generating 3.6× more tokens per verify call, translating to ~2.5–3× wall-clock speedup (accounting for draft overhead).

## Medusa: Multi-Head Drafting

**Medusa** (Cai et al., 2024) attaches $k$ lightweight draft heads to the final layer of a frozen LLM. Each head predicts a token at a different future position:

- Head 0: predicts token at position $t+1$ (same as the base model).
- Head 1: predicts token at position $t+2$.
- Head $i$: predicts token at position $t+i+1$.

Each Medusa head is a shallow MLP applied to the last hidden state $h_t$:

$$\hat{p}^{(i)}(\cdot \mid h_t) = \text{softmax}\!\left(W_i \cdot \text{SiLU}(U_i h_t)\right)$$

The heads are trained independently with a cross-entropy loss on the ground-truth next tokens. Importantly, head $i$ predicts position $t+i+1$ conditioned only on $h_t$ — not on the intermediate tokens at $t+1, \ldots, t+i$. This means head $i$'s predictions are less accurate for larger $i$, but the heads are cheap to evaluate (small MLP vs. full transformer layer).

### Tree Attention for Verification

Medusa uses **tree-structured verification** to maximize accepted length. Instead of generating a single chain $(\hat{t}_1, \hat{t}_2, \ldots, \hat{t}_k)$, Medusa samples top-$m$ candidates from each head and constructs a tree of possible continuations:

$$\text{Tree: } \{(c_1^{(1)}, c_2^{(j)}, \ldots) \mid c_i^{(j)} \in \text{top-}m(\hat{p}^{(i)})\}$$

A single batched attention pass with a **tree attention mask** verifies all paths in the tree simultaneously. The longest accepted prefix in the tree is adopted. This turns the multi-head predictions into a combinatorial branching search at virtually no extra compute cost.

### Medusa-2: Joint Training

**Medusa-2** co-trains the draft heads alongside the base model using a combined loss, rather than freezing the base model and training heads separately. This allows the base model's representations to evolve to be more "draft-friendly," improving acceptance rates at the cost of requiring full model training access.

## EAGLE: Feature-Level Speculation

**EAGLE** (Li et al., 2024) identifies the key limitation of Medusa: predicting future tokens from a single hidden state $h_t$ (without access to intermediate tokens) is fundamentally limited. EAGLE's draft model instead operates at the **feature level** — it receives the hidden states as input, not just the last one.

### EAGLE Architecture

EAGLE prepends a small autoregressive draft model (a single transformer decoder layer) that operates on the sequence of hidden states $\{h_0, h_1, \ldots, h_t\}$ from the frozen target model:

$$\tilde{h}_{t+1} = \text{DraftLayer}([h_t, \text{embed}(\hat{t}_1)])$$

$$\hat{t}_2 = \text{LM\_head}(\tilde{h}_{t+1})$$

By conditioning on the actual last hidden state (not just the final-layer output of an MLP head), EAGLE captures much richer context. The draft layer receives: the last hidden state $h_t$, the embedding of the last accepted token, and attends over all previous draft hidden states.

This design achieves significantly higher acceptance rates than Medusa (typically 80–90% vs. 65–75% per token), at the cost of slightly more overhead per draft step (one transformer layer vs. one MLP).

### EAGLE-2

**EAGLE-2** introduces a **dynamic draft tree**: instead of a fixed tree structure, EAGLE-2 uses the draft model's confidence scores to adaptively expand only the most promising branches. This improves efficiency on easy tokens (where one path dominates) while maintaining coverage on ambiguous tokens.

## Jacobi Decoding

**Jacobi decoding** (Song et al., 2021) takes a completely different approach based on iterative fixed-point equations. Instead of sequential token generation, all tokens in a sequence are predicted simultaneously and then iteratively refined:

1. Initialize the sequence with a rough guess: $y^{(0)} = (y_1^{(0)}, \ldots, y_n^{(0)})$ (e.g., the most frequent token).
1. At each iteration, update all positions in parallel using the current sequence as context:

$$y_i^{(t+1)} = \arg\max_y p_\theta(y \mid y_1^{(t)}, \ldots, y_{i-1}^{(t)}, x)$$

1. Repeat until convergence (when the sequence stops changing).

In the best case, convergence occurs in one or two iterations because many tokens are "easy" (determined by local context) and converge immediately. The parallel update means all $n$ tokens are processed in each iteration using a single batched forward pass with a modified attention mask.

The theoretical speedup is $O(n / \tau)$ where $\tau$ is the number of Jacobi iterations to convergence. In practice, $\tau$ averages 2–5, giving a 2–5× speedup on sufficiently long generations.

## Lookahead Decoding

**Lookahead decoding** (Fu et al., 2024) combines Jacobi iteration with $n$-gram caching:

- Maintain a **lookahead buffer**: a small set of candidate token sequences generated by running Jacobi iterations one step ahead of the current generation position.
- Maintain an **$n$-gram pool**: a cache of recently verified $(n-1)$-grams and their continuations from the current generation.
- At each step, simultaneously: (1) verify the most promising candidate from the pool, (2) extend the lookahead buffer by one more Jacobi iteration.

When an $n$-gram match is found in the pool, several tokens can be accepted at once without drafting. The pool grows over the course of a single generation, making lookahead decoding more effective on longer sequences.

## Comparison

| Method | Draft Source | Tree/Parallel | Acceptance Rate | Extra Params | Speedup |
| --- | --- | --- | --- | --- | --- |
| Speculative Decoding | Separate small model | Chain | High (if models align) | Full draft model | 2–3× |
| Medusa | Extra MLP heads | Tree | Moderate (65–75%) | ~1% of model | 2–3× |
| EAGLE | Single draft layer | Tree | High (80–90%) | ~1% of model | 3–4× |
| Jacobi | None (fixed-point) | Parallel | Variable | None | 1.5–3× |
| Lookahead | None + n-gram cache | Parallel + cache | High for repetitive text | None (cache only) | 2–4× |

## Deployment Considerations

- **Medusa and EAGLE** require small amounts of additional training (or fine-tuning heads) but are drop-in compatible with existing model weights.
- **Jacobi and Lookahead** require no training and can be applied to any autoregressive model without modification to weights.
- All methods are **lossless** with respect to the target model's distribution when using rejection sampling — the output distribution is identical to standard autoregressive decoding.
- Methods work best on **long generations** (>100 tokens) where the fixed overhead of tree construction is amortized.
- Speedups are more pronounced on **memory-bandwidth-bound** inference (small batch sizes, large models) than on compute-bound inference (large batches where the GPU is already saturated).

## Summary

Self-speculative decoding methods — Medusa, EAGLE, Jacobi, and Lookahead — accelerate LLM inference by generating multiple candidate tokens per forward pass without requiring a separate draft model. Medusa attaches lightweight MLP heads to predict future tokens and verifies them via tree attention; EAGLE improves accuracy by conditioning the draft on hidden states rather than logits; Jacobi decoding reformulates generation as parallel fixed-point iteration; Lookahead caches verified n-grams for future reuse. All methods are lossless — they preserve the target model's output distribution — and achieve 2–4× practical speedups on memory-bandwidth-bound inference workloads.
