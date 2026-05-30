---
title: "Speculative Decoding: Faster LLM Inference"
description: "Learn how speculative decoding uses a small draft model to accelerate large language model inference by 2-4x without changing output quality."
---

Autoregressive generation is inherently sequential — each token depends on all previous tokens, making it hard to parallelize. **Speculative decoding** is a technique that breaks this bottleneck by using a small, fast model to draft multiple tokens at once, then verifying them in parallel with the large model.

## The Problem with Autoregressive Decoding

A 70B parameter model generating 500 tokens must run 500 sequential forward passes. Each pass is memory-bandwidth bound, not compute bound — the GPU spends most of its time loading weights, not doing math. This means the GPU is underutilized during generation.

## How Speculative Decoding Works

1. **Draft Phase**: A small "draft model" (e.g., 7B) generates K candidate tokens autoregressively. This is fast because the model is small.
2. **Verify Phase**: The large "target model" runs a single forward pass over the original context plus all K draft tokens simultaneously. This is efficient because it's one batched pass.
3. **Accept/Reject**: The target model's probability distribution is compared to the draft model's. Tokens are accepted or rejected using a rejection sampling scheme that guarantees the output distribution matches the target model exactly.
4. **Correction**: If a draft token is rejected, the target model's corrected token is used and drafting restarts from there.

## Why It's Lossless

The acceptance/rejection criterion is mathematically designed so that the final token distribution is identical to what the target model would have produced alone. There is no quality tradeoff — only a speed improvement.

## Speedup in Practice

Speedup depends on the **acceptance rate** — how often the draft model's tokens match the target model's distribution. For tasks where the draft model is reasonably accurate (e.g., code completion, factual recall), acceptance rates of 70–90% are common, yielding **2–4x throughput improvements**.

## Variants

- **Self-Speculative Decoding**: Uses early exit layers of the same model as the draft, avoiding the need for a separate model.
- **Medusa**: Adds multiple decoding heads to the target model to predict several future tokens simultaneously.
- **EAGLE**: Uses a lightweight feature-level draft model that conditions on the target model's hidden states for higher acceptance rates.

## When to Use It

Speculative decoding is most effective when:
- Latency (time to complete a response) matters more than throughput.
- A good draft model exists for your target model family.
- The task has predictable token patterns (code, structured output, continuation).

It's now a standard optimization in production inference stacks like vLLM and TGI.
