---
title: Introduction to Speculative Decoding
description: A performance optimization technique that uses a smaller "draft" model to speed up LLM inference.
---

Speculative Decoding is a powerful technique for speeding up Large Language Model (LLM) inference by predicting multiple tokens ahead with a lightweight model before validating them with the larger, primary model.

## Why Inference is Slow

Modern LLMs like GPT-4 or Llama-3 are colossal. Conventional autoregressive decoding generates one token at a time, where each token requires a full forward pass through the massive model. This is often "memory-bound" and slower than the actual computation bottleneck.

## How Speculative Decoding Works

1. **Drafting:** A small, fast "draft" model (e.g., Llama-7B for a Llama-70B target) generates several potential next tokens in parallel.
2. **Verification:** The large "target" model performs a *single* forward pass on the entire draft sequence.
3. **Acceptance:** The target model verifies which of the drafted tokens are correct according to its own probability distribution.

If the draft model is accurate, multiple tokens can be "accepted" in the time it would normally take to generate one.

## Key Benefits

- **Speed:** Can achieve 2-3x speedup with no loss in output quality.
- **Efficiency:** Better utilization of GPU memory bandwidth.
- **Compatibility:** Works with most Transformer-based LLMs without re-training.
