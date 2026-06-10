---
title: Speculative Decoding
description: How speculative decoding speeds up LLM inference by using a small draft model to generate candidate tokens verified in parallel by the main model.
---

Speculative decoding is an inference optimization technique that significantly accelerates large language model generation without changing the model's outputs. It exploits the fact that verifying a sequence of tokens is much faster than generating them one at a time.

## The Bottleneck: Autoregressive Generation

Standard LLM generation is sequential — the model generates one token at a time, each requiring a full forward pass through the model. A 70B parameter model doing 1000 forward passes to generate 1000 tokens is slow, even on high-end hardware. The bottleneck is not compute but **memory bandwidth**: loading model weights from GPU memory for every single token.

## The Idea

Speculative decoding uses two models:

1. **Draft model:** A small, fast model (same architecture, much fewer parameters) that generates k candidate tokens quickly.
2. **Target model:** The large, slow model that verifies all k draft tokens in a **single forward pass** (since attention can process a sequence in parallel).

If the target model agrees with a draft token, it is accepted for free. If it disagrees, it rejects that token and all subsequent draft tokens, and generation resumes from the point of disagreement.

## Why It Preserves Output Distribution

The acceptance/rejection criterion is carefully designed so that the final token distribution is mathematically identical to what the target model would have produced alone. This is not an approximation — speculative decoding is lossless.

## Speedup

If the draft model's accuracy on the target model's distribution is high (i.e., the models agree on most tokens), each verification step accepts several draft tokens at once. The practical speedup is typically **2–4×** for tasks where the draft model is reliable (coding, simple Q&A). The speedup depends on the acceptance rate.

## Variants

- **Self-speculative / Medusa:** Use draft heads attached directly to the target model instead of a separate draft model.
- **Lookahead decoding:** Generate draft tokens using n-gram patterns from the input, requiring no draft model at all.
- **Eagle / Eagle-2:** Learns a feature-based draft head that achieves very high acceptance rates.

## Practical Use

Speculative decoding is available in:
- **vLLM** and **TGI (Text Generation Inference)** as a built-in option.
- **llama.cpp** with `--draft-model`.
- Hugging Face `transformers` via `assisted_generation`.

It is especially beneficial for interactive applications where latency matters — chatbots, coding assistants, and real-time translation.
