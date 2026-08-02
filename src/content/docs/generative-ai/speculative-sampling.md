---
title: Speculative Sampling for Faster LLM Inference
description: How draft models and verification enable faster token generation without quality loss.
---

Speculative sampling is an optimization technique that dramatically speeds up large language model inference by using a smaller draft model to propose multiple tokens, then verifying them in parallel with the larger target model.

## The Inference Bottleneck

Autoregressive generation is inherently sequential. Each token depends on all previous tokens, limiting parallelization. For a 70B parameter model generating 100 tokens, you need 100 sequential forward passes through the entire model.

## How Speculative Sampling Works

**1. Draft Phase**
A small, fast model (like a 7B model) generates K candidate tokens (typically 4-8) in a single forward pass.

**2. Verification Phase**
The target model (like a 70B model) evaluates all K tokens in one forward pass, accepting or rejecting each.

**3. Acceptance**
Accepted tokens are kept. If token K is rejected, the sequence is truncated, and the target model generates the correct token instead.

## Mathematical Foundation

The acceptance probability follows:

```
P(accept) = min(1, p_target(x) / p_draft(x))
```

Where `p_target` and `p_draft` are the probability distributions from the target and draft models respectively. This guarantees the output distribution matches the target model exactly.

## Performance Gains

- **2-3x speedup** when draft and target models are well-matched
- **No quality loss**: The output distribution is mathematically identical to standard sampling
- **Memory efficient**: Only need to load one extra small model

## Implementation Considerations

**Draft Model Selection**
- Smaller version of same model family (e.g., Llama-7B for Llama-70B)
- Same tokenizer is essential
- Similar training data improves acceptance rates

**Speculation Length**
- K=4-8 tokens works well in practice
- Longer speculation increases potential speedup but also rejection probability
- Adaptive speculation length based on acceptance history improves efficiency

**Tree-Structured Speculation**
- Generate multiple candidate sequences in parallel
- Verify using tree-attention for even higher acceptance rates
- Used in systems like Medusa and SpecTr

## Practical Applications

- Real-time chat systems needing low latency
- Batch inference where throughput matters
- On-device inference with limited compute budget
