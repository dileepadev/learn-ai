---
title: Introduction to Groq - Purpose-Built Hardware for Fast LLM Inference
description: Learn how Groq's Language Processing Unit architecture delivers extremely low-latency LLM inference, and where that speed matters most.
---

Groq builds custom hardware, the Language Processing Unit (LPU), specifically designed for the sequential, memory-bandwidth-bound nature of LLM token generation, and offers a cloud API (GroqCloud) that runs open models on this hardware at very high token throughput.

## Why Standard GPUs Aren't Optimal for Inference

GPUs are optimized for the massively parallel matrix multiplications that dominate training, but autoregressive LLM inference generates tokens sequentially, one at a time, which is fundamentally memory-bandwidth-limited rather than compute-limited — the bottleneck is moving model weights to compute units fast enough for each token, not raw floating-point throughput. Groq's LPU architecture is designed around this specific bottleneck, using a deterministic, compiler-scheduled execution model rather than the dynamic scheduling GPUs use.

## Using the Groq API

GroqCloud exposes an OpenAI-compatible API, so switching an existing application from another provider often requires only changing the base URL and model name:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="GROQ_API_KEY"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Summarize this in one sentence: ..."}]
)
```

## Where Inference Speed Matters Most

Extremely low per-token latency matters most for interactive, latency-sensitive applications — voice agents that need sub-second round trips to feel conversational, real-time coding assistants, and agent pipelines that chain many LLM calls where each call's latency compounds across the chain. For batch or offline workloads where total throughput per dollar matters more than per-request latency, the tradeoff calculus is different.

## Practical Guidance

Benchmark Groq against your actual latency-sensitive workload rather than trusting published throughput numbers alone — real-world latency also depends on prompt length, output length, and network round trip. Groq currently hosts a curated set of open-weight models rather than every model available from every provider, so confirm your required model is available before committing a latency-critical path to it.
