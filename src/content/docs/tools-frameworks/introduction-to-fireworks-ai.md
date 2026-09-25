---
title: Introduction to Fireworks AI - Fast Inference and Fine-Tuning for Open Models
description: Learn how Fireworks AI provides optimized hosting, fine-tuning, and compound AI tooling for open-weight LLMs and other models.
---

Fireworks AI is an inference platform for open-weight models, offering optimized serving, fine-tuning, and deployment tooling aimed at teams that want open-model flexibility without operating their own GPU infrastructure.

## Serving Open Models

Fireworks hosts a broad catalog of open-weight LLMs, image models, and embedding models behind a single API, with proprietary inference optimizations (custom CUDA kernels, quantization, and speculative decoding techniques) aimed at reducing latency and cost relative to running the same open-weight checkpoint on unoptimized infrastructure:

```python
import fireworks.client

fireworks.client.api_key = "..."
response = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    messages=[{"role": "user", "content": "Explain vector quantization briefly."}]
)
```

## Fine-Tuning and Model Deployment

Fireworks supports fine-tuning open models (including LoRA-based fine-tuning) directly on the platform and deploying the resulting adapters or merged models for inference without managing training infrastructure separately from serving infrastructure. This closes the loop from "fine-tune on our data" to "serve in production" within one platform rather than stitching together separate training and serving providers.

## FireFunction and Structured Output

Fireworks offers models specifically tuned for reliable function calling and structured JSON output, addressing a common pain point with open-weight models, which historically followed function-calling schemas less reliably than proprietary frontier models.

## Practical Guidance

Choose a platform like Fireworks when you want the cost and customization benefits of open-weight models — fine-tuning on proprietary data, avoiding per-token pricing of closed frontier models at scale — without taking on GPU procurement and inference optimization work yourself. Compare inference cost and latency against both self-hosting (via vLLM or similar) and closed-model APIs for your specific traffic pattern, since the right choice depends heavily on request volume and latency requirements.
