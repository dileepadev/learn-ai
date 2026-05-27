---
title: "LLM Inference Optimization: Making Models Faster and Cheaper"
description: "Learn practical techniques for optimizing LLM inference — from batching and KV cache sharing to continuous batching, prefix caching, and speculative decoding."
---

Deploying LLMs in production requires squeezing every bit of performance from your infrastructure. Inference optimization can reduce latency by 10× and costs by 5× with the right techniques. This guide covers the most impactful optimizations for production workloads.

## The Inference Bottlenecks

LLM inference has two distinct phases with different bottlenecks:

### Prefill Phase
- **What**: Process the prompt tokens in parallel.
- **Bottleneck**: Compute-bound. The model must process all prompt tokens through the full transformer.
- **Optimization**: Increase batch size (more prompts per batch), use faster kernels (FlashAttention).

### Decode Phase
- **What**: Generate tokens one at a time, each depending on all previous tokens.
- **Bottleneck**: Memory-bound. The model spends most time loading weights, not computing.
- **Optimization**: Reduce memory bandwidth (quantization, KV cache optimization), increase batch sizes to amortize weight loading.

## Batching Strategies

### Naive Batching
Queue requests and process them in batches. Simple but inefficient — all requests in a batch must finish before any can return.

### Continuous Batching (Dynamic Batching)
Process requests in a batch but allow completions to finish at different times. When one request finishes, slot its resources for a new request. This dramatically improves throughput for variable-length requests.

```
Batch at time T:
[A complete, B 50%, C 75%, D 25%]

After A completes and D starts:
[B 50%, C 75%, D 10%, E 25%]
```

Most production inference servers (vLLM, TensorRT-LLM, SGLang) use continuous batching by default.

### Preemptive Batching
Prioritize latency-sensitive requests by evicting lower-priority requests from the batch. Critical for interactive applications.

## KV Cache Optimization

The KV cache stores key-value pairs from attention for all tokens in the context. It dominates memory usage during long-context inference.

### PagedAttention (vLLM)
Memory is allocated in fixed-size pages, like virtual memory. This eliminates memory fragmentation and allows more efficient sharing of KV cache across requests with shared prefixes (like system prompts).

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-70b-chat-hf")

outputs = llm.generate(
    prompts=["Explain quantum mechanics..."],
    sampling_params=SamplingParams(max_tokens=1000),
    lora_request=None,  # Optional LoRA adapter
)

# PagedAttention handles KV cache efficiently under the hood
```

### Prefix Caching
Cache the KV cache for shared prompt prefixes (system prompts, few-shot examples). When a new request shares a prefix, reuse the cached KV cache instead of recomputing it.

```
Request 1: [System: You are a helpful assistant. User: What is AI?]
→ Compute and cache KV for "You are a helpful assistant."

Request 2: [System: You are a helpful assistant. User: How does photosynthesis work?]
→ Reuse cached KV for "You are a helpful assistant."
→ Only compute KV for the unique user messages.
```

### KV Cache Quantization
Quantize the KV cache from fp16 to int8 or int4, reducing memory usage by 2–4× with minimal quality loss. Requires careful calibration.

## Model-Level Optimizations

### Speculative Decoding
Use a smaller, faster model to draft multiple tokens, then verify them with the larger model. Covered in detail in the "Speculative Decoding" guide — can achieve 2–4× speedups.

### KV Cache Eviction Policies
For long-running services with many concurrent requests, the KV cache can exhaust memory. Smart eviction policies:
- **LRU (Least Recently Used)**: Evict caches not used recently.
- **Attention-based**: Evict caches with lower attention scores (less "important" tokens).

### Model Parallelism

**Tensor Parallelism**: Split individual layers across GPUs. Good for very large models that don't fit on one GPU.

**Pipeline Parallelism**: Split layers across GPUs, processing different layers in parallel. Simpler but higher latency due to pipeline bubbles.

## Inference Server Comparison

| Server | Best For | Key Features |
|--------|----------|--------------|
| vLLM | General purpose | PagedAttention, continuous batching |
| TensorRT-LLM | NVIDIA GPUs | Optimized kernels, tensor parallelism |
| SGLang | Long context | RadixAttention for prefix caching |
| TGI (Hugging Face) | Ease of use | Containers, OpenAI-compatible API |
| OpenAI-compatible API | Interoperability | Standard API for any backend |

## Cost Optimization

### Model Selection
Don't use GPT-4 or Claude Opus for tasks that work with smaller models. Route simple queries to smaller, cheaper models.

### Output Length Limits
Set reasonable max_tokens limits to prevent runaway generations and control costs.

### Smart Routing
Route queries to the appropriate model:
- Factual lookups → Small, fast model.
- Complex reasoning → Larger model.
- Structured extraction → Model with function calling.

### Spot/Preemptible Instances
Cloud providers offer 60–90% discounts on interrupted-capable instances. Use checkpointing for training and redundant serving capacity for fault tolerance.

## Measurement and Profiling

Track these metrics to identify bottlenecks:

```python
# Per-request metrics from vLLM
{
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "time_to_first_token": 0.05,    # Prefill time
    "time_per_output_token": 0.01,  # Decode time per token
    "total_latency": 5.5,           # 0.05 + 500 × 0.01
    "gpu_memory_usage": 0.85,       # 85% of GPU memory
}
```

- **High time_to_first_token**: Prefill bottleneck → increase batch size or reduce prompt length.
- **High time_per_output_token**: Decode bottleneck → more efficient kernels, prefix caching.
- **High gpu_memory_usage**: Memory bottleneck → quantization, KV cache eviction.

Inference optimization is where production LLM systems spend most of their engineering effort. The techniques here — continuous batching, PagedAttention, prefix caching, and speculative decoding — are the foundations of cost-effective, low-latency LLM serving.