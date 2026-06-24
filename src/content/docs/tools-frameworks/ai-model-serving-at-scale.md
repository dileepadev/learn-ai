---
title: AI Model Serving at Scale
description: A comprehensive guide to serving large language models and AI models in production — covering inference optimization, batching strategies, KV cache management, tensor parallelism, hardware selection, latency-throughput tradeoffs, and the architecture of modern model serving systems like vLLM, TensorRT-LLM, and SGLang.
---

**AI model serving** is the engineering discipline of deploying trained models to handle real-time or batch inference requests efficiently, reliably, and at the scale demanded by production traffic. For large language models in particular, serving is a distinct and challenging engineering domain — the cost, latency, and throughput characteristics of LLM inference are fundamentally different from those of classical ML models, and specialized infrastructure is required to serve them economically.

This article covers the full stack of modern LLM serving: from the hardware to the inference kernel to the serving framework to the deployment architecture.

## The LLM Inference Problem

LLM inference has two distinct phases with very different computational characteristics:

### Prefill (Prompt Processing)

The model processes the **entire input prompt** in a single forward pass. For a prompt of length $L$:
- All $L$ tokens are processed in **parallel** through the transformer
- This is **compute-bound**: the GPU's matrix multiply units are fully utilized
- Time scales with prompt length and model size
- Produces the **KV cache** — key-value tensors for each layer, used in subsequent generation

### Decode (Token Generation)

The model generates **one token at a time** autoregressively. For each new token:
- The new token's query attends over all previously generated keys and values (stored in KV cache)
- This is **memory-bandwidth-bound**: reading the full KV cache and model weights dominates cost, not computation
- Time is approximately constant per token (independent of sequence length if the KV cache fits in memory)
- Throughput is limited by GPU memory bandwidth, not compute

The prefill-decode distinction is crucial for understanding serving system design. A single A100 GPU has:
- ~312 TFLOPS of BF16 compute
- ~2 TB/s of HBM memory bandwidth

A Llama-3 70B model has ~140GB of parameters. Reading them once at BF16 costs 70ns at peak bandwidth — meaning a **single decode step is memory-bandwidth limited** at ~140GB / 2TB/s ≈ 70ms of bandwidth-limited time (much less than actual step time due to other overheads, but illustrating the bottleneck).

## Batching Strategies

### Static Batching

Process a fixed batch of $B$ requests together. All requests in a batch are processed synchronously — when the longest request finishes, all requests are returned.

**Problem**: LLM outputs have highly variable lengths. If one request in a batch of 8 generates 1000 tokens while others generate 50, the 7 short requests wait idle for the long one — wasting GPU cycles.

### Continuous Batching (Iteration-Level Batching)

**Continuous batching** (Orca, Yu et al., 2022) processes requests at the **token level** rather than the request level. At each decode step:
- Completed requests are immediately removed from the batch
- New requests are immediately added to fill the slot
- The batch size is approximately constant across steps

This dramatically improves GPU utilization — no idle waiting for long requests. Continuous batching is now the standard in all production LLM serving systems and yields 10–23× throughput improvement over static batching on realistic workloads.

### Chunked Prefill

**Chunked prefill** (Agrawal et al., 2023) breaks long prompt processing into chunks, interleaving prefill chunks with decode steps. This prevents a single long prompt from monopolizing the GPU and blocking decode steps for other requests — reducing tail latency for concurrent requests.

### Speculative Decoding in Serving

**Speculative decoding** uses a small **draft model** to generate $k$ candidate tokens, then verifies all $k$ in parallel with the large **target model** in a single forward pass:

- If all $k$ draft tokens are accepted, we generate $k$ tokens in the time of one target model step
- If draft tokens are rejected, the generation is equivalent to standard decoding

The speedup is **lossless** — the output distribution is identical to standard sampling. Speedup factors of 2–4× are typical when the draft model has >70% acceptance rate. Optimal for latency-sensitive applications.

## KV Cache Management

The **KV cache** stores key and value tensors for all previously generated tokens. For a batch of requests with long sequences, KV cache memory consumption can dominate:

$$\text{KV cache size} = 2 \times \text{n\_layers} \times \text{d\_head} \times \text{n\_heads} \times L \times \text{precision}$$

For Llama-3 70B (80 layers, 64 KV heads, 128 head dim) at BF16, each token uses $2 \times 80 \times 128 \times 8 \times 2 = 327,680$ bytes ≈ 328KB per token. A context of 8K tokens requires 2.5GB of KV cache — and this must be maintained for each request in the batch simultaneously.

### PagedAttention

**PagedAttention** (Kwon et al., 2023), the core innovation in **vLLM**, manages KV cache memory analogously to OS virtual memory paging:

- KV cache is divided into **fixed-size blocks** (pages)
- Each request's KV cache is stored in non-contiguous blocks
- A **block table** maps logical token positions to physical block addresses
- Blocks are allocated on demand; freed when requests complete

Benefits:
- **Near-zero memory waste**: Classical systems pre-allocate the maximum sequence length, wasting memory for short sequences. PagedAttention uses only what's needed.
- **Memory sharing**: Multiple requests can share the same KV cache blocks for common prefixes (system prompts, few-shot examples) — physically the same memory pages are referenced by multiple requests
- **Flexible scheduling**: Because memory is tracked at page granularity, the scheduler can make fine-grained decisions about which requests to serve

PagedAttention enables **3–4× more requests in memory** simultaneously, dramatically improving throughput.

### Prefix Caching

**Prefix caching** (also called **prompt caching**) stores computed KV cache for shared prefixes across requests. If many requests share the same system prompt or few-shot examples, the KV cache for that prefix is computed once and reused.

All major serving frameworks (vLLM, SGLang, TensorRT-LLM) implement prefix caching. Cache hit rate depends on workload — API providers with many requests using the same system prompt (e.g., a customer service bot) achieve very high hit rates.

## Parallelism for Large Models

Models that don't fit in a single GPU require **model parallelism** during serving.

### Tensor Parallelism

**Tensor parallelism** (Megatron-LM) splits individual weight matrices across GPUs:

- Each GPU holds $1/N$ of each weight matrix
- For a matmul $Y = XW$, each GPU computes $Y_i = XW_i$ and results are all-reduced
- Requires **fast interconnect** (NVLink within a node; InfiniBand across nodes) since each layer requires communication

For a 70B model: 4-way tensor parallelism across 4×A100s (each with 80GB) gives 20B parameters per GPU, fitting within memory. Latency overhead is the all-reduce communication per layer — scales with model depth.

### Pipeline Parallelism

**Pipeline parallelism** assigns different transformer layers to different GPUs (or nodes). Input activations flow through the pipeline:

- GPU 1: layers 1–20
- GPU 2: layers 21–40
- GPU 3: layers 41–60
- GPU 4: layers 61–80

**Bubble overhead**: Without microbatching, GPUs idle while waiting for the pipeline to fill/drain. **Microbatching** fills the pipeline but adds latency — less suited for single-request serving than for batch serving.

### Context Parallelism (Sequence Parallelism)

**Context parallelism** splits the **sequence dimension** across GPUs for very long contexts. Each GPU processes $L/N$ tokens' attention, with cross-GPU communication for global attention. Enables serving 100K+ token contexts that would exceed single-GPU memory.

## Quantization in Serving

Quantization reduces model weights and/or activations to lower precision, reducing memory and improving throughput.

### Weight-Only Quantization

**INT8 / INT4 weight quantization** (GPTQ, AWQ, bitsandbytes) quantizes model weights while keeping activations in float16/bfloat16:
- Model size reduced 2–4× (16-bit → 8-bit or 4-bit)
- Weights are dequantized to float16 before matrix multiplication — no integer arithmetic
- Memory bandwidth savings directly translate to faster decode
- Quality degradation: minimal at INT8; modest at INT4 with careful calibration

**AWQ** (Activation-Aware Weight Quantization) finds the weight quantization that minimizes activation error — achieving near-lossless quality at 4-bit.

### FP8 Inference

**FP8** (8-bit floating point) is natively supported on NVIDIA H100/H200 and Google TPU v4+. Unlike INT4 weight quantization, FP8 supports both weight and activation quantization with hardware-accelerated arithmetic:
- **~2× memory savings** over BF16
- **~2× throughput improvement** from hardware FP8 matrix multiply units
- Minimal quality degradation on most models when using per-tensor or per-channel scaling

FP8 is rapidly becoming the standard for serving frontier models.

## Serving Frameworks

### vLLM

**vLLM** (UC Berkeley) is the most widely used open-source LLM serving framework. Key features:
- PagedAttention for KV cache management
- Continuous batching
- OpenAI-compatible API server
- Speculative decoding support
- Supports most open-weight model architectures
- Tensor parallelism across multiple GPUs/nodes

vLLM is the de facto standard for serving open-weight models (Llama, Mistral, Qwen, etc.).

### TensorRT-LLM

**TensorRT-LLM** (NVIDIA) provides NVIDIA-optimized inference kernels for LLMs:
- Hand-tuned CUDA kernels for every attention pattern and quantization scheme
- Integrated with Triton Inference Server for production deployment
- Highest throughput on NVIDIA hardware at the cost of flexibility
- Used by cloud providers for maximum GPU utilization

### SGLang

**SGLang** (LMSYS) focuses on **structured generation and multi-call efficiency**:
- **RadixAttention**: Extends prefix caching to arbitrary tree-structured shared prefixes, not just fixed system prompts
- **Runtime interpreter**: Batches multiple LLM calls from structured programs, scheduling efficiently across the GPU
- Particularly suited for agentic workloads with many LLM calls and complex output structure

### Ollama

**Ollama** is optimized for **single-user, local deployment** on consumer hardware (MacBooks, gaming PCs with Nvidia RTX). It prioritizes ease of use over production throughput — one-command model downloading and serving. Not suitable for multi-user serving but ideal for developer experimentation.

## Latency vs. Throughput Tradeoffs

These two objectives are in fundamental tension:

- **Low latency** (fast first token, fast per-token): Requires small batch sizes (less to compute per step) and high clock rates. Best served with smaller models or distillation.
- **High throughput** (more tokens/second/dollar): Requires large batch sizes (more requests processed simultaneously), favoring maximum GPU memory utilization.

**SLO-aware scheduling** allocates batch slots dynamically based on:
- Time-to-first-token (TTFT) latency SLO
- Inter-token latency (ITL) SLO
- Request priority
- Current queue depth

### Disaggregated Prefill-Decode

An advanced architecture emerging in 2024: **separate GPU pools** for prefill and decode:

- **Prefill pool**: Compute-optimized GPUs (or CPU offload) handle prompt processing
- **Decode pool**: Memory-bandwidth-optimized GPUs handle token generation

KV cache for a request is transferred from the prefill GPU to the decode GPU after prompt processing. This allows independent scaling and optimization of each phase — critical for workloads with very long prompts or very long generations.

## Hardware Selection

| Hardware | Best For | Notes |
|---|---|---|
| **NVIDIA A100 80GB** | General production | Mature ecosystem, NVLink |
| **NVIDIA H100 80GB** | Frontier models, FP8 | 3× A100 throughput for LLMs |
| **NVIDIA H200 141GB** | Long context, large models | 2× HBM capacity vs. H100 |
| **AMD MI300X 192GB** | Memory-capacity-limited models | Largest HBM pool available |
| **Google TPU v5** | Google Cloud workloads | High throughput, custom chip |
| **Groq LPU** | Ultra-low latency | Memory-bandwidth optimized |

The right hardware depends on **model size** (determines minimum memory), **batch size** (determines compute requirements), and **latency target** (memory bandwidth limited → H200 or MI300X; compute limited → H100).

## Cost Optimization

LLM inference is expensive. Key cost levers:

- **Model selection**: A 7B model is ~10× cheaper to serve than a 70B model per token
- **Quantization**: INT4 or FP8 reduces cost 2–4× with minimal quality loss for most applications
- **Caching**: Prompt caching for shared prefixes can reduce costs 50–90% for applications with shared system prompts
- **Request routing**: Route simple requests to small/cheap models; escalate complex requests to large models (LLM cascades)
- **Batching**: Maximize batch size to amortize fixed costs across more requests
- **Right-sizing hardware**: Match GPU memory and compute to actual workload — don't over-provision

A well-optimized serving stack for a production application can achieve 5–20× better cost-efficiency than a naïve deployment using the same model weights.
