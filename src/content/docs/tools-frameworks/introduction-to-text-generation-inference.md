---
title: Introduction to Text Generation Inference (TGI)
description: Master Hugging Face's Text Generation Inference (TGI), covering FlashAttention-2, continuous batching, streaming SSE tokens, speculative decoding, and tensor parallelism.
---

Deploying open-weight Large Language Models (LLMs) in production requires an inference engine capable of maximizing GPU hardware saturation while delivering low latency and predictable token streaming.

**Text Generation Inference (TGI)** is Hugging Face’s purpose-built, enterprise-grade solution for serving large language models. Built with a high-performance **Rust web server** and a hardware-optimized **Python gRPC backend**, TGI powers Hugging Face’s Inference Endpoints, HuggingChat, and hundreds of enterprise production environments.

---

## High-Performance Architecture

TGI decouples HTTP request routing from GPU tensor computation to guarantee that web server concurrency never stalls the CUDA kernel execution pipeline:

```
                            [ Client Applications (OpenAI SDK, cURL, Web UI) ]
                                                   │ (HTTP / Server-Sent Events)
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ High-Throughput Rust Router                                                                 │
│ • Validates requests, manages client queues, tokenizes prompts                              │
│ • Executes Continuous (In-Flight) Dynamic Batching across active requests                   │
│ • Streams tokens back to clients via Server-Sent Events (SSE) immediately as they are generated│
└──────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                           │ (Low-Latency gRPC IPC)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Hardware-Optimized Python Backend                                                           │
│ • Fused Attention Kernels: FlashAttention-2 & PagedAttention                                │
│ • Multi-GPU Tensor Parallelism across NVLink (NCCL)                                         │
│ • Hardware Quantization: FP8, AWQ, GPTQ, BitsAndBytes                                       │
│ • Speculative Decoding Engine (Draft model verification)                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Feature Highlights

### 1. Continuous (In-Flight) Batching
Traditional request batching forces all prompts in a batch to wait until the slowest, longest generation completes. TGI implements **iteration-level batching**:
- As soon as a request emits an `<eos>` token, it is immediately removed from the batch.
- A waiting request from the queue is dynamically merged into the batch at the next step.
- Maximizes GPU utilization and eliminates queue starvation.

### 2. Paged Attention & KV Cache Optimization
By integrating PagedAttention, TGI eliminates memory fragmentation in GPU High Bandwidth Memory (HBM). Key-Value caches are partitioned into discrete, non-contiguous virtual memory blocks, allowing models to serve **up to $3\times\text{--}5\times$ more concurrent requests** without running out of memory (OOM).

### 3. Native Quantization Support
TGI features day-0 integration with cutting-edge weight and activation quantization schemes:
- **FP8:** Native Hopper (H100) and Blackwell GEMM execution.
- **AWQ (Activation-aware Weight Quantization):** 4-bit weights with minimal perplexity degradation.
- **EETQ / GPTQ:** Fast INT8/INT4 weight-only quantization kernels.

### 4. Speculative Decoding
TGI accelerates inference by pairing a heavy target model (e.g., LLaMA-3-70B) with a tiny draft model (e.g., LLaMA-3-8B):
- The draft model speculatively generates $K$ candidate tokens quickly.
- The target model verifies all $K$ tokens in a **single forward pass**.
- Delivers a $1.5\times\text{--}2.5\times$ latency speedup with zero loss in output quality.

---

## Deploying TGI with Docker

The fastest way to deploy TGI is via its official pre-compiled Docker container, which bundles all CUDA drivers, FlashAttention-2 kernels, and Rust binaries:

```bash
# Deploy Meta-Llama-3.1-8B-Instruct on a single GPU
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v /data/models:/data \
    -e HF_TOKEN="hf_your_token_here" \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantize bitsandbytes-nf4 \
    --max-input-tokens 2048 \
    --max-total-tokens 4096
```

### Multi-GPU Tensor Parallelism Example
For large models exceeding a single GPU (e.g., 70B parameter models), pass `--num-shard`:

```bash
docker run --gpus all --shm-size 2g -p 8080:80 \
    -v /data/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Meta-Llama-3.1-70B-Instruct \
    --num-shard 4 \
    --quantize fp8
```

---

## Client Integration: OpenAI SDK Compatibility

TGI exposes an **OpenAI-compatible HTTP API** at `/v1`, allowing developers to integrate it as a drop-in replacement into any existing application using the standard `openai` Python package:

```python
from openai import OpenAI

# Point client to the local TGI instance
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed" # TGI handles authentication via reverse proxy
)

# Stream tokens in real time
response = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are an expert systems engineer."},
        {"role": "user", "content": "Explain continuous batching in high-performance inference servers."}
    ],
    stream=True,
    max_tokens=300
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

---

## TGI vs. vLLM vs. Ollama

| Dimension | Hugging Face TGI | vLLM | Ollama |
| :--- | :--- | :--- | :--- |
| **Primary Target** | Enterprise Production Cloud | Research & Scaled Serving | Local Desktop & Prototyping |
| **Server Architecture** | Rust Router + Python Backend | Pure Python (AsyncIO) | Go Router + llama.cpp C++ |
| **FlashAttention-2** | Native Fused Kernels | Native | CPU/Metal/CUDA Kernels |
| **Speculative Decoding**| Built-in Draft Engine | Built-in | Limited |
| **Setup Complexity** | Docker (1 Command) | Pip / Docker | Native Executable Installer |

---

## Key Takeaways

- TGI combines a Rust router with a Python backend to deliver low-latency continuous batching and streaming tokens.
- Native FlashAttention-2, PagedAttention, and quantization formats (FP8, AWQ) maximize GPU memory bandwidth efficiency.
- Seamless compatibility with OpenAI-formatted APIs allows drop-in integration into existing enterprise AI pipelines.
