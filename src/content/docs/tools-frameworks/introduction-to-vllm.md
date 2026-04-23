---
title: Introduction to vLLM
description: Learn how vLLM accelerates LLM inference through PagedAttention, continuous batching, and an OpenAI-compatible API — enabling high-throughput, memory-efficient serving of large language models in production.
---

**vLLM** is an open-source library for fast and memory-efficient large language model inference and serving. Developed at UC Berkeley and released in 2023, vLLM introduced **PagedAttention** — a novel memory management algorithm that dramatically increases GPU memory utilization during inference — enabling significantly higher throughput than previous serving frameworks while maintaining low latency.

vLLM has become one of the most widely adopted LLM serving frameworks, used by organizations ranging from startups to hyperscalers. It supports a growing list of model architectures, provides an OpenAI-compatible REST API out of the box, and integrates with the broader LLM ecosystem including Hugging Face, LangChain, and LlamaIndex.

## The KV Cache Problem

To understand vLLM's innovations, it helps to understand why LLM inference is memory-constrained.

During **autoregressive generation**, a language model generates one token at a time. For efficiency, the **key-value (KV) cache** stores the attention keys and values computed for all previous tokens, so they don't need to be recomputed at each new token position.

The KV cache grows linearly with sequence length and batch size:

$$\text{KV cache size} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times L_{seq} \times B_{batch} \times \text{bytes per element}$$

For a LLaMA-2 13B model with a 4,096 token context, generating a single sequence requires approximately 3 GB of GPU memory for the KV cache alone. Processing a batch of 8 sequences requires 24 GB — leaving little room for the model weights (26 GB in FP16).

**Memory fragmentation** compounds the problem: since different sequences in a batch have different lengths, naive memory management wastes significant GPU memory. Prior systems either padded all sequences to a fixed maximum length (wasting memory) or reserved contiguous memory blocks per sequence (causing fragmentation).

## PagedAttention

**PagedAttention** (Kwon et al., 2023) solves the KV cache memory problem by borrowing the concept of **virtual memory paging** from operating systems.

### How PagedAttention Works

Rather than allocating a contiguous block of GPU memory for each sequence's KV cache, PagedAttention divides the KV cache into fixed-size **pages** (blocks) of tokens:

- Each page holds the KV cache for a fixed number of tokens (typically 16 or 32 tokens per page).
- Pages are allocated on demand as sequences grow — no upfront memory reservation.
- Pages for a single sequence need not be contiguous in physical GPU memory; a **block table** maps logical token positions to physical page addresses.
- When a sequence terminates, its pages are immediately freed and returned to the pool for reuse.

This design has two key benefits:

1. **Near-zero internal fragmentation**: Each page is nearly fully utilized, because pages are filled token-by-token.
2. **Near-zero external fragmentation**: Pages can be allocated from anywhere in GPU memory, avoiding the need for large contiguous blocks.

PagedAttention achieves 4–5% memory waste (from partially filled pages at sequence ends), compared to 60–80% waste in prior systems.

### Memory Sharing for Parallel Sampling

PagedAttention enables **physical memory sharing** between sequences that share common prefixes:

- **System prompt sharing**: When many requests share the same system prompt, their KV cache pages for the prompt are shared in physical memory — requiring the pages to be written only once regardless of how many concurrent requests share that prefix.
- **Beam search and parallel sampling**: When generating multiple candidate sequences from the same prefix (beam search, parallel decoding), the divergent portion of each sequence has its own pages, while the common prefix pages are shared. Pages are copy-on-write: shared until a sequence writes to them, then copied.

This sharing can reduce KV cache memory consumption by 55% for beam search scenarios.

## Continuous Batching

Traditional LLM serving batches requests by **static batching**: grouping requests into a batch, processing the entire batch until all sequences complete, then starting the next batch. This is inefficient because:

- Short sequences in the batch complete early, leaving their GPU resources idle while longer sequences continue.
- New requests that arrive while a batch is running must wait until the entire batch completes.

**Continuous batching** (also called iteration-level scheduling or dynamic batching) operates at the **token level** rather than the request level:

- At each decoding step, completed sequences are immediately ejected from the batch.
- New requests waiting in the queue are added to the batch at the same step.
- The batch size dynamically adjusts, keeping GPU utilization high.

This reduces average request latency and dramatically increases throughput — vLLM achieves 2–24× higher throughput than Hugging Face Transformers with comparable latency, depending on request patterns.

## Getting Started with vLLM

### Installation

```bash
pip install vllm
```

vLLM requires CUDA-capable GPUs. For specific CUDA versions or ROCm (AMD) support, refer to the vLLM documentation for the appropriate installation command.

### Offline Batch Inference

For processing a fixed set of prompts without a server:

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# Generate completions
prompts = [
    "Explain the theory of relativity in simple terms.",
    "What is the difference between machine learning and deep learning?",
    "Write a Python function to compute Fibonacci numbers.",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated: {generated_text!r}\n")
```

### OpenAI-Compatible API Server

vLLM's API server is compatible with the OpenAI API, enabling drop-in replacement of OpenAI API calls with locally hosted or cloud-hosted open models:

```bash
# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2
```

Once running, any OpenAI-compatible client works without modification:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require an API key by default
)

completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is PagedAttention?"},
    ],
    temperature=0.7,
    max_tokens=512,
)

print(completion.choices[0].message.content)
```

## Tensor Parallelism

For models too large for a single GPU, vLLM supports **tensor parallelism** via the `--tensor-parallel-size` flag:

```bash
# Serve a 70B model across 4 GPUs
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16
```

Tensor parallelism in vLLM uses the same approach as Megatron-LM — splitting attention heads and MLP layers across GPUs, with all-reduce operations at each layer boundary.

## Quantization Support

vLLM supports multiple quantization formats for reduced memory footprint and faster inference:

### AWQ (Activation-aware Weight Quantization)

```python
llm = LLM(
    model="TheBloke/Llama-2-13B-AWQ",
    quantization="awq",
)
```

### GPTQ

```python
llm = LLM(
    model="TheBloke/Llama-2-13B-GPTQ",
    quantization="gptq",
)
```

### FP8

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization="fp8",
    dtype="auto",
)
```

Quantization allows serving models with significantly reduced GPU memory requirements — an AWQ 4-bit quantized LLaMA-3 8B fits in under 5 GB of GPU memory, enabling deployment on consumer GPUs.

## Structured Output and JSON Mode

vLLM integrates with **Outlines** and supports guided generation for structured outputs:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

# JSON schema-constrained generation
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age", "email"]
}

sampling_params = SamplingParams(
    temperature=0.0,
    guided_json=schema,
    max_tokens=200,
)

output = llm.generate(
    ["Extract the contact information: John Smith, 34 years old, john@example.com"],
    sampling_params
)
```

## Prefix Caching

**Automatic Prefix Caching (APC)** in vLLM reuses KV cache from previous requests when new requests share a common prefix. This is particularly valuable for:

- **System prompts**: A shared system prompt is computed once and its KV cache reused across all requests.
- **Multi-turn conversations**: As a conversation grows, the KV cache from previous turns is reused rather than recomputed.
- **RAG applications**: When the same retrieved documents are used across multiple queries, their KV cache is shared.

Enable prefix caching with the `--enable-prefix-caching` flag:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-prefix-caching
```

Prefix caching can reduce time-to-first-token (TTFT) by 80%+ for requests with long shared prefixes, and significantly reduces GPU compute load for high-traffic deployments.

## Speculative Decoding

**Speculative decoding** uses a small draft model to propose multiple tokens, which the larger target model verifies in a single forward pass. vLLM supports speculative decoding as a drop-in throughput enhancement:

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_model="meta-llama/Meta-Llama-3-8B-Instruct",
    num_speculative_tokens=5,
)
```

For typical conversational prompts, speculative decoding achieves 1.5–3× throughput improvement with no change in output quality (the target model accepts or rejects each draft token, maintaining exact equivalence with non-speculative decoding).

## Comparing vLLM to Alternatives

| Feature | vLLM | TGI (HuggingFace) | TensorRT-LLM | Ollama |
| --- | --- | --- | --- | --- |
| Core innovation | PagedAttention + continuous batching | Continuous batching | NVIDIA-specific CUDA kernels | Ease of use |
| Best for | High-throughput production serving | Quick deployment | Maximum NVIDIA GPU performance | Local development |
| Multi-GPU | Tensor + pipeline parallel | Tensor parallel | Full 3D parallel | Limited |
| Quantization | AWQ, GPTQ, FP8, INT8 | GPTQ, AWQ, FP8 | INT8, INT4, FP8 | GGUF (llama.cpp) |
| OpenAI API | Yes | Yes | Yes (via Triton) | Yes |
| Hardware | NVIDIA, AMD (ROCm) | NVIDIA, AMD | NVIDIA only | CPU + GPU |

vLLM is the leading choice for production serving scenarios requiring maximum throughput on NVIDIA hardware, with strong AMD support through the ROCm backend.

## Production Deployment Patterns

**Kubernetes deployment** with vLLM typically uses:

- **Horizontal Pod Autoscaler**: Scaling the number of vLLM replicas based on request queue depth.
- **GPU operator**: Scheduling vLLM pods on GPU nodes.
- **Load balancer**: Routing requests across replicas, with sticky routing for prefix caching efficiency.

**Multi-model serving**: Running multiple model instances on different GPU allocations within the same Kubernetes cluster, routing requests to the appropriate model based on the request's model parameter.

**Observability**: vLLM exposes Prometheus metrics including request throughput, queue depth, KV cache utilization, and token generation rate — enabling capacity planning and SLA monitoring.
