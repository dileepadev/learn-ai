---
title: Introduction to TensorRT-LLM
description: Explore NVIDIA TensorRT-LLM, an enterprise-grade inference engine delivering peak throughput with in-flight batching, FP8 GEMMs, KV caching, and multi-GPU tensor parallelism.
---

Deploying Large Language Models (LLMs) in production is computationally challenging. Autoregressive token generation is fundamentally **memory-bandwidth bound**: for every generated token, multi-billion-parameter weight matrices and rapidly growing Key-Value (KV) cache tensors must be fetched from GPU High Bandwidth Memory (HBM) into SRAM/registers. Naive PyTorch implementations often utilize less than 20% of peak GPU compute capacity.

**NVIDIA TensorRT-LLM** is an open-source library that compiles and optimizes LLMs for execution on NVIDIA GPUs (Ampere, Ada Lovelace, Hopper, and Blackwell). TensorRT-LLM combines customized CUDA kernels, high-throughput batching algorithms, multi-GPU parallelism, and native 8-bit/4-bit quantization to achieve **up to $4\times\text{--}8\times$ higher inference throughput** compared to standard Hugging Face runtimes.

---

## The LLM Inference Bottleneck: Prefill vs. Decode

LLM inference consists of two fundamentally distinct computational phases:

```
1. Prefill Phase (Compute-Bound):
Input Prompt Tokens [T_1, T_2, ..., T_512] ──► Massive Matrix Multiplications (GEMMs)
GPU Compute Utilization: HIGH (~80-95%)

2. Decode Phase (Memory-Bandwidth Bound):
Generated Token [T_{n}] ──► Fetches entire multi-gigabyte model weights & KV Cache from HBM
To compute a single vector-matrix product!
GPU Compute Utilization: LOW (<20% in naive implementations)
```

TensorRT-LLM restructures the execution pipeline to saturate GPU tensor cores and maximize memory bandwidth reuse during both phases.

---

## Core Optimization Technologies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TensorRT-LLM Acceleration Stack                                             │
│                                                                             │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌────────────────┐ │
│  │ In-Flight Batching      │  │ Paged KV Cache        │  │ FlashAttention │ │
│  │ (Iteration-Level        │  │ Dynamic non-contiguous│  │ Flash-Decoding │ │
│  │ Scheduling)             │  │ memory blocks         │  │ Fused Kernels  │ │
│  └─────────────────────────┘  └───────────────────────┘  └────────────────┘ │
│                                                                             │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌────────────────┐ │
│  │ FP8 / INT4 Quantization │  │ Multi-GPU Parallelism │  │ Chunked        │ │
│  │ SmoothQuant, AWQ, FP8   │  │ Tensor (TP) &         │  │ Prefill &      │ │
│  │ Hopper/Blackwell GEMMs  │  │ Pipeline (PP)         │  │ Speculative    │ │
│  └─────────────────────────┘  └───────────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. In-Flight (Continuous) Batching
Traditional serving frameworks use static request-level batching: a batch of prompts executes together until the longest sequence completes, leaving GPU threads idle while shorter requests wait.

TensorRT-LLM implements **In-Flight Batching** (iteration-level scheduling):
- As soon as a request emits an `<eos>` (End-of-Sequence) token, it is immediately evicted from the active batch.
- A new incoming request from the queue enters the batch at the very next decoding step.
- Eliminates idle GPU bubbles and increases overall serving throughput by up to $3\times$.

### 2. Paged KV Cache
The Key-Value (KV) cache stores past token keys and values so attention mechanisms avoid redundant computation. In naive setups, GPU memory must be pre-allocated contiguously for the maximum possible sequence length (e.g., 4096 tokens), leading to 60–80% memory fragmentation.

TensorRT-LLM manages the KV cache as **virtual memory pages**:
- Physical memory is allocated in non-contiguous 64-token blocks on demand.
- Virtual-to-physical block tables map memory transparently.
- Memory waste drops to near zero, allowing significantly larger batch sizes.

### 3. Native Quantization (FP8, INT8, INT4 AWQ)
TensorRT-LLM provides hardware-accelerated kernels for modern quantization formats:
- **FP8 (Hopper H100 / Blackwell B200):** Halves weight and activation footprint while maintaining accuracy within $99\%$ of FP16 baselines.
- **INT4 AWQ / GPTQ:** Compresses weights to 4 bits with on-the-fly dequantization in registers during GEMM execution.
- **SmoothQuant (W8A8):** Migrates systematic activation outliers into weights, enabling full INT8 matrix multiplications.

### 4. Tensor and Pipeline Parallelism
For models that exceed a single GPU's VRAM (e.g., LLaMA-70B requires $\approx 140\text{ GB}$ in FP16), TensorRT-LLM distributes layers using high-speed NVLink interconnects:
- **Tensor Parallelism (TP):** Splits linear weight matrices ($W_Q, W_K, W_V, W_O$) across GPUs within a node using Megatron-LM styles.
- **Pipeline Parallelism (PP):** Distributes sequential transformer layers across separate nodes.

---

## Workflow: From Hugging Face to TensorRT-LLM Engine

Building an optimized deployment follows a structured three-step compilation process:

```
Hugging Face PyTorch Model
           │
           ▼
[ Step 1: Weight Conversion ]
Extracts tensors, applies quantization calibrations (FP8 / AWQ)
           │
           ▼
[ Step 2: Engine Build (trtllm-build) ]
Fuses layers (Conv/GEMM/LayerNorm), compiles kernel graphs for target GPU
           │
           ▼
TensorRT-LLM Serialized Engine (.engine)
           │
           ▼
[ Step 3: Production Serving via Triton Inference Server ]
```

### CLI Example: Building an Optimized LLaMA Engine

```bash
# 1. Convert Hugging Face model weights to TensorRT-LLM format with FP8
python3 convert_checkpoint.py \
    --model_dir meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_dir ./tllm_checkpoint_llama3_fp8 \
    --dtype float16 \
    --use_fp8

# 2. Build the optimized engine using trtllm-build
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint_llama3_fp8 \
    --output_dir ./tllm_engine_llama3_fp8 \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --paged_kv_cache enable
```

### Python Runtime Execution

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# Load the compiled engine
runner = ModelRunner.from_dir("./tllm_engine_llama3_fp8")

# Execute high-throughput batched generation
prompt = "Explain in-flight batching in distributed AI systems:"
outputs = runner.generate(
    batch_input_ids=[runner.tokenizer.encode(prompt)],
    max_new_tokens=150,
    end_id=runner.tokenizer.eos_token_id,
    temperature=0.7
)

response_text = runner.tokenizer.decode(outputs[0][0])
print(response_text)
```

---

## Performance Benchmark Comparison

| Metric | Stock PyTorch (Hugging Face) | vLLM Baseline | TensorRT-LLM (FP8 Engine) |
| :--- | :--- | :--- | :--- |
| **Throughput (Tokens / Sec / GPU)** | Baseline ($1\times$) | $2.5\text{--}3.5\times$ | **$4.0\text{--}6.5\times$** |
| **Time-to-First-Token (TTFT)** | $450\text{ ms}$ | $180\text{ ms}$ | **$85\text{ ms}$** |
| **Memory Footprint (LLaMA-70B)** | $>140\text{ GB}$ ($2\times$ A100 80GB) | $70\text{ GB}$ (FP8) | **$38\text{ GB}$ (INT4 AWQ)** |
| **KV Cache Efficiency** | Contiguous (Wasteful) | Paged KV | Hardware-Fused Paged KV |

---

## Key Takeaways

- TensorRT-LLM transforms memory-bound decoding into a compute-saturated pipeline through fused CUDA kernels and hardware-specific compilation.
- In-flight batching and paged KV caching eliminate memory fragmentation and idle execution bubbles.
- Native FP8 support unlocks the full potential of NVIDIA Hopper (H100) and Blackwell (B200) Tensor Cores for enterprise-scale serving.
