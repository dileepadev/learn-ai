---
title: AI Inference Hardware
description: Understand the hardware landscape for AI inference — GPUs, TPUs, NPUs, and purpose-built accelerators — how they differ architecturally, and how to choose the right hardware for LLM and deep learning workloads.
---

**AI inference hardware** refers to the processors and accelerators purpose-built or optimized for running trained AI models at production scale. While training a model requires powerful hardware over days or weeks, inference runs continuously — handling millions of requests per day. Hardware choice has a direct impact on cost, latency, and throughput.

## Why Dedicated Hardware Matters

Neural networks are dominated by one operation: **matrix multiplication**. A standard forward pass through a transformer layer involves:

$$Y = XW^\top$$

Where $X \in \mathbb{R}^{B \times d}$ (batch × hidden dim) and $W \in \mathbb{R}^{d_{out} \times d}$. This is a dense GEMM (General Matrix Multiply) operation that must be performed thousands of times per inference call.

General-purpose CPUs are designed for low-latency, sequential, branchy code — not for the massively parallel matrix arithmetic that AI demands. Dedicated accelerators achieve 10–1000× better performance per watt for AI workloads.

## GPUs: Graphics Processing Units

GPUs were originally designed for real-time 3D graphics — which also requires massively parallel matrix math. NVIDIA recognized this overlap early and built the CUDA programming model, making GPUs the dominant AI hardware platform.

### GPU Architecture

| Component | Role |
|---|---|
| **CUDA Cores / Shader Units** | General-purpose parallel floating-point compute |
| **Tensor Cores** | Dedicated matrix multiply-accumulate units (GEMM acceleration) |
| **High-Bandwidth Memory (HBM)** | Ultra-wide memory bus (up to 3.35 TB/s on H200) |
| **NVLink / NVSwitch** | High-bandwidth GPU-to-GPU interconnect |
| **L2 Cache / Shared Memory** | Fast on-chip memory for register spilling and tiling |

**Tensor Cores** are the key innovation: they perform a $4 \times 4$ matrix multiply-accumulate in a single clock cycle in various precisions (FP32, BF16, FP16, INT8, FP8).

### Key NVIDIA GPU Generations

| GPU | Year | HBM Memory | Memory Bandwidth | Key Feature |
|---|---|---|---|---|
| **A100** | 2020 | 40/80 GB | 2 TB/s | First PCIe/SXM, MIG support |
| **H100** | 2022 | 80 GB | 3.35 TB/s | Transformer Engine, FP8 |
| **H200** | 2024 | 141 GB HBM3e | 4.8 TB/s | Larger memory for larger models |
| **B200** | 2025 | 192 GB HBM3e | 8 TB/s | Blackwell arch, FP4 support |
| **GB200 NVL72** | 2025 | 13.5 TB total | Rack-scale | 72 GPUs unified memory pool |

**Memory bandwidth** is often the binding constraint for LLM inference — moving model weights from HBM to compute units is the bottleneck, not the computation itself.

### Consumer vs. Data Center GPUs

Consumer GPUs (RTX 4090, RTX 5090) have less memory bandwidth, no HBM, and lack enterprise reliability features but are cost-effective for development and small-scale inference. The RTX 4090 (24 GB GDDR6X) can run 7B–13B parameter models locally.

## TPUs: Tensor Processing Units

**TPUs** are Google's custom ASIC (Application-Specific Integrated Circuit) designed specifically for neural network workloads. They are available exclusively through Google Cloud and are used internally for training and serving Google's AI products.

### TPU Architecture

TPUs are built around a **Systolic Array** — a grid of multiply-accumulate units that passes data through in a wave, eliminating the need to load/store intermediate results to memory.

The systolic array computes:

$$C = A \cdot B$$

by streaming $A$ row-wise and $B$ column-wise through the grid simultaneously, accumulating partial sums as they flow. This is extremely efficient for GEMM but less flexible than GPU compute.

### TPU v5p and v5e (2024)

| Spec | TPU v5p | TPU v5e |
|---|---|---|
| **Use Case** | Training, large inference | Cost-optimized inference |
| **FLOPS (BF16)** | 459 TFLOPS | 197 TFLOPS |
| **HBM** | 95 GB | 16 GB |
| **Interconnect** | ICI (high-bandwidth) | ICI |

**TPU pods** interconnect thousands of chips into a single logical accelerator, enabling training and inference at scales impossible with GPU clusters of equivalent size.

## NPUs: Neural Processing Units

**NPUs** are low-power AI accelerators integrated into consumer devices — smartphones, laptops, and edge devices. Their goal is **on-device inference**: running AI models locally without cloud connectivity.

### NPU Examples

| Chip | Device | NPU Performance |
|---|---|---|
| **Apple Neural Engine** | iPhone 15 Pro, M4 Mac | 38 TOPS (M4) |
| **Qualcomm Hexagon** | Snapdragon 8 Gen 3 | 45 TOPS |
| **Intel NPU** | Core Ultra (Meteor Lake) | 34 TOPS |
| **AMD XDNA 2** | Ryzen AI 300 | 50 TOPS |

NPUs are optimized for INT8/INT4 quantized inference and excel at models that fit in device memory (typically 1–8 GB for mobile NPUs).

**TOPS** (Tera Operations Per Second) measures INT8 throughput, which is the primary metric for NPU comparisons.

## Purpose-Built AI Chips

Several companies have designed chips specifically for LLM inference:

### Groq LPU (Language Processing Unit)

Groq's LPU uses a **deterministic, compiler-scheduled** architecture with no caches or branch predictors. All memory accesses are statically scheduled at compile time, eliminating the memory latency variability that degrades GPU LLM throughput.

Result: **extremely high token throughput** — Groq systems have demonstrated 500+ tokens/second on Llama models vs. 50–100 tokens/second on equivalent GPU setups.

### AWS Trainium and Inferentia

- **AWS Trainium 2**: Amazon's custom training chip — up to 3.5× better price-performance than GPU instances for training.
- **AWS Inferentia 2**: Inference-optimized; up to 4× lower latency and 10× higher throughput vs. comparable GPU instances for deployed models.

### Cerebras Wafer-Scale Engine

Cerebras builds chips at wafer scale — a single chip the size of an entire silicon wafer (46,225 mm², vs ~800 mm² for an H100). This provides:

- 900,000 cores.
- 44 GB on-chip SRAM (no off-chip memory bottleneck).
- 20 PB/s memory bandwidth.

The elimination of off-chip memory is particularly valuable for LLM inference, where weight loading is the primary bottleneck.

## Memory: The Critical Bottleneck

For LLM inference, **memory capacity and bandwidth** matter more than raw FLOPS:

- A 70B parameter model in FP16 requires ~140 GB of memory just to load weights.
- Each forward pass requires reading all weights from HBM to compute units.
- At 2 TB/s bandwidth, reading 140 GB takes ~70ms — limiting throughput even if compute is idle.

**Strategies to address the memory wall:**

- **Quantization** (INT8, INT4): Halves or quarters weight size.
- **Continuous batching**: Batch multiple requests to amortize weight reads.
- **Speculative decoding**: Use a small draft model to propose tokens, reducing large model calls.
- **KV cache management**: Efficiently manage the KV cache to maximize GPU memory utilization.

## Choosing Hardware for AI Workloads

| Scenario | Recommended Hardware |
|---|---|
| **LLM training (frontier)** | NVIDIA H100/H200, Google TPU v5p |
| **LLM inference (cloud)** | NVIDIA H100/A100, AWS Inferentia2, Groq LPU |
| **LLM inference (on-device)** | Apple M-series (Mac), Qualcomm Snapdragon NPU |
| **Fine-tuning (small to medium)** | NVIDIA A100, RTX 4090 (for LoRA) |
| **Edge / IoT inference** | Raspberry Pi + Coral TPU, NVIDIA Jetson Orin |
| **Cost-optimized cloud inference** | AWS Inferentia2, Google TPU v5e |

## Further Reading

- [NVIDIA H100 Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [Google TPU v5p Cloud Documentation](https://cloud.google.com/tpu/docs/v5p)
- [Groq LPU Architecture Overview](https://groq.com/technology/)
- [Flash Attention: Fast and Memory-Efficient Attention — Dao et al., 2022](https://arxiv.org/abs/2205.14135)
