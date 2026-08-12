---
title: "Introduction to Cerebras and Wafer-Scale AI Chips"
description: Understand Cerebras Systems and the Wafer-Scale Engine — how breaking the reticle limit produced the largest chips ever built, why this architecture dramatically accelerates large model training, and what it means for the future of AI hardware.
---

Modern AI training is a hardware problem. GPUs were not designed for AI — they were adapted from graphics rendering pipelines. The dominant bottleneck in training large neural networks isn't compute; it's **memory bandwidth and data movement**. Moving activations, weights, and gradients between compute units and memory consumes more time and energy than the matrix multiplications themselves.

Cerebras Systems took a radical approach: instead of optimizing a conventional chip, they asked what a chip designed exclusively for AI from first principles would look like. The answer was the **Wafer-Scale Engine (WSE)** — a processor the size of an entire silicon wafer, with an unprecedented amount of compute, memory, and interconnect on a single die.

## What Is Wafer-Scale Integration?

Traditional semiconductor manufacturing produces chips by placing a rectangular **reticle** (die mask) on a silicon wafer, exposing and etching the circuitry, then slicing the wafer into individual chips. The reticle limit — determined by the size of the lithography stepper's light field — constrains each die to roughly 800mm².

**Wafer-Scale Integration (WSI)** skips the dicing step entirely. The entire 300mm wafer becomes a single chip. Cerebras WSE-3 (2023) spans **57,600 mm²** — 57× the area of a large NVIDIA A100 GPU.

The engineering challenges this creates are immense:
- **Defect tolerance:** A wafer-sized chip will have manufacturing defects. Cerebras developed a redundancy fabric that routes around defective cores automatically
- **Power delivery:** Distributing power across a wafer without voltage droops requires novel power planes
- **Thermal management:** Cooling a wafer uniformly demands custom liquid cooling solutions
- **Reticle-scale interconnects:** Lithography stitching across reticle boundaries at die scale

## Cerebras WSE Architecture

The WSE is not a giant version of a GPU. It is a fundamentally different architecture:

### Fabric of Processing Elements

The WSE consists of hundreds of thousands of small **Processing Elements (PEs)** — simple compute units each with their own small SRAM memory, arranged in a 2D mesh:

| Specification | WSE-1 (2019) | WSE-2 (2021) | WSE-3 (2023) |
|---------------|--------------|--------------|--------------|
| Die area | 46,225 mm² | 46,225 mm² | 57,600 mm² |
| Transistors | 1.2 trillion | 2.6 trillion | 4 trillion |
| AI-optimized cores | 400,000 | 850,000 | 900,000 |
| On-chip SRAM | 18 GB | 40 GB | 44 GB |
| Memory bandwidth | 9 PB/s | 20 PB/s | 21 PB/s |
| Fabric bandwidth | 100 Pb/s | 220 Pb/s | 214 Pb/s |

The critical insight: by placing memory physically adjacent to each compute unit on-chip, the WSE eliminates the **memory bandwidth wall** that GPU designs face when large batches of activations must move repeatedly between HBM memory and compute cores.

### The Memory-Near-Compute Advantage

In a GPU training setup for a large transformer, the gradient update loop looks approximately like:

```
Forward pass:
  Load weights from HBM → Compute → Store activations to HBM
  (repeated for each layer, each batch element)

Backward pass:
  Load activations from HBM → Compute gradients → Store to HBM
  (repeated for each layer)

Weight update:
  Load weights from HBM → Apply gradient → Store weights to HBM
```

Each `Load/Store from HBM` step crosses the PCIe/NVLink bus and the HBM memory bandwidth. For large models, memory bandwidth is consistently the bottleneck, not FLOP throughput.

On the WSE, weights and activations for a portion of the network live in the SRAM attached to the local PE cluster. There is no HBM. Memory latency is nanoseconds rather than microseconds, and bandwidth is 20 PB/s (20,000 TB/s) — compared to ~3.35 TB/s for A100 HBM.

### The 2D Mesh Interconnect

Unlike GPU tensor cores connected by NVLink or PCIe buses, WSE PEs are connected in a 2D torus mesh. Data flows between adjacent PEs at memory bandwidth speeds without any hierarchy of switches, routers, or bus arbiters.

This enables a computation model where the **neural network is mapped directly onto the physical fabric** — each layer or section of a layer resides on a slice of the wafer, and activations flow spatially through the mesh as forward passes propagate.

## Cerebras Software Stack

The hardware alone doesn't help unless you can program it. Cerebras provides a software stack that bridges standard deep learning frameworks:

```
PyTorch / TensorFlow (user code)
        ↓
Cerebras Model Zoo (optimized reference implementations)
        ↓
CS Compiler (graph compilation + PE mapping)
        ↓
Cerebras Runtime (execution on CS-2 hardware)
```

### Using the Cerebras CS-2

From the user perspective, running on a Cerebras CS-2 looks similar to running on a GPU cluster — you write PyTorch, configure the Cerebras runtime, and submit jobs:

```python
import cerebras.pytorch as cstorch
import torch
import torch.nn as nn

# Define a standard PyTorch model
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class SmallLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads=8, d_ff=d_model * 4)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# Cerebras execution context
backend = cstorch.backend("CSX", artifact_dir="./artifacts")

model = SmallLM(vocab_size=32000)
optimizer = cstorch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

# Compile model for CS-2 execution
compiled_model = cstorch.compile(model, backend)

@cstorch.trace
def training_step(batch):
    input_ids, labels = batch
    logits = compiled_model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, 32000), 
        labels.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss

# Training loop (identical to GPU training pattern)
for batch in dataloader:
    loss = training_step(batch)
```

The Cerebras compiler traces the computation graph, maps it onto the PE fabric, and handles all the low-level orchestration.

### The Cerebras Model Zoo

Cerebras maintains an open-source Model Zoo with reference implementations of popular architectures optimized for the CS-2:

- GPT-2, GPT-3, GPT-J, GPT-NeoX
- BERT, RoBERTa
- T5, BLOOM
- LLaMA, Falcon, Mistral
- Diffusion models (experimental)

The Model Zoo provides configuration files that define model size (number of layers, heads, hidden dimensions) — scaling from small research models to frontier-scale architectures.

## Weight Streaming: Beyond On-Chip Memory

The on-chip SRAM of the WSE-3 holds 44 GB — enough for models up to roughly 20B parameters in half precision. But frontier models (GPT-4, LLaMA 70B+) are far larger.

Cerebras solves this with **Weight Streaming**: weights are stored in a large attached memory system and streamed through the WSE fabric layer by layer. Rather than fitting the entire model on-chip at once, the WSE processes one layer at a time, streaming weights in and activations out.

The Cerebras **MemoryX** system (used in their cloud offering) provides up to 2.4 PB of DRAM attached to the wafer via a high-bandwidth interconnect. This enables training models with hundreds of billions or trillions of parameters on a single WSE.

Weight streaming inverts the typical compute-memory tradeoff: the WSE's 214 Pb/s fabric bandwidth far exceeds the weight streaming rate, so the wafer's compute is never starved.

## Cerebras Inference Cloud

Beyond training, Cerebras has positioned its cloud offering for inference on large models. The CS-2's memory bandwidth advantage is, if anything, even more pronounced during inference:

- **Autoregressive decoding** generates one token at a time — extremely low arithmetic intensity (few FLOPs per byte loaded)
- GPU inference is severely bottleneck by HBM bandwidth during token generation
- The WSE's near-compute SRAM allows the full attention KV cache to be retained on-chip, eliminating HBM bandwidth pressure

Cerebras Inference benchmarks have shown **inference speeds of thousands of tokens per second** for large models — significantly faster than A100 or H100 GPU serving for latency-critical applications.

## How WSE Compares to GPU Clusters

Training a large language model on a GPU cluster requires solving a distributed systems problem. With hundreds or thousands of GPUs:

- **Tensor parallelism** splits individual matrices across GPUs
- **Pipeline parallelism** assigns different layers to different GPUs
- **Data parallelism** replicates the model across GPU groups processing different data

Each parallelism strategy requires collective communication operations (all-reduce, all-gather, scatter) that introduce synchronization overhead. At scale, **communication overhead can consume 30–50% of training time**.

The WSE's single-chip architecture avoids inter-node communication:

| Aspect | GPU Cluster (A100) | Cerebras CS-2 |
|--------|-------------------|---------------|
| Memory hierarchy | HBM → NVLink → IB | On-chip SRAM only |
| Parallelism strategy | Complex 3D + ZeRO | Layer pipeline on fabric |
| Communication | All-reduce across GPUs | On-chip mesh (no network) |
| Programming model | Multi-GPU distributed | Single-device (simpler) |
| Per-step overhead | Communication latency | None (same chip) |

The tradeoff: a single CS-2 has less raw FLOP throughput than a large GPU cluster. But for models that fit the WSE training paradigm (i.e., most transformer language models), the elimination of communication overhead can make overall throughput competitive.

## When to Use Cerebras

Cerebras hardware and cloud excels for:

- **Large model training from scratch:** Where memory bandwidth dominates GPU training time
- **Low-latency inference:** Sub-100ms response time for large model deployments
- **Researcher-friendly scaling:** Single-device simplicity vs. complex distributed training
- **Models with many small parameters:** The PE fabric handles sparsity well

GPU clusters are typically better for:
- **Existing GPU-optimized code:** Mature CUDA libraries, well-tuned kernels
- **Models with large batch sizes:** GPUs achieve high utilization with large batches
- **Multi-modal workloads:** Vision, RL, custom ops not in Cerebras Model Zoo
- **Very large-scale distributed training:** Petabyte-scale runs requiring thousands of accelerators

## Getting Started with Cerebras

Cerebras offers cloud access through **Cerebras Inference API** (for language model inference) and their **AI Supercomputer** cloud service (for training).

For inference via the API:

```python
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key="your-api-key")

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain attention mechanisms in one paragraph."},
    ],
    model="llama3.1-8b",  # Or llama3.1-70b for the larger model
)

print(response.choices[0].message.content)
```

The API is compatible with the OpenAI SDK format, making migration straightforward.

## The Broader Context: AI Hardware Innovation

Cerebras represents one bet in a larger hardware innovation wave. The AI hardware landscape now includes:

- **NVIDIA** (H100, H200, Blackwell): Dominant market share, CUDA ecosystem moat
- **Google** (TPU v5): Vertically integrated for TensorFlow/JAX, used for Gemini training
- **AMD** (MI300X): HBM3-equipped GPUs competing on memory capacity
- **Cerebras** (CS-2): Wafer-scale, memory-near-compute
- **Groq** (LPU): SRAM-only, deterministic tensor-streaming for inference
- **Graphcore** (IPU): Bulk synchronous parallel, SRAM-based
- **Tenstorrent** (Grayskull, Wormhole): RISC-V based, open architecture
- **SambaNova** (RDU): Reconfigurable dataflow for flexibility

The diversity reflects a genuine unsettled question: what is the right architecture for AI workloads at scale? The memory wall, communication bottleneck, and power constraints affect everyone, and different architectural bets address different aspects.

Cerebras's wafer-scale bet — eliminate the memory hierarchy by making compute and memory physically adjacent at unprecedented scale — is the most radical departure from conventional thinking, and it has produced the fastest large model inference benchmarks publicly reported.

## Resources

- **Cerebras documentation:** docs.cerebras.net — model zoo, tutorials, API reference
- **CS-2 Technical Papers:** Cerebras has published several papers on WSE architecture and training methodology on arXiv
- **Cerebras Research:** research.cerebras.com — publications on training large models with WSE
- **Cerebras Blog:** cerebras.net/blog — announcements, benchmarks, and technical deep dives
