---
title: Flash Attention
description: How Flash Attention rewrites the attention algorithm to be IO-aware, dramatically reducing memory usage and improving speed for training and inference of transformer models.
---

Flash Attention is a hardware-aware implementation of the attention mechanism that achieves the same mathematical result as standard attention but with dramatically lower memory usage and higher speed. It became one of the most practically impactful algorithmic improvements in modern deep learning.

## The Problem with Standard Attention

The self-attention operation at the core of every transformer computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For a sequence of length `N`, computing `QK^T` produces an `N×N` matrix. This has two critical problems:

1. **Quadratic memory:** The `N×N` attention matrix must be stored in GPU memory (HBM). For N=16,384 tokens, that's 16,384² = 268M float32 values — roughly 1GB just for one attention layer in one batch.
2. **Slow memory access:** GPUs are fast at compute but slow at memory transfers. Standard attention repeatedly reads and writes the large intermediate `N×N` matrix to and from HBM, spending far more time on memory I/O than on actual computation.

## The Flash Attention Solution

Flash Attention (introduced by Tri Dao et al., 2022) reformulates attention to avoid materializing the full `N×N` matrix. The key insights:

### IO-Aware Algorithm
Modern GPUs have a small, fast on-chip SRAM (shared memory) and a large, slow HBM (high-bandwidth memory). Flash Attention restructures the computation to:

1. **Tile the input:** Divide the Q, K, and V matrices into small blocks that fit in SRAM.
2. **Compute attention incrementally:** Process one tile at a time in SRAM, maintaining running statistics (the softmax denominator) across tiles.
3. **Write output once:** The output O is written back to HBM only at the end, not after each intermediate step.

This turns attention from a memory-bound operation into a compute-bound one, which is where GPUs excel.

### Online Softmax
Computing softmax across the full sequence is normally needed before applying attention weights to V. Flash Attention uses a numerically stable online softmax algorithm that computes the correct result by processing tiles sequentially, accumulating a running maximum and normalization factor.

### Mathematical Equivalence
Despite the tiling and reordering, Flash Attention produces exactly the same output as standard attention. It is not an approximation.

## Performance Improvements

Flash Attention delivers significant real-world gains:

- **2–4× faster** attention computation in practice.
- **5–20× less HBM memory** for the attention operation, enabling:
  - Longer sequence lengths with the same GPU.
  - Larger batch sizes.
  - Training models that previously ran out of memory.
- **Better GPU utilization:** Memory bandwidth was the bottleneck; removing it lets the GPU's tensor cores run closer to peak utilization.

## Flash Attention 2 and 3

**Flash Attention 2** (2023) improved parallelism across the sequence length dimension and reduced the number of non-GEMM operations, achieving closer to theoretical peak GPU throughput.

**Flash Attention 3** (2024) takes advantage of NVIDIA Hopper architecture features (H100 GPUs) — particularly asynchronous memory copies and WGMMA instructions — to push further toward hardware limits.

## Impact on Modern LLMs

Flash Attention is now the de facto standard attention implementation:

- Used by default in most major model training frameworks (HuggingFace Transformers, PyTorch `scaled_dot_product_attention`, NanoGPT, Megatron-LM).
- Enabled training of models with much longer contexts than previously feasible. GPT-3 used 2K context; modern models routinely use 32K–128K, partially made possible by Flash Attention.
- Available via `torch.nn.functional.scaled_dot_product_attention()` in PyTorch 2.0+, which automatically uses Flash Attention when possible.

## Using Flash Attention in Practice

```python
# PyTorch 2.0+ — uses Flash Attention automatically when inputs are on CUDA
import torch
import torch.nn.functional as F

# [batch, heads, seq_len, head_dim]
q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)

# Flash Attention is used automatically here
out = F.scaled_dot_product_attention(q, k, v)
```

With HuggingFace Transformers:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
```

## Limitations

- Requires inputs in **fp16 or bf16** (not fp32) for peak performance.
- Originally CUDA-only; now supported on AMD ROCm and some other accelerators.
- Not applicable to attention variants that require access to the full `N×N` matrix (e.g., certain relative position encodings).
- On very short sequences (< 128 tokens), standard attention may be faster since the overhead of tiling isn't justified.

Flash Attention is a prime example of how understanding hardware constraints can yield large practical improvements without changing the underlying model architecture.
