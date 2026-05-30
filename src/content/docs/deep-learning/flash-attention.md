---
title: "FlashAttention: Fast and Memory-Efficient Attention"
description: "Understand how FlashAttention rewrites the attention algorithm to be IO-aware, achieving 2-4x speedups and enabling much longer context windows without changing model outputs."
---

Standard attention is slow and memory-hungry not because of the number of floating-point operations, but because of how it moves data between GPU memory tiers. **FlashAttention** is an exact attention algorithm that reorders computations to minimize memory reads and writes, achieving dramatic speedups with no change to outputs.

## The Memory Bottleneck in Standard Attention

A standard attention implementation:

1. Computes the full N×N attention matrix (N = sequence length).
2. Writes it to HBM (high-bandwidth memory, the main GPU memory).
3. Reads it back to apply softmax.
4. Writes the result back to HBM.
5. Reads it again to compute the weighted sum of values.

For a sequence of length 4096, the attention matrix is 4096×4096 = 16M entries. Reading and writing this repeatedly is the bottleneck — not the math itself.

## The FlashAttention Solution: Tiling

FlashAttention uses **tiling** to compute attention in blocks that fit in SRAM (the fast on-chip memory, ~100x faster than HBM):

1. Divide Q, K, V into blocks.
2. For each block of Q, iterate over blocks of K and V.
3. Compute partial attention scores and maintain running softmax statistics.
4. Accumulate the output without ever materializing the full N×N matrix.

The full attention matrix is never written to HBM. The result is mathematically identical to standard attention.

## Performance Gains

- **2–4x faster** than standard PyTorch attention on A100 GPUs.
- **5–20x less memory** for the attention computation.
- Enables training with sequences 5–10x longer for the same GPU memory budget.

## FlashAttention-2 and FlashAttention-3

**FlashAttention-2** improved parallelism across the sequence dimension and reduced non-matrix-multiply operations, achieving ~2x additional speedup over FA1.

**FlashAttention-3** targets H100 GPUs specifically, exploiting:
- Asynchronous memory copies (TMA instructions).
- FP8 precision support.
- Warp specialization for overlapping compute and memory operations.

FA3 achieves up to 75% of H100 theoretical peak throughput.

## Impact on the Field

FlashAttention has become the default attention implementation in virtually every serious training framework. It's a key enabler of:

- Long-context models (128K, 1M token contexts).
- Efficient training of large models.
- Faster inference for production deployments.

It's a rare example of a systems paper that changed the entire field's training infrastructure within months of publication.
