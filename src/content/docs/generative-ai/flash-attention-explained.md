---
title: Introduction to Flash Attention
description: Learn how Flash Attention optimizes Transformer performance by reducing memory reads/writes.
---

Flash Attention is a fast and memory-efficient algorithm for computing exact attention in Transformers, designed to address the memory bottleneck of standard attention.

## The Problem: Memory Bound

Standard attention has $O(N^2)$ time and space complexity relative to sequence length $N$. However, on modern GPUs, the bottleneck isn't usually computation (FLOPS) but memory access (reading and writing the large attention matrix).

## How Flash Attention Works

Flash Attention uses **tiling** to process the attention matrix in small blocks that fit into the GPU's fast SRAM, rather than the slower HBM (High Bandwidth Memory).

1. **Tiling:** Breaks the Query, Key, and Value matrices into blocks.
2. **Recomputation:** Avoids storing the large $N \times N$ attention matrix by recomputing parts of it during the backward pass.
3. **IO-Awareness:** Minimizes the number of times data is read from or written to HBM.

## Benefits

- **Speed:** Up to 3x faster training for Transformers.
- **Context Window:** Enables much longer sequence lengths (e.g., 16k, 32k+) on the same hardware.
- **Exactness:** Unlike "sparse" attention, Flash Attention provides the exact same result as standard attention.
