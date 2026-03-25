---
title: Introduction to Model Quantization
description: Learn how to shrink LLMs into smaller, faster versions without significant performance loss.
---

Model quantization is a memory-saving technique for reducing the precision of the numerical weights in a large language model (LLM).

## How Quantization Works

Large models are typically trained in **float32** or **bfloat16**. Quantization maps these values to lower-precision formats like **int8** or **int4**.

1. **Weight Mapping:** Convert continuous high-precision floats into discrete integer values.
2. **De-quantization:** During inference, weights are scaled back to their original range for computation.

## Key Benefits

- **Efficiency:** Drastically reduces the VRAM required to load and run large models.
- **Portability:** Enables running large models (e.g., Llama-3 70B) on consumer-grade GPUs.
- **Latency:** Faster memory reads can speed up token generation in certain workflows.

## Common Formats

- **GGUF:** Popular for CPU/GPU inference with llama.cpp.
- **EXL2:** Optimized specifically for high-performance GPU inference.
- **AWQ:** Activation-aware weight quantization for minimal accuracy loss.
