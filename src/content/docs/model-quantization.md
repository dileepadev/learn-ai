---
title: "Quantization: Making Large Models Run on Small Hardware"
description: "Understanding the techniques used to compress AI models by reducing the precision of their numerical weights."
---

To run a massive model like Llama-3 or GPT-4, you typically need thousands of dollars in high-end GPUs. Quantization is the magical technique that allows these models to run on consumer-grade hardware, including laptops and even some smartphones.

## What is Quantization?

AI models consist of billions of "weights," which are typically stored as high-precision floating-point numbers (FP32 or FP16). Quantization reduces the precision of these numbers (e.g., to 8-bit, 4-bit, or even 1.5-bit integers).

## Why It Works

While reducing precision might seem like it would destroy the model's intelligence, neural networks are surprisingly resilient. A 4-bit version of a model often retains over 95% of the original's capabilities while requiring 75% less memory.

## Key Benefits

1. **Reduced Memory Footprint:** A 70B parameter model that requires 140GB of VRAM in FP16 can fit into ~40GB when quantized to 4-bit.
2. **Faster Inference:** Lower-precision math is computationally cheaper and faster for hardware to process.
3. **Local Deployment:** Enables "Edge AI," where sensitive data never has to leave the user's device.

## Popular Formats

- **GGUF (used with llama.cpp):** Highly optimized for CPU and Apple Silicon execution.
- **EXL2:** Optimized for high-speed GPU inference.
- **AWQ / GPTQ:** Common formats for server-side deployment on NVIDIA hardware.
