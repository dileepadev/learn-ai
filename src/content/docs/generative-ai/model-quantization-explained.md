---
title: Introduction to Model Quantization
description: How to make AI models run faster on less hardware.
---

Model quantization is a model optimization technique that converts the weights and activations of a model from high-precision formats (like 32-bit floating point) to lower-precision ones (like 8-bit integers).

## Why Quantize Models?

Quantization allows us to run large language models on devices with limited hardware resources (like mobile phones or edge devices) without significantly sacrificing accuracy.

### Key Benefits

1. **Reduced Model Size:** Quantizing from 32-bit to 8-bit can reduce the model's footprint by 75%.
2. **Reduced Memory Bandwidth:** Lower precision numbers require less data to be moved between the CPU/GPU and system memory.
3. **Faster Inference:** Integer arithmetic is often faster than floating-point math on modern hardware.

## How Large Language Models use Quantization

Large Language Models (LLMs) often use specialized quantization techniques:

- **PTQ (Post-Training Quantization):** Applying quantization after the model has already been trained.
- **QAT (Quantization-Aware Training):** Incorporating quantization into the training process itself for better accuracy.
- **Auto-GPTQ and GGUF:** Popular libraries and formats for quantizing and running LLMs on consumer hardware.

## Use Cases

- **Mobile AI Applications:** Running on-device LLMs or image generators.
- **Edge Computing:** Deploying AI on IoT devices or small servers.
- **Cost Reduction:** Lowering the hardware cost of running AI services in the cloud.
