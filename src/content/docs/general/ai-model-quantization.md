---
title: "AI Model Quantization: Speed vs. Accuracy"
description: "How quantization reduces model size and inference time, and when it's worth the accuracy tradeoff."
---

A full-precision LLM might be 70GB. After quantization, it could be 7GB. This dramatic compression enables running powerful models on consumer hardware, but there's a cost: some accuracy loss.

## Understanding Quantization

Quantization reduces the precision of model weights and activations. Instead of storing numbers as 32-bit floats, you store them as 16-bit, 8-bit, or even 4-bit integers.

**Example:**
- Float32: 3.14159265 (8 bytes)
- Float16: 3.141 (4 bytes, some precision lost)
- Int8: 3 (1 byte, significant precision lost)

## Common Quantization Approaches

1. **Post-Training Quantization (PTQ):** Quantize an already-trained model quickly, no retraining needed
2. **Quantization-Aware Training (QAT):** Train the model with quantization in mind for better accuracy
3. **Mixed Precision:** Keep critical weights in high precision, others in low precision

## Real-World Formats

- **GGML/GGUF:** Popularized by llama.cpp, runs models locally
- **GPTQ:** 4-bit quantization optimized for inference speed
- **AWQ:** Activations-aware quantization, better accuracy at 4-bit
- **Ollama Models:** Pre-quantized models ready to download and run

## Speed and Size Comparison

| Format | Size | Speed | Accuracy |
|--------|------|-------|----------|
| Full (FP32) | 70GB | 1x | 100% |
| Half (FP16) | 35GB | 1.5x | 99.5% |
| GPTQ 4-bit | 7GB | 3-4x | 97-99% |
| GPTQ 3-bit | 5GB | 4-5x | 95-97% |

## When to Quantize

- **Running Locally:** Essential for consumer hardware
- **Cost-Sensitive:** Reduce API costs by using self-hosted quantized models
- **Latency-Critical:** Need faster responses for real-time applications
- **Low-Resource Deployment:** Edge devices, mobile, embedded systems

## When NOT to Quantize

- **High-Accuracy Requirements:** Medical, legal, or safety-critical applications
- **Complex Reasoning:** Heavily quantized models struggle with multi-step logic
- **You have GPU/TPU:** Cloud resources make quantization less necessary