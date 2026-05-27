---
title: "LLM Quantization: From fp16 to 4-bit and Beyond"
description: "Understand LLM quantization techniques — including GPTQ, AWQ, GGUF, and fp4/E4M3 formats — and how to achieve 4x memory reduction with minimal quality loss."
---

Quantization reduces the precision of model weights and activations from 32-bit or 16-bit floating-point to lower-precision formats. For LLMs, this is transformative: a 70B model that requires 140GB in fp16 fits in 35GB with 4-bit quantization, enabling inference on consumer GPUs.

## Why Quantize LLMs

- **Memory reduction**: Weights dominate memory usage. 4-bit quantization reduces memory by 4×.
- **Faster inference**: Lower precision arithmetic is faster on most hardware.
- **Lower latency**: Smaller models fit in GPU memory, avoiding slow CPU offloading.
- **Cost reduction**: Fewer GPUs needed for deployment.

The catch: quantization introduces approximation error that can degrade output quality. The art of quantization is minimizing this error.

## Quantization Fundamentals

### Data Types

| Format | Bits per Weight | Memory vs. fp16 | Typical Quality |
|--------|----------------|-----------------|-----------------|
| fp32 | 32 | 2× | Reference |
| fp16 | 16 | 1× | Reference for modern models |
| bf16 | 16 | 1× | Better numerical range than fp16 |
| int8 | 8 | 0.5× | Minimal quality loss |
| int4 | 4 | 0.25× | 1–3% quality loss typical |
| int2 | 2 | 0.125× | Significant quality loss |
| fp4/E4M3 | 4 | 0.25× | Better than int4 for LLMs |

### Post-Training Quantization (PTQ)
Quantize a pretrained model after training. Fast and simple, but may introduce larger errors.

### Quantization-Aware Training (QAT)
Simulate quantization during training, allowing the model to adapt to lower precision. Better quality but requires retraining.

## GPTQ: Post-Training Quantization

**GPTQ** (2022) introduced accurate post-training quantization to 4 bits. Key innovations:

1. **Layer-wise optimization**: Quantize one layer at a time, adjusting remaining layers to compensate.
2. **Second-order information**: Uses the Hessian matrix to compute optimal adjustments.
3. **Optimal brain quantization**: Selects which weights to quantize and which to keep full precision.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,              # 4-bit quantization
    group_size=128,      # Group weights for finer granularity
    damp_percent=0.01,   # Damping factor for Hessian
    desc_act=False,      # Disable for better memory layout
)

model = AutoGPTQForCausalLM.from_pretrained(
    model, quantize_config=quantize_config
)
model.quantize(dataset)  # Run calibration on a small dataset
model.save_quantized(save_dir)
```

GPTQ achieves near-reference quality at 4-bit but requires calibration data (usually 128–512 samples) to minimize accuracy loss.

## AWQ: Activation-Aware Weight Quantization

**AWQ** (2023) improves on GPTQ by considering activation magnitudes during quantization:

- **Important weights**: Protect weights that have high activation magnitude (they're more important).
- **Smooth activations**: Scale activations before quantization to reduce outlier values.
- **No calibration needed**: Uses statistics from a forward pass, no gradient-based optimization.

AWQ often matches or exceeds GPTQ quality with simpler setup.

## GGUF: The Inference Format

**GGUF** (formerly GGML) is a file format designed for efficient inference on CPU and GPU. Key features:

- **Memory-mapped files**: The OS pages in only the parts of the model needed.
- **K-quantizations**: Various quantization schemes (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) with different quality/compression tradeoffs.
- **KV cache quantization**: Even the key-value cache can be quantized.

```bash
# Convert to GGUF using llama.cpp
./convert-hf-to-gguf ./model_directory --outtype q4_0
```

### GGUF K-Quantization Schemes

| Scheme | Memory vs. fp16 | Quality |
|--------|-----------------|---------|
| Q4_0 | 25% | Good for 7B+ models |
| Q4_1 | 28% | Slightly better than Q4_0 |
| Q5_0 | 33% | Better quality, more memory |
| Q5_1 | 37% | Near-fp16 quality |
| Q8_0 | 50% | Minimal quality loss |

## bitsandbytes: 8-bit and 4-bit Training

The `bitsandbytes` library provides quantization for training:

```python
from bitsandbytes import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_8bit=True,        # 8-bit inference
    load_in_4bit=True,        # 4-bit inference
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",  # NormalFloat4 format
)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=config,
    device_map="auto",
)
```

## FP4 and E4M3: The New Standards

FP4 and E4M3 are IEEE-standardized 4-bit floating-point formats:

- **E4M3**: 1 sign bit, 4 exponent bits, 3 mantissa bits.
- **FP4 (E2M1)**: 1 sign bit, 2 exponent bits, 1 mantissa bit.

These formats have native hardware support on newer GPUs (NVIDIA Hopper, AMD MI300) and often outperform integer quantization in quality.

```python
# Using fp4 in transformers
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=True,
)
```

## Double Quantization

**Double quantization** quantizes the quantization constants themselves. The constants for each weight group normally take significant space; double quantization reduces this overhead dramatically (from ~0.5 bits/parameter to ~0.1 bits/parameter).

This is used in GPTQ, AWQ, and NF4 to achieve higher compression without quality loss.

## Quantization Best Practices

1. **Start with 4-bit**: For models 7B and above, 4-bit quantization typically has negligible quality impact.
2. **Use GPTQ or AWQ for inference**: PTQ is faster than training-aware quantization.
3. **QLoRA for fine-tuning**: 4-bit quantized base model with LoRA adapters for efficient fine-tuning.
4. **Check quality on your data**: Benchmark on your actual use case — quality varies by task.
5. **Mind the KV cache**: For long context, the KV cache can dominate memory even with quantized weights.

Quantization is now essential knowledge for anyone deploying LLMs. The technique has democratized access to large models, letting you run 70B+ models on consumer hardware that would previously require enterprise GPUs.