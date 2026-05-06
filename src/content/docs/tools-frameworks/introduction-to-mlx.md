---
title: Introduction to Apple MLX
description: A practical guide to Apple MLX, the array framework for machine learning on Apple Silicon, covering its unified memory model, lazy computation, functional transformations, and fine-tuning large language models with mlx-lm.
---

# Introduction to Apple MLX

**MLX** is Apple's open-source array framework designed specifically for machine learning on Apple Silicon (M1/M2/M3/M4 chips). Unlike PyTorch or JAX which manage separate CPU and GPU memory pools, MLX exploits the **unified memory architecture** of Apple Silicon — arrays live in shared memory accessible by both the CPU and GPU-class Neural Engine without any copies. This makes it uniquely efficient for on-device inference and fine-tuning.

## Why MLX on Apple Silicon

Apple Silicon's key advantage for ML:

- **Unified memory**: no PCIe bandwidth bottleneck between CPU and GPU memory
- **Neural Engine**: dedicated ML accelerator reaching 38 TOPS on M4 Pro
- **Memory bandwidth**: up to 120 GB/s on M3 Max — comparable to A100 for memory-bound inference
- **Power efficiency**: competitive ML performance at 20–40W vs 300–700W for datacenter GPUs

MLX's design philosophy mirrors JAX: immutable arrays, functional transformations, and lazy evaluation — but targets Metal GPU rather than CUDA.

## Installation

```bash
pip install mlx mlx-lm
```

## Core Array Operations

```python
import mlx.core as mx

# Create arrays (default dtype: float32)
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([[1, 2], [3, 4]], dtype=mx.float16)

# Arithmetic is lazy — no computation until evaluated
c = a * 2 + 1
mx.eval(c)   # trigger computation
print(c)     # [3.0, 5.0, 7.0]

# Operations default to the default device (GPU on Apple Silicon)
x = mx.random.normal(shape=(1024, 1024))
y = mx.linalg.matmul(x, x.T)
mx.eval(y)
```

## Lazy Computation Model

MLX uses lazy evaluation — operations build a computation graph that executes only when `mx.eval()` is called or when a value is read:

```python
# This builds a graph, no actual computation yet
a = mx.array([1, 2, 3], dtype=mx.float32)
b = a ** 2
c = b.sum()         # still lazy

# Trigger execution
mx.eval(c)
print(c.item())     # 14.0

# mx.eval can take multiple arrays — evaluates them together for efficiency
x = mx.random.uniform(shape=(512, 512))
y = mx.random.uniform(shape=(512, 512))
z1 = x @ y
z2 = (x + y).sum()
mx.eval(z1, z2)     # single GPU dispatch for both
```

## Functional Transformations

Like JAX, MLX provides composable functional transformations:

### Gradient Computation

```python
import mlx.core as mx


def loss_fn(w: mx.array, x: mx.array, y: mx.array) -> mx.array:
    pred = x @ w
    return ((pred - y) ** 2).mean()


# Compute gradient with respect to first argument (w)
grad_fn = mx.grad(loss_fn)

w = mx.random.normal(shape=(10, 1))
x = mx.random.normal(shape=(32, 10))
y = mx.random.normal(shape=(32, 1))

grads = grad_fn(w, x, y)   # (10, 1) — gradient of loss w.r.t. w
print(grads.shape)
```

### Value and Gradient Together

```python
loss_and_grad = mx.value_and_grad(loss_fn)
loss_val, grads = loss_and_grad(w, x, y)
```

### JIT Compilation

```python
@mx.compile
def matrix_ops(a: mx.array, b: mx.array) -> mx.array:
    return mx.tanh(a @ b + a.sum())

# First call compiles; subsequent calls use the compiled Metal kernel
result = matrix_ops(x, w)
```

## Building a Neural Network with `mlx.nn`

```python
import mlx.nn as nn
import mlx.optimizers as optim


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)


model = MLP(784, 256, 10)
optimizer = optim.AdamW(learning_rate=1e-3)


def loss_and_grad(model, x, y):
    def _loss(model):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y).mean()
    loss, grads = nn.value_and_grad(model, _loss)(model)
    return loss, grads


# Training step
loss, grads = loss_and_grad(model, x_batch, y_batch)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
```

## LLM Inference and Fine-Tuning with `mlx-lm`

`mlx-lm` is the MLX library for running and fine-tuning language models:

```bash
# Run inference
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --prompt "Explain quantum entanglement in one paragraph"

# LoRA fine-tuning
mlx_lm.lora --model mlx-community/Mistral-7B-v0.3-4bit \
    --train \
    --data data/ \
    --batch-size 4 \
    --num-layers 16 \
    --iters 1000
```

Programmatic fine-tuning:

```python
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import train

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Generate
response = generate(
    model, tokenizer,
    prompt="What is the capital of France?",
    max_tokens=100,
    verbose=True,
)
```

## Performance Comparison

On M3 Max (128GB) for Llama 3.1 8B:

| Framework | Tokens/sec (generation) | Memory (GB) |
|---|---|---|
| MLX (4-bit) | 65–80 | 6.5 |
| llama.cpp (Q4_K_M) | 55–70 | 5.5 |
| Ollama (MLX backend) | 60–75 | 6.5 |
| PyTorch MPS (fp16) | 20–30 | 16 |

MLX's unified memory means the full 128GB is available for models — enabling Llama 3.1 70B at 4-bit (35GB) without any quantization tricks.

## Key Differences from PyTorch/JAX

| Feature | PyTorch | JAX | MLX |
|---|---|---|---|
| Memory model | Separate CPU/GPU | Separate CPU/GPU | Unified (Apple Silicon) |
| Computation | Eager (default) | Lazy + JIT | Lazy + JIT |
| Grad transform | autograd tape | `jax.grad` | `mx.grad` |
| Target hardware | CUDA/CPU | CUDA/TPU/CPU | Apple Silicon |
| Ecosystem | Very large | Large | Growing |

## Summary

MLX offers Apple Silicon users a first-class ML framework that leverages unified memory to eliminate the CPU↔GPU data transfer bottleneck. Its JAX-inspired design — lazy evaluation, composable transforms, functional gradient computation — makes it clean and expressive for research. For practitioners wanting to run and fine-tune frontier language models on MacBooks and Mac Studios without a cloud GPU subscription, `mlx-lm` provides a compelling local alternative with performance that rivals dedicated ML hardware for inference workloads.
