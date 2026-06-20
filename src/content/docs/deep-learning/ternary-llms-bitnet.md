---
title: Ternary LLMs & 1.58-bit Quantization (BitNet)
description: Discover Ternary LLMs and the BitNet 1.58B architecture, which restricts weights to {-1, 0, 1}, replacing floating-point multiplications with addition and subtraction operations.
---

Quantization is crucial for running Large Language Models on commodity hardware. Typically, models are trained in 16-bit floating-point (FP16 or BF16) and post-quantized to 8-bit or 4-bit integers. 

**Ternary LLMs** (popularized by architectures like **BitNet b1.58**) take quantization to the extreme. Instead of continuous real values, every weight in the network is restricted to just three states: $\{-1, 0, 1\}$. This reduces the weight bit-width to approximately $\log_2(3) \approx 1.58$ bits, replacing expensive floating-point matrix multiplications with simple additions and subtractions.

---

## Moving Beyond Matrix Multiplications

In standard deep learning hardware, the primary bottleneck is the Multiply-Accumulate (MAC) operation. When computing a layer output $y = W x$, a GPU performs millions of floating-point multiplications:

$$\text{MAC}(w, x) = w \times x + \text{accum}$$

In a 1.58-bit Ternary model:
- Weights are restricted to $-1$, $0$, or $1$.
- If $W_{i,j} = 1$, the operation becomes a simple addition: $y_i += x_j$.
- If $W_{i,j} = -1$, the operation becomes a subtraction: $y_i -= x_j$.
- If $W_{i,j} = 0$, the operation is skipped.

This eliminates almost all floating-point multiplications from the core linear layers, drastically reducing the silicon area, power consumption, and latency of AI hardware.

---

## The BitNet b1.58 Quantization Formula

To train a ternary model without losing representation capacity, BitNet b1.58 uses a weight quantization function during the forward pass.

Given a weight matrix $W$:
1. We compute the average absolute value of the matrix:
   
   $$\gamma = \frac{1}{d \times k} \sum_{i,j} |W_{i,j}|$$

2. We scale and round the weights to the nearest integer in $\{-1, 0, 1\}$:

   $$\tilde{W} = \text{Round}\left( \frac{W}{\gamma + \epsilon} \right)$$

3. We clip the values to ensure they fall strictly within $[-1, 1]$:

   $$\tilde{W}_{\text{quant}} = \text{Clip}\left( \tilde{W}, -1, 1 \right)$$

Where $\epsilon$ is a small constant to prevent division by zero.

### Activation Quantization
Unlike weights, activations cannot be easily ternary-quantized without severe information loss. Instead, BitNet quantizes activations to 8-bit integers (INT8) per tensor:

$$\tilde{x} = \text{Clip}\left( \text{Round}\left( x \times \frac{127}{\max(|x|)} \right), -128, 127 \right)$$

Thus, the multiplication of quantized weights and activations uses low-power INT8 additions rather than FP16 operations.

---

## Training Ternary Models: Straight-Through Estimator (STE)

Because the rounding and clipping functions are non-differentiable (their gradients are zero almost everywhere), ternary models cannot be trained using standard backpropagation directly on the quantized weights.

To solve this, BitNet uses a **Straight-Through Estimator (STE)**:
1. **Latent Weights:** The optimizer maintains a high-precision (FP32) copy of the weights.
2. **Forward Pass:** The weights are quantized to $\{-1, 0, 1\}$ using the formula above, and the forward pass is executed.
3. **Backward Pass:** The gradients are calculated with respect to the quantized weights, but they are applied directly to update the high-precision latent weights, bypassing the quantization step:
   
   $$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \tilde{W}_{\text{quant}}}$$

---

## Why 1.58-bit is a Paradigm Shift

1. **Energy Efficiency:** Floating-point additions and INT8 additions consume up to 20x less energy than FP16/FP32 multiplications.
2. **Memory Bandwidth:** Model weights are loaded from memory at 1.58 bits per weight, reducing the memory bandwidth bottleneck by up to 10x.
3. **Lossless Performance:** Research demonstrates that once a model scales beyond 3 billion parameters, a BitNet b1.58 model matches the perplexity and downstream task accuracy of a full FP16 model of the same size.
