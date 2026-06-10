---
title: Model Compression and Quantization
description: Reducing model size and computational cost through pruning, quantization, and knowledge distillation for efficient deployment.
---

**Model compression** techniques reduce the size and computational requirements of neural networks, enabling deployment on resource-constrained devices (mobile phones, edge devices, IoT sensors) and reducing inference latency and cost.

A compressed model can be 10-100x smaller, run 10-50x faster, while maintaining accuracy — crucial for real-world AI deployment.

## The Compression Problem

Modern deep learning models are massive:

- **GPT-3**: 175 billion parameters.
- **Vision Transformers**: 300+ million parameters.
- **Fine-tuned BERT**: 340 million parameters.

Deploying these on a smartphone or embedded device is infeasible. Compression bridges the gap between model capability and deployment constraints.

## Pruning

**Pruning** removes less important weights or neurons, assuming redundancy.

### Magnitude-Based Pruning

Remove weights with small absolute values:

$$\text{mask}_i = \begin{cases} 1 & \text{if } |w_i| > \theta \\ 0 & \text{if } |w_i| \leq \theta \end{cases}$$

where $\theta$ is a threshold. Weights below the threshold are set to zero and can be skipped during computation.

**Sparsity**: The fraction of weights removed. 90% sparsity means only 10% of weights remain — 10x smaller.

**Trade-off**: Sparsity-accuracy trade-off. At ~90% sparsity, most models lose 1-5% accuracy. Beyond 95%, degradation accelerates.

### Structured Pruning

Remove entire neurons, channels, or filters rather than individual weights:

$$\text{Remove entire channel } c \text{ if } \sum_i |w_{ic}| < \theta$$

**Advantage**: Structured sparsity is hardware-efficient; BLAS libraries and GPUs can exploit it directly.

**Trade-off**: Removes fewer parameters than unstructured pruning for the same accuracy loss, but requires less specialized hardware.

### Lottery Ticket Hypothesis

**Key insight**: Dense networks contain subnetworks (the "lottery tickets") that, if trained in isolation, achieve comparable accuracy to the full network.

Process:
1. Train a dense network to convergence.
2. Prune (remove) low-magnitude weights.
3. Reset remaining weights to their initial values.
4. Train the pruned network.

Surprisingly, the pruned network often matches the original's accuracy. This suggests dense models are overparameterized; efficient sparse subnetworks exist from initialization.

## Quantization

**Quantization** reduces the precision of weights and activations, typically from 32-bit floating-point (float32) to lower bit-widths (int8, int4, or even 1-bit).

### Post-Training Quantization (PTQ)

Quantize a trained model without retraining:

1. **Calibration**: Run the model on representative data to measure the range of activations.
2. **Map**: Map float32 values to lower precision (e.g., int8): map float range [a, b] to int range [0, 255].
3. **Dequantize**: During inference, convert back to float for computation.

**Advantage**: Fast; no retraining required.

**Trade-off**: Accuracy loss can be significant (2-5%) depending on bit-width.

### Quantization-Aware Training (QAT)

Train with quantization in mind:

1. **Simulate quantization** during training using straight-through estimators.
2. **Learn scale factors** (quantization parameters) that minimize accuracy loss.
3. **Fine-tune** with quantization in the loop.

After training, quantization causes minimal accuracy loss.

**Trade-off**: Requires retraining but achieves better accuracy-compression balance.

### Bit-Width Trade-offs

| Bit-Width | Model Size Reduction | Accuracy Loss | Hardware Efficiency |
|-----------|---------------------|--------------|-------------------|
| **FP32** | Baseline (1x) | None | Baseline |
| **INT8** | 4x | 0-2% | Native on most hardware |
| **INT4** | 8x | 2-5% | Requires special libraries |
| **Binary (1-bit)** | 32x | 10-20% | Highly efficient but accuracy-limited |

### Symmetric vs. Asymmetric Quantization

**Symmetric**: Map range [-a, a] → int range symmetrically.

**Asymmetric**: Map range [a, b] → int range [0, 255], capturing zero-offset distributions.

Asymmetric is more flexible but requires storing both scale and zero-point.

## Knowledge Distillation

**Knowledge distillation** transfers knowledge from a large teacher model to a small student model:

1. **Train teacher**: Large, accurate model.
2. **Distill**: Train student to mimic teacher's outputs (soft targets from softmax with temperature $T$):

$$\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(\text{student}, \text{hard labels}) + (1-\alpha) \mathcal{L}_{\text{KL}}(\text{student}, \text{teacher})$$

where KL divergence measures the difference in soft predictions.

**Why it works**: Teacher outputs encode relational knowledge (e.g., if samples are similar, their probabilities should be close). Student learns these relationships, not just memorization.

**Temperature**: Higher $T$ smooths probability distributions, creating softer targets that are easier for the student to learn from. Typical values: $T = 3$ to $T = 20$.

### Advantages:
- Student can be orders of magnitude smaller than teacher.
- Works well with other compression techniques (quantization, pruning).
- Improves student generalization beyond just matching teacher's accuracy.

## Combined Approaches

**Compression techniques synergize**:

- **Pruning + Quantization**: Prune to 50% sparsity, quantize to int8 → 8x compression.
- **Distillation + Quantization**: Distill to smaller student, then quantize → compact model.
- **Iterative compression**: Progressively prune, quantize, and fine-tune.

A typical aggressive compression pipeline:
1. **Distill** large model to smaller student.
2. **Prune** student to 80-90% sparsity.
3. **Quantize** to int4.
4. **Result**: 100x+ smaller, 50x+ faster.

## Hardware Acceleration

Compression only helps if hardware exploits sparsity/reduced precision:

- **TPUs**: Natively support int8 and lower precision.
- **GPUs**: NVIDIA Tensor Cores support int8 efficiently.
- **Mobile**: ARM processors (in iPhones, Androids) support int8 via NEON/QUANT.
- **Specialized**: FPGA, ASIC designs for specific bit-widths (e.g., binary neural networks).

## Practical Considerations

### Accuracy-Efficiency Trade-off

Aggressive compression significantly reduces accuracy. The choice depends on the application:

- **Latency-critical** (autonomous vehicles, real-time chat): Accept small accuracy loss for speed.
- **Accuracy-critical** (medical diagnosis): Sacrifice efficiency for accuracy.

### Calibration Data

PTQ requires representative calibration data. If calibration data differs from deployment data, quantization mismatch occurs.

### Mixed Precision

Not all layers need the same bit-width. Sensitive layers (critical for accuracy) use higher precision; less sensitive layers use lower precision.

Example: Quantize early layers (low-level features) to int4, keep later layers (high-level semantics) at int8.

## Deployment Implications

Compressed models enable:

- **On-device inference**: No network latency; privacy preserved (data stays on device).
- **Lower cloud costs**: Smaller models run faster on fewer resources.
- **Energy efficiency**: Battery-powered devices can run longer.

Example: A compressed BERT runs on a mobile phone in 100ms; uncompressed would take 5+ seconds.

## Limitations

**Accuracy floor**: Heavy compression (>95% sparsity or <4-bit quantization) causes significant accuracy loss. Different tasks have different compression limits.

**Task-specific optimization**: Compression strategies optimal for one task may not generalize to others.

**Recompilation**: Changing bit-widths or sparsity patterns requires recompiling; less flexible than software model serving.

## Emerging Techniques

- **Dynamic quantization**: Adjust precision adaptively based on input (easier inputs use lower precision).
- **Neural architecture search**: Automatically find efficient architectures.
- **Hardware-software codesign**: Design models and hardware jointly for optimal efficiency.

Model compression is essential for making AI practical and accessible, enabling powerful models on resource-constrained devices and reducing environmental impact of AI inference.
