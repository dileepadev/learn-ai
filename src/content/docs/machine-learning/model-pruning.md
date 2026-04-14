---
title: Neural Network Pruning
description: Understand model pruning — a compression technique that removes redundant parameters from neural networks to reduce size and inference cost without significant accuracy loss.
---

**Model pruning** is a neural network compression technique that removes parameters — weights, neurons, attention heads, or entire layers — that contribute little to model accuracy. The result is a smaller, faster model that requires less memory and compute to run.

Pruning is especially relevant as models grow larger: a parameter-efficient pruned model can achieve similar accuracy to its dense counterpart at a fraction of the inference cost.

## Why Prune?

Modern deep learning models are over-parameterized by design — they contain far more parameters than necessary to represent the learned function. Studies have consistently shown that large fractions of weights can be zeroed or removed with minimal impact on accuracy.

Key motivations:

- **Inference speed** — Fewer operations per forward pass.
- **Memory footprint** — Smaller models fit on edge devices or in RAM-constrained environments.
- **Energy efficiency** — Critical for mobile, IoT, and sustainability goals.
- **Storage** — Pruned models are smaller to store and distribute.

## Types of Pruning

### Unstructured Pruning

Individual weights are set to zero regardless of their position in the weight matrix. This produces a **sparse** model.

- Very high compression ratios are possible.
- Requires **sparse hardware/software support** to realize speedups (standard dense matrix operations don't benefit unless sparsity is 90%+).
- Common approach: **magnitude pruning** — remove the weights with the smallest absolute values.

### Structured Pruning

Entire structural units are removed: neurons, filters, attention heads, or layers. The resulting model remains **dense** and benefits immediately from standard hardware without special sparse support.

| Unit Removed | Common in |
|---|---|
| Neurons / channels | CNNs, MLPs |
| Attention heads | Transformers |
| Layers | Deep networks |
| Entire blocks | Large language models |

Structured pruning is generally more hardware-friendly but less flexible — it is harder to achieve very high compression ratios without accuracy degradation.

## The Lottery Ticket Hypothesis

The **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019) proposes that within every large neural network there exists a small **winning subnetwork** ("lottery ticket") that, when trained in isolation from the same initialization, achieves comparable accuracy to the full network.

This has important implications:

- Over-parameterization is not required at *inference* time.
- The dense model serves as a useful vehicle for *finding* the sparse subnetwork.
- Identifying winning tickets early in training is an active research area.

## The Pruning Pipeline

A standard iterative pruning workflow:

1. **Train** a dense model to convergence.
2. **Score** parameters by importance (magnitude, gradient, Hessian-based, etc.).
3. **Prune** a fraction of the least-important parameters.
4. **Fine-tune** (recover accuracy lost from pruning).
5. **Repeat** steps 2–4 until the target sparsity or size is reached.

This iterative approach consistently outperforms one-shot pruning.

## Importance Criteria

Different criteria are used to decide which parameters to remove:

| Criterion | Description |
|---|---|
| **Magnitude** | Remove weights with smallest $\|w\|$ |
| **Gradient magnitude** | Remove weights whose gradients are near zero |
| **Taylor expansion** | Approximate the loss change from removing a weight: $\Delta L \approx g_i w_i$ |
| **Fisher information** | Use second-order curvature to estimate importance |
| **Activation statistics** | Remove neurons/channels with low mean activation |

Magnitude pruning is simple and often competitive despite its simplicity.

## Pruning for Transformers and LLMs

Pruning transformer-based LLMs involves specific approaches:

- **Attention head pruning** — Remove heads that are found to be redundant or low-importance across many inputs.
- **Layer dropping** — Remove entire transformer blocks (early layers are typically less important).
- **Width pruning** — Reduce the hidden dimension of FFN layers.

Notable methods:

- **SparseGPT** — One-shot unstructured pruning for LLMs using a second-order method, achieving 50–60% sparsity with minimal perplexity increase.
- **LLM-Pruner** — Structured pruning of LLMs that prunes coupled structures (e.g., in grouped attention), followed by LoRA-based recovery fine-tuning.
- **Wanda** — Prunes weights based on both magnitude and input activations, outperforming pure magnitude pruning on LLMs.

## Pruning vs. Other Compression Techniques

| Technique | What it removes | Requires retraining? | Hardware benefit |
|---|---|---|---|
| **Pruning** | Parameters | Usually yes (fine-tune) | Structured: yes; Unstructured: needs sparse HW |
| **Quantization** | Bit precision of weights | Sometimes | Yes (immediate) |
| **Knowledge distillation** | Full model → smaller model | Yes (train from scratch) | Yes |
| **Low-rank factorization** | Redundant rank in matrices | Sometimes | Yes |

These techniques are complementary — pruning + quantization is a common combination.

## Practical Considerations

- **Pruning ratio** — 30–50% sparsity is usually "safe". Beyond 70–80%, accuracy drops more significantly for most tasks.
- **Fine-tuning budget** — The more aggressive the pruning, the more fine-tuning is needed to recover quality.
- **Task sensitivity** — Some tasks (e.g., rare-category classification) degrade faster under aggressive pruning.
- **Hardware target** — Choose structured pruning if deploying on CPUs/mobile GPUs; unstructured pruning is best suited for NVIDIA sparse tensor cores or custom accelerators.

Pruning is a key pillar of the **efficient deep learning** toolkit, enabling practitioners to push powerful models into resource-constrained production environments without sacrificing prohibitive amounts of accuracy.
