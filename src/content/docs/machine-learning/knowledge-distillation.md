---
title: Knowledge Distillation
description: Learn how knowledge distillation trains smaller, faster models by transferring knowledge from a large teacher model.
---

Knowledge distillation is a model compression technique where a smaller **student** model is trained to mimic the behavior of a larger, more capable **teacher** model. The goal is to produce a compact model that retains most of the teacher's performance at a fraction of the compute and memory cost.

## The Core Idea

A standard model is trained to predict hard labels (e.g., "cat" or "dog"). But the teacher's **soft probability outputs** — the full distribution over all classes — contain richer information. For example, a teacher might output 70% cat, 25% tiger, 5% dog. These soft labels reveal similarity relationships between classes that a one-hot label discards entirely.

The student is trained on a combination of:
- **Hard loss:** Cross-entropy against the true labels.
- **Distillation loss:** KL divergence between the student's and teacher's soft probability distributions.

```
L = α · L_hard + (1 - α) · L_distill
```

A **temperature** parameter T softens the distributions further, making small probabilities more informative:

```
p_i = exp(z_i / T) / Σ exp(z_j / T)
```

Higher temperature produces softer distributions with more signal for the student.

## Types of Knowledge Distillation

- **Response-based:** Student mimics the teacher's final output logits. The simplest and most common form.
- **Feature-based:** Student also mimics intermediate activations (hidden layers) of the teacher. More information transferred but more complex to set up.
- **Relation-based:** Student learns to reproduce relationships between data points as seen by the teacher.

## Applications

- **DistilBERT:** A 40% smaller, 60% faster version of BERT that retains ~97% of its performance, created via distillation.
- **TinyBERT, MobileBERT:** Further compressed transformer variants for edge deployment.
- **On-device models:** Distilling large cloud models into versions that run on phones and embedded devices.

## When to Use Distillation

- You need a model that is fast at inference time (low latency).
- You are deploying to resource-constrained environments (mobile, edge, browser).
- You cannot retrain the teacher but want to compress its knowledge.
- You want a specialized student for a narrow task from a general teacher.

## Distillation vs. Other Compression Methods

| Method | What It Does |
|---|---|
| **Distillation** | Trains a new smaller model using teacher supervision |
| **Pruning** | Removes weights or neurons from an existing model |
| **Quantization** | Reduces numerical precision of weights (e.g., FP32 → INT8) |
| **Low-rank factorization** | Decomposes weight matrices into smaller ones |

These techniques are complementary — you can distill a model and then quantize or prune the student for additional gains.
