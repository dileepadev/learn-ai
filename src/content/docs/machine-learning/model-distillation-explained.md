---
title: Introduction to Model Distillation
description: Learn how to "distill" the knowledge of large, complex models into smaller and faster ones.
---

Model distillation is a powerful machine learning technique for compressing a large "teacher" model into a smaller, more efficient "student" model while preserving as much performance as possible.

## How Model Distillation Works

Instead of training a small model from scratch on raw labels, model distillation trains it to mimic the *outputs* (the logits) of a larger, pre-trained model.

1. **Hard Targets:** The student model is trained on the actual ground-truth labels.
2. **Soft Targets:** The student also tries to match the probability distribution of the teacher's outputs.
3. **Loss Function:** A weighted combination of both losses guides the student's training.

## Why Use Distillation?

- **Efficiency:** Drastically reduces the model's footprint and inference cost.
- **Portability:** Enables running complex models on edge devices like smartphones.
- **Accuracy:** The student model often performs better than a small model trained only on labels.

## Common Techniques

- **Knowledge Distillation (KD):** The classic approach focused on distilling output logits.
- **Feature Distillation:** Matching the internal activations (hidden states) of the teacher.
- **Relation Distillation:** Distilling the relative relationships between inputs.
