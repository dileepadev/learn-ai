---
title: "Hyper-Networks: Models That Generate Models"
description: "An introduction to Hyper-Networks, where one neural network produces the weights for another network."
---

A **Hyper-Network** is an architecture where a primary network outputs the weight parameters for a target network. This allows for dynamic weight generation based on input context or task requirements.

## Mechanism

Instead of learning fixed weights, the Hyper-Network learns a mapping from an embedding (e.g., a style vector) to the weight space of the target model.

## Applications

- **Multi-Task Learning**: Adapting a single model to different tasks by generating task-specific weights.
- **Neural Architecture Search**: Quickly evaluating different model structures.
- **Personalization**: Generating custom model weights for individual users without full fine-tuning.
