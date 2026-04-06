---
title: "Test-Time Compute: Scaling Intelligence at Inference"
description: "Understanding how models can 'think' longer during inference to solve more complex reasoning problems."
---

Traditionally, we scale AI by increasing model size or training data. **Test-Time Compute** scaling suggests that we can also improve performance by allowing the model to spend more "compute" during the inference phase.

## Methods of Scaling at Inference

1. **Chain-of-Thought (CoT)**: Forcing the model to generate intermediate reasoning steps before reaching a final answer.
2. **Best-of-N Sampling**: Generating multiple candidate answers and using a separate "reward model" or "verifier" to select the best one.
3. **Iterative Refinement**: Allowing the model to critique and revise its own output multiple times before presenting it to the user.

## Why It Matters

This shift allows smaller, more efficient models to punch above their weight class on difficult reasoning tasks, making high-level intelligence more accessible without needing trillion-parameter architectures.
