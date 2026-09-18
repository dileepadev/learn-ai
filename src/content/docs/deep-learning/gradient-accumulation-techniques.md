---
title: Gradient Accumulation Techniques for Training with Limited Memory
description: Learn how gradient accumulation simulates large batch sizes on memory-constrained hardware by summing gradients across multiple smaller forward-backward passes.
---

Gradient accumulation lets you train with an effectively large batch size even when GPU memory can only fit a much smaller batch in a single forward-backward pass, by accumulating gradients across several small "micro-batches" before performing a single optimizer update.

## Why Batch Size Is Memory-Constrained

Larger batch sizes generally produce more stable gradient estimates and can improve training throughput, but every additional example in a batch consumes additional GPU memory for activations stored during the forward pass and needed again during backpropagation. Once memory is exhausted, increasing batch size further isn't possible without either more GPU memory or a technique that decouples the effective batch size used for the gradient update from the batch size processed in any single forward-backward pass.

## How Gradient Accumulation Works

Instead of computing gradients from one large batch and updating weights immediately, gradient accumulation runs several forward-backward passes on smaller micro-batches, summing (or averaging) the resulting gradients, and only applies the optimizer update after a specified number of accumulation steps:

```python
optimizer.zero_grad()
for i, micro_batch in enumerate(micro_batches):
    loss = model(micro_batch) / accumulation_steps
    loss.backward()          # gradients accumulate in .grad by default
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Dividing the loss by `accumulation_steps` before calling `backward()` ensures the accumulated gradient matches what a single large-batch forward-backward pass would have produced, since PyTorch's default behavior is to sum (not average) gradients across repeated `backward()` calls without an intervening `zero_grad()`.

## What Gradient Accumulation Doesn't Fix

Gradient accumulation reduces peak memory for the batch dimension, but it doesn't reduce memory needed for a single micro-batch's own activations — if even one example's forward pass exceeds available memory (common with very long sequences or very large models), gradient accumulation alone won't help, and techniques like gradient checkpointing (recomputing activations during the backward pass instead of storing them) or model parallelism become necessary alongside it. Batch normalization layers are also affected differently by gradient accumulation than data-parallel training, since batch statistics are computed per micro-batch rather than over the full effective batch, which can subtly change training dynamics for models sensitive to this.

## Practical Guidance

Use gradient accumulation whenever you want a larger effective batch size than fits in memory but don't need the throughput improvement of distributed training across multiple devices — it's a purely single-device solution to a memory constraint, trading wall-clock training time for reduced memory footprint since accumulation steps run sequentially rather than in parallel. Combine it with mixed-precision training and gradient checkpointing when a single micro-batch still doesn't fit in memory on its own, addressing the per-example memory problem that accumulation alone cannot solve.
