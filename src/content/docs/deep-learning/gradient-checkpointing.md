---
title: "Gradient Checkpointing: Training Large Models on Limited Memory"
description: "Understand how gradient checkpointing trades compute for memory, enabling training of models that would otherwise exceed GPU memory limits."
---

Modern deep learning models have grown enormous. A 70B parameter model in full precision requires 280GB just to store weights — far exceeding any single GPU's memory. **Gradient checkpointing** is a memory optimization technique that makes training such models possible by strategically recomputing activations during the backward pass.

## The Memory Problem

Training a neural network requires storing:

1. **Model weights**: The parameters being optimized.
2. **Gradients**: Derivatives of the loss with respect to each weight.
3. **Activations**: The outputs of each layer, needed to compute gradients during backpropagation.
4. **Optimizer states**: Momentum and variance terms for adaptive optimizers like Adam.

For a model with L layers, activations from all L layers are stored simultaneously in naive implementation. For a model processing a batch of 8 on 2048-token sequences, these activations can easily exceed 100GB.

## How Checkpointing Works

Gradient checkpointing is based on a simple insight: not all activations need to be stored simultaneously. During the forward pass, select a subset of layers as **checkpoints**. Only the checkpoint activations are saved; intermediate activations are discarded.

During the backward pass, recompute the discarded activations on the fly by re-running the forward computation from the nearest checkpoint.

```
Forward Pass:
  Layer 1 → Layer 2 ✓ checkpoint → Layer 3 → Layer 4 ✓ checkpoint → Layer 5 → ... → Layer N

Backward Pass:
  ... → recompute 4 → checkpoint 4 → recompute 3 → checkpoint 2 → recompute 1
```

## Memory vs. Compute Tradeoff

Checkpointing reduces memory but increases compute:

| Strategy | Memory | Compute Overhead |
|----------|--------|------------------|
| No checkpointing | O(N × activation_size) | 1× |
| Checkpoint every layer | O(activation_size) | ~1.5× |
| Checkpoint every K layers | O(N/K × activation_size) | ~1 + 1/K × |

The sweet spot is usually checkpointing every 2–4 layers, achieving 3–5× memory reduction with 20–50% compute overhead.

## Implementation in PyTorch

PyTorch provides gradient checkpointing via `torch.utils.checkpoint`:

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
    
    def forward(self, x):
        # Checkpoint the first half of layers
        for i, layer in enumerate(self.layers[:len(self.layers)//2]):
            x = layer(x)
            if i < len(self.layers)//2 - 1:
                x = checkpoint(lambda _: layer(_), x)
        
        # Second half without checkpointing (smaller activations due to pooling, etc.)
        for layer in self.layers[len(self.layers)//2:]:
            x = layer(x)
        return x
```

## Memory Analysis

Here's how memory changes for a 70B model on 8-GPU training:

| Configuration | Memory per GPU | Feasibility |
|---------------|----------------|-------------|
| No checkpointing, fp32 | 280GB weights alone | Impossible |
| No checkpointing, fp16 | 140GB weights | Requires 8x80GB GPUs, still hits OOM |
| Checkpointing, fp16, 8 GPUs | ~40GB active, 20GB params | Works on 8xA100 |
| Checkpointing, fp16, 4 GPUs | ~60GB active, 40GB params | Works on 4xA100 with ZeRO |

## Activation Checkpointing Strategies

### Uniform Checkpointing
Checkpoint every N layers uniformly. Simple and works well when layers are similar in size.

### Selective Checkpointing
Checkpoint layers with large activation sizes (early layers in transformers often have larger activations due to longer sequences). The most memory-efficient approach but requires profiling.

### Gradient Checkpointing + ZeRO
Gradient checkpointing combines powerfully with ZeRO optimizer sharding. ZeRO reduces parameter, gradient, and optimizer state memory across GPUs; checkpointing reduces activation memory. Together, they enable training models 10× larger than either technique alone.

## When to Use Checkpointing

Use gradient checkpointing when:
- You run out of memory during training.
- You're limited to 1–4 GPUs and need to fit a larger model.
- The 20–50% compute overhead is acceptable for your budget.

Don't use checkpointing when:
- You have abundant GPU memory (no need to pay the compute cost).
- Training time is critical and compute is scarce.
- The model is small enough to fit in memory without it.

Gradient checkpointing is essential for anyone training or fine-tuning large models on limited hardware. The technique has enabled the entire ecosystem of open-source 70B+ models to be fine-tuned on consumer hardware.