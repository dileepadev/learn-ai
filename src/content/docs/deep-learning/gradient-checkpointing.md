---
title: Gradient Checkpointing
description: Learn how gradient checkpointing (activation recomputation) reduces GPU memory usage during deep network training by trading memory for compute — covering the memory-compute tradeoff, sublinear memory algorithms, selective checkpointing strategies, and PyTorch implementation patterns.
---

Training deep neural networks requires storing all intermediate activations produced during the forward pass so that gradients can be computed during backpropagation. For a network with $L$ layers and batch size $B$, the naive approach stores $O(L \cdot B)$ activation tensors simultaneously in GPU memory. As models grow deeper and batch sizes larger, activation memory becomes the dominant constraint — often exceeding the memory required for the model weights themselves.

**Gradient checkpointing** (also called **activation recomputation**) is a technique that reduces activation memory from $O(L)$ to $O(\sqrt{L})$ — or even less — by storing only a subset of activations during the forward pass and recomputing the rest during backpropagation. The cost is additional forward-pass compute; the gain is the ability to train much larger models or with much larger batch sizes on the same hardware.

## The Memory-Compute Tradeoff in Backpropagation

Standard backpropagation proceeds in two phases:

### Forward Pass

Each layer $l$ computes its output $a_l = f_l(a_{l-1}, \theta_l)$ from the previous activation $a_{l-1}$ and layer parameters $\theta_l$. All activations $\{a_0, a_1, \ldots, a_L\}$ are retained in memory because they are needed to compute gradients.

### Backward Pass

The gradient of the loss with respect to layer $l$'s parameters requires the activation $a_{l-1}$ (the input to layer $l$):

$$\frac{\partial \mathcal{L}}{\partial \theta_l} = \frac{\partial \mathcal{L}}{\partial a_l} \cdot \frac{\partial f_l(a_{l-1}, \theta_l)}{\partial \theta_l}$$

Without activation storage, the backward pass would need to recompute every activation from scratch — which is correct but wastes compute. With full activation storage, memory scales as $O(L \cdot B \cdot d)$ where $d$ is the feature dimension, quickly exhausting GPU memory for deep models.

## Gradient Checkpointing Algorithm

Gradient checkpointing divides the network into **segments** and saves only the activations at segment boundaries (the **checkpoints**). Within each segment, activations are discarded after the forward pass. During the backward pass, when a segment's internal activations are needed, they are **recomputed** from the nearest checkpoint.

For a network of $L$ layers divided into $k$ segments of $L/k$ layers each:

- **Memory**: stores only $k$ checkpoint activations simultaneously → $O(k \cdot B \cdot d)$.
- **Recompute cost**: each segment is run forward once during the initial forward pass and once more during backpropagation → $\approx 1 + \frac{L/k}{L} = 1 + 1/k$ forward passes.

### Optimal Checkpoint Spacing

The total cost is:

$$\text{Memory} = O(k \cdot d), \quad \text{Extra Compute} = O(L/k)$$

Setting $k = \sqrt{L}$ minimizes the product (memory × compute overhead), giving:

$$\text{Memory} = O(\sqrt{L} \cdot d), \quad \text{Extra Compute} \approx 33\%$$

With $\sqrt{L}$ checkpoints, a 1,000-layer network's activation memory drops from $O(1000)$ to $O(32)$ — a 31× reduction — at the cost of roughly one-third more forward-pass compute. This is the standard "sublinear memory" result for gradient checkpointing.

## PyTorch Implementation

PyTorch provides `torch.utils.checkpoint.checkpoint` for segment-level gradient checkpointing:

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # Recompute self.layer(x) during backward instead of storing activations
        return checkpoint(self.layer, x, use_reentrant=False)

# Apply to every Nth transformer block
class TransformerWithCheckpointing(torch.nn.Module):
    def __init__(self, blocks, checkpoint_every=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.checkpoint_every = checkpoint_every

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i % self.checkpoint_every == 0:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
```

The `use_reentrant=False` flag is the recommended setting for modern PyTorch (2.0+) — it uses a more memory-efficient and composable implementation that handles complex autograd graphs correctly.

### Hugging Face Transformers Integration

In Hugging Face Transformers, gradient checkpointing is enabled with a single call:

```python
model.gradient_checkpointing_enable()
```

This applies checkpointing to every transformer block by default. Fine-grained control over which blocks to checkpoint is available through custom `gradient_checkpointing_kwargs`.

## Selective Checkpointing

Not all layers benefit equally from checkpointing. Activation memory varies by operation type:

| Operation | Activation Memory | Recompute Cost |
| --- | --- | --- |
| Attention ($O(n^2 d)$ for sequence length $n$) | High | Medium |
| FFN / MLP | Medium | Low |
| LayerNorm | Low | Very low |
| Embedding lookup | Low | Very low |

**Selective checkpointing** applies recomputation only to the highest-memory operations while retaining cheap-to-store activations. For transformer models, attention blocks — which store $O(n^2)$ activations for the attention weights — are the primary targets. Selectively checkpointing only attention blocks reduces memory nearly as much as full checkpointing with lower compute overhead.

### FlashAttention Integration

**FlashAttention** (Dao et al., 2022) is often described as an orthogonal optimization to gradient checkpointing, but they interact synergistically:

- FlashAttention avoids materializing the $O(n^2)$ attention weight matrix by computing attention in tiled chunks — eliminating the largest single activation tensor.
- With FlashAttention, the attention activation cost drops dramatically; the remaining memory bottleneck shifts to FFN activations, where gradient checkpointing is applied.

The combination of FlashAttention + gradient checkpointing enables training with sequence lengths of 64K+ tokens on standard A100 GPUs.

## Rematerialization in XLA and JAX

In JAX and XLA (used by TPU training), gradient checkpointing is expressed through `jax.checkpoint` (formerly `jax.remat`):

```python
import jax
from functools import partial

@partial(jax.checkpoint, prevent_cse=False)
def transformer_block(params, x):
    return attention(params, x) + ffn(params, x)
```

JAX's compiler can automatically determine which activations to rematerialize based on the compute graph, without requiring manual specification of checkpoint boundaries. The `prevent_cse=False` flag allows common subexpression elimination across rematerialized computations, improving efficiency.

## Memory Savings in Practice

For LLaMA-3 8B (32 transformer layers) with sequence length 4096 and batch size 1 on A100 80GB:

| Configuration | Activation Memory | Total Training Memory |
| --- | --- | --- |
| No checkpointing | ~18 GB | ~34 GB (exceeds A100) |
| Full checkpointing | ~2.3 GB | ~18 GB |
| Selective (attention only) | ~4 GB | ~20 GB |
| No checkpointing + gradient accumulation (micro-batch=1) | ~4 GB | ~20 GB |

Full checkpointing reduces activation memory by ~87% at the cost of ~33% more compute, making training feasible on a single 80GB GPU.

## Interaction with Mixed Precision and Gradient Accumulation

Gradient checkpointing interacts with other memory optimizations:

- **Mixed precision (BF16/FP16)**: activations are stored in the training dtype (BF16). Recomputed activations are also in BF16, matching the original forward pass.
- **Gradient accumulation**: reduces effective batch size per step. Combining gradient accumulation (to simulate large batches) with gradient checkpointing (to reduce activation memory) allows training with very large effective batch sizes on memory-constrained hardware.
- **ZeRO / FSDP**: model parallel training splits parameters and gradients across GPUs. Gradient checkpointing is complementary — ZeRO/FSDP reduce parameter and optimizer state memory, while checkpointing reduces activation memory.

## Limitations

- **Throughput reduction**: the 33% compute overhead translates directly to slower training wall-clock time. For memory-abundant setups (e.g., A100 with small models), gradient checkpointing may not be worth the throughput cost.
- **Non-determinism with dropout**: when activations within a checkpointed segment are recomputed, any stochastic operations (like dropout) produce different values than the original forward pass unless the random state is explicitly saved and restored. PyTorch's `checkpoint` handles this by saving and restoring the RNG state automatically.
- **Custom autograd functions**: checkpointing does not work out of the box with `torch.autograd.Function` subclasses — these must explicitly support checkpointing via the `saved_tensors` mechanism.

## Summary

Gradient checkpointing reduces activation memory from $O(L)$ to $O(\sqrt{L})$ by recomputing discarded intermediate activations during backpropagation, enabling training of models that would otherwise exhaust GPU memory. With optimal checkpoint spacing at every $\sqrt{L}$ layers, the compute overhead is approximately 33% — a favorable tradeoff for most large-model training scenarios. PyTorch's `torch.utils.checkpoint`, Hugging Face's `gradient_checkpointing_enable()`, and JAX's `jax.checkpoint` provide production-ready implementations. Combined with FlashAttention (which eliminates $O(n^2)$ attention memory), gradient checkpointing enables training of billion-parameter models with long sequences on commodity GPU hardware.
