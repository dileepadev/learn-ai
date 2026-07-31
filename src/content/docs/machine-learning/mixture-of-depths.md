---
title: Mixture of Depths for Efficient Transformers
description: How dynamic layer routing enables more efficient computation in transformer models.
---

Mixture of Depths (MoD) is an architecture that allows transformers to skip computation for certain tokens at certain layers, making models more efficient without sacrificing performance.

## The Problem with Fixed Computation

Standard transformers apply the same computation to every token at every layer. This is wasteful:
- Some tokens are easy to process and don't need deep computation
- Some tokens are critical and benefit from extra processing
- All tokens get the same number of layers regardless of need

## How Mixture of Depths Works

**Routing Mechanism**
Each token at each layer gets a routing score from a learned function. Tokens with high scores proceed through the layer normally. Tokens with low scores skip the layer's computation.

**Token Pool Management**
- Fixed compute budget per layer
- Top-K tokens by routing score receive full computation
- Remaining tokens bypass the layer via residual connection
- Total compute is constant regardless of content

**Training**
Models learn which tokens need computation where. The routing function is trained end-to-end with the rest of the model using reinforcement learning or differentiable top-K approximations.

## Benefits

**Efficiency**
- Same quality with fewer FLOPs
- Or better quality with same FLOPs
- Faster inference on hardware that supports conditional execution

**Interpretability**
Routing patterns reveal which tokens the model considers important at different processing stages. Useful for understanding model behavior.

**Adaptive Computation**
Models naturally allocate more compute to:
- Ambiguous or rare tokens
- Reasoning-critical positions
- Complex syntactic structures

## Architecture Variants

**Static MoD**
Pre-defined routing pattern based on position (e.g., every other token skips certain layers).

**Dynamic MoD**
Learned routing that adapts to input content.

**MoD with Experts**
Combine with Mixture of Experts where each layer has multiple expert sub-networks. Tokens route to different experts and some skip entirely.

## Implementation Considerations

**Hardware Constraints**
Skipping computation doesn't always translate to speedups on GPUs. Hardware must support:
- Dynamic control flow
- Irregular memory access patterns
- Efficient sparse computation

**Load Balancing**
Ensure tokens are distributed evenly across layers to avoid compute hotspots.

**Gradient Flow**
Skipped tokens still need gradient information for training. Residual connections maintain gradient paths.

## Research Directions

- Combining with other efficiency techniques like quantization
- Task-specific routing patterns
- Cross-layer routing optimization
- Optimal compute budget learning
