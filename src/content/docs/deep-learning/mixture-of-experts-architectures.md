---
title: Mixture-of-Experts Architectures - Scaling Models Without Scaling Compute Equally
description: Learn how mixture-of-experts layers let models grow to huge parameter counts while activating only a fraction of them per input.
---

A Mixture-of-Experts (MoE) layer replaces a single large feed-forward network with many smaller "expert" networks, plus a router that decides which experts process each input token. Only a subset of experts run for any given token, decoupling total parameter count from per-token compute cost.

```text
token -> router -> select top-k experts -> combine expert outputs -> layer output
                    (out of N total experts, only k are used)
```

## Sparse Routing

The router is typically a small learned network that scores each expert for a given token and selects the top `k` (commonly 1 or 2) to activate:

$$y = \sum_{i \in \text{top-}k} g_i(x)\, E_i(x)$$

`E_i` is the i-th expert network and `g_i(x)` is its gating weight, usually a softmax score over the selected experts. Because only `k` out of `N` experts run per token, a model can have, say, 8 times more total parameters than a dense model of equal computational cost per token—capacity grows without proportional inference cost.

## The Load Balancing Problem

If left unconstrained, routers tend to collapse onto favoring a small subset of experts, leaving others undertrained and wasting capacity. Training adds an auxiliary **load balancing loss** that penalizes uneven expert utilization, encouraging the router to spread tokens roughly evenly across experts. Some architectures also cap the number of tokens each expert can process per batch, dropping or rerouting overflow tokens.

## Trade-offs

- **Memory**: all experts must be held in memory (or fetched across devices) even though only a few run per token, so total parameter count still drives hardware requirements.
- **Communication overhead**: in distributed training, routing tokens to experts on different devices adds all-to-all communication costs that can bottleneck training if not carefully scheduled.
- **Training stability**: sparse routing introduces discrete decisions that complicate gradient flow and can destabilize early training without careful initialization and load balancing.

## Where It's Used

MoE layers underlie some of the largest known language models, letting them reach trillions of total parameters while keeping the compute cost of each forward pass closer to that of a much smaller dense model. The approach trades additional engineering complexity and memory footprint for a favorable capacity-to-compute ratio at inference time.
