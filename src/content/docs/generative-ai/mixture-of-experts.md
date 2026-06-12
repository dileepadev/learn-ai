---
title: Mixture of Experts (MoE)
description: Understanding the Mixture of Experts architecture — how sparse activation enables massive models with efficient inference.
---

Mixture of Experts (MoE) is a neural network architecture that replaces dense layers with a collection of specialized sub-networks called **experts**, where only a small subset of experts is activated for each input. This allows models to scale to enormous parameter counts without a proportional increase in compute.

## The Basic Idea

In a standard dense transformer, every token passes through every parameter in every layer. In an MoE layer, each token is routed to only a few of the available expert networks — typically 2 out of potentially hundreds. The parameters not selected for a given token are simply not used for that computation.

This creates a separation between:
- **Total parameters** (the full model size — very large)
- **Active parameters** (used per token — much smaller)

For example, Mixtral 8x7B has ~47B total parameters but only ~13B active parameters per token, making it roughly as fast as a 13B dense model at inference while having the representational capacity of a 47B model.

## How Routing Works

A **router** (also called a gating network) is a small learned function that, for each token, produces a probability distribution over all experts and selects the top-k (usually 2) with the highest scores.

```
gate_scores = softmax(W_g · x)
selected_experts = top_k(gate_scores, k=2)
output = Σ gate_score_i · expert_i(x)  for i in selected_experts
```

The router is trained end-to-end alongside the experts.

## Load Balancing

A key challenge is that without constraints, the router tends to collapse — always picking the same few experts while others never get trained (expert collapse). To prevent this, an **auxiliary load balancing loss** encourages the router to distribute tokens roughly evenly across experts.

## Notable MoE Models

- **Mixtral 8x7B and 8x22B** (Mistral AI) — open-weight MoE models competitive with much larger dense models.
- **GPT-4** — widely believed to use an MoE architecture.
- **Gemini 1.5** — MoE-based, enabling the large context window.
- **DeepSeek-V2 and V3** — efficient MoE designs with fine-grained expert routing.
- **Switch Transformer** — Google's early demonstration that MoE scales better than dense models.

## Advantages

- **Efficient scaling:** Add more experts (and parameters) without proportionally increasing FLOPs.
- **Specialization:** Different experts can learn different skills, knowledge domains, or token types.
- **Better performance per FLOP** compared to equivalently-sized dense models.

## Challenges

- **Memory:** All expert weights must be loaded into memory even if only a few are used per token. Requires significant GPU memory or efficient sharding.
- **Communication overhead:** In distributed settings, routing tokens to experts on different devices adds communication latency.
- **Training instability:** Load balancing and router training require careful tuning.
- **Expert collapse:** Needs auxiliary losses to ensure all experts are utilized.

## MoE in Practice

MoE is primarily used in very large language models where the goal is maximum capability at controlled inference cost. For most practitioners, MoE is a reason to prefer models like Mixtral over denser alternatives of similar speed — the architecture is largely transparent from the user's perspective.
