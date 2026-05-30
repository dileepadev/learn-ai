---
title: "Foundation Model Fine-Tuning Strategies"
description: "Compare full fine-tuning, LoRA, QLoRA, prefix tuning, and other parameter-efficient fine-tuning methods — with practical guidance on when to use each approach."
---

Fine-tuning adapts a pretrained foundation model to a specific task or domain. With models ranging from 7B to 70B+ parameters, choosing the right fine-tuning strategy is as much an engineering decision as a research one.

## Why Fine-Tune?

Prompting alone has limits. Fine-tuning is worth the investment when:
- You need consistent output format or style that prompting can't reliably achieve.
- You have domain-specific knowledge not well-represented in the base model.
- You need to reduce latency or cost by using a smaller, specialized model.
- You want to remove capabilities (e.g., refusals) or add new ones.

## Full Fine-Tuning

Update all model parameters on your task-specific dataset. Produces the best results but requires:
- GPU memory proportional to model size × 4 (weights + gradients + optimizer states).
- A 7B model requires ~60GB GPU memory with AdamW.
- Risk of catastrophic forgetting of general capabilities.

Use full fine-tuning when you have the compute budget and need maximum performance on a narrow task.

## LoRA (Low-Rank Adaptation)

Instead of updating all parameters, LoRA adds small trainable rank-decomposition matrices to specific layers (typically attention projections). The original weights are frozen.

```
W' = W + BA   where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

With rank r=16, a 7B model's trainable parameters drop from 7B to ~20M — a 350x reduction. Memory requirements drop proportionally.

**Key hyperparameters**:
- `r` (rank): Higher rank = more capacity = more memory. Start with 16–64.
- `alpha`: Scaling factor, typically set to r or 2r.
- `target_modules`: Which layers to apply LoRA to. Attention layers are standard; adding MLP layers helps for knowledge-intensive tasks.

## QLoRA

Combines LoRA with 4-bit quantization of the base model. The frozen base model weights are stored in 4-bit NF4 format, reducing memory by ~4x. LoRA adapters are trained in 16-bit.

A 70B model that would require 140GB in fp16 can be fine-tuned on a single 48GB GPU with QLoRA. The quality tradeoff is minimal for most tasks.

## DoRA (Weight-Decomposed Low-Rank Adaptation)

Decomposes weight updates into magnitude and direction components, applying LoRA only to the direction. Often outperforms LoRA at the same rank, especially for instruction following.

## Prefix Tuning and Prompt Tuning

Add trainable "soft prompt" tokens to the input. The model weights are completely frozen — only the soft prompt embeddings are trained.

- **Prompt Tuning**: Prepend trainable tokens to the input.
- **Prefix Tuning**: Prepend trainable tokens to every layer's key and value matrices.

These methods use very few parameters but are less expressive than LoRA and work best with large models (>10B parameters).

## Instruction Fine-Tuning

Fine-tune on instruction-response pairs to improve the model's ability to follow instructions. Key considerations:

- **Data quality over quantity**: 1,000 high-quality examples often outperform 100,000 noisy ones.
- **Diversity**: Cover a wide range of instruction types and domains.
- **Format consistency**: Use a consistent chat template (ChatML, Alpaca, etc.).

## RLHF and DPO

For alignment fine-tuning:

- **RLHF**: Train a reward model on human preference data, then use PPO to optimize the policy. Complex but powerful.
- **DPO (Direct Preference Optimization)**: Directly optimize on preference pairs without a separate reward model. Simpler and often competitive with RLHF.
- **ORPO**: Combines SFT and preference optimization in a single training stage.

## Practical Decision Guide

| Scenario | Recommended Approach |
|---|---|
| Limited GPU (< 24GB) | QLoRA |
| Good GPU, 7B model | LoRA or full fine-tuning |
| 70B+ model | QLoRA |
| Frozen model required | Prefix/prompt tuning |
| Alignment/RLHF | DPO for simplicity, RLHF for maximum control |
| Production deployment | Full fine-tuning if possible (no adapter overhead) |
