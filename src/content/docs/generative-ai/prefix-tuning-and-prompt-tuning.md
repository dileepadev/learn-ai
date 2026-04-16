---
title: Prefix Tuning and Prompt Tuning
description: Learn how prefix tuning and prompt tuning enable efficient LLM adaptation by optimizing soft prompt tokens in the input or activation space — without updating the full model weights.
---

**Prefix tuning** and **prompt tuning** are parameter-efficient fine-tuning (PEFT) techniques that adapt large language models by learning a small set of **soft (continuous) prompt tokens** rather than updating all model parameters. They sit alongside LoRA and adapter methods as alternatives to full fine-tuning.

## Motivation: The Cost of Full Fine-Tuning

Full fine-tuning updates every parameter of a pretrained model for each task. For models with billions of parameters this is:

- **Computationally expensive** — requires the same memory and compute as pretraining at scale.
- **Storage-intensive** — each task needs a full copy of model weights.
- **Prone to catastrophic forgetting** — fine-tuning on a narrow task can degrade broad capabilities.

PEFT methods like prefix tuning and prompt tuning reduce the number of trainable parameters by 100–1000× while achieving comparable performance.

## Prompt Tuning

Introduced by Lester et al. (2021), **prompt tuning** prepends a small sequence of **learnable embedding vectors** (soft tokens) to the input token embeddings. Only these soft tokens are updated during training; all original model weights are frozen.

### How It Works

Given an input $x$ tokenized into embeddings $E(x)$, a soft prompt $P \in \mathbb{R}^{k \times d}$ of $k$ learned vectors is prepended:

$$\text{Input to model} = [P; E(x)]$$

The loss is computed as usual on the output tokens, and gradients flow only into $P$.

### Key Properties

- **Frozen backbone** — The entire pretrained model is unchanged.
- **Lightweight** — Typically 10–100 soft tokens; for a 10B parameter model this is a tiny fraction of parameters.
- **Task-specific** — Each task gets its own soft prompt; the base model is shared.
- **Scales with model size** — On very large models (T5-11B), prompt tuning matches full fine-tuning performance. On smaller models, the gap is significant.

```python
# Conceptual illustration with HuggingFace PEFT
from peft import PromptTuningConfig, get_peft_model, TaskType

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,          # Number of soft prompt tokens
    prompt_tuning_init="TEXT",      # Initialize from text: "Classify the sentiment:"
    prompt_tuning_init_text="Classify the sentiment of the following review:",
    tokenizer_name_or_path="gpt2",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 15,360 || all params: 124,454,400 || trainable%: 0.01
```

## Prefix Tuning

Introduced by Li & Liang (2021), **prefix tuning** is a more expressive variant that prepends learned vectors to the **key-value pairs of every Transformer layer's attention**, not just the input embeddings.

### How It Works

For each layer $l$ in the Transformer, prefix tuning prepends trainable prefix vectors $P^K_l$ and $P^V_l$ to the key and value matrices before computing attention:

$$\text{Attention}(Q, [P^K_l; K], [P^V_l; V])$$

This gives the soft prompt direct influence over the attention patterns at every layer — significantly more expressive than input-level prompt tuning.

The prefix parameters are typically reparameterized through a small MLP during training (to aid optimization), then fixed at inference.

### Architecture

```
Input Tokens → Embedding → Transformer Layer 1 → ... → Output
                               ↑
               [Prefix Keys₁ | Prefix Values₁] (learned)
                                          Transformer Layer 2
                               ↑
               [Prefix Keys₂ | Prefix Values₂] (learned)
                                          ...
```

### Trainable Parameter Count

For a model with $L$ layers, $d$ hidden dimensions, and prefix length $k$:

$$\text{Params} = 2 \times L \times k \times d$$

For GPT-2 (12 layers, $d=768$) with $k=10$: $2 \times 12 \times 10 \times 768 \approx 184K$ parameters vs. 117M total — a **636× reduction**.

## Prompt Tuning vs. Prefix Tuning vs. LoRA

| Method | Where learned params live | Expressiveness | Inference overhead |
|---|---|---|---|
| **Prompt Tuning** | Input embedding layer | Low | Minimal (longer input) |
| **Prefix Tuning** | K/V pairs, every layer | Higher | Minimal (extended K/V cache) |
| **LoRA** | Additive low-rank matrices in weight matrices | High | None (can be merged) |
| **Full Fine-Tuning** | All parameters | Highest | None |

## Initialization Strategies

Both methods benefit from thoughtful initialization:

- **Prompt tuning** initialized from task-relevant tokens (e.g., the instruction text) outperforms random initialization.
- **Prefix tuning** initialized by forward-passing real text through the model produces better starting points than random vectors.

## Practical Considerations

- **Prompt tuning works best at scale** — Below ~1B parameters, LoRA or adapter methods typically outperform it.
- **Prefix tuning** is more competitive at moderate scales due to its deeper influence over the model.
- **Inference**: both methods add a few tokens' worth of computation and cache; this is negligible for long contexts.
- **Multi-task serving**: a single pretrained backbone with task-specific soft prompts stored per task is an efficient deployment pattern — one copy of model weights, $N$ lightweight task adapters.

## When to Use Them

- **Prompt tuning**: Very large models (>10B params), inference environments where model switching is expensive, and you want the simplest possible adaptation.
- **Prefix tuning**: Moderate-scale models, generation tasks (summarization, translation) where deep guidance improves output structure.
- **LoRA**: The most broadly recommended PEFT method today — effective at all scales and easy to merge at inference time.

Prefix and prompt tuning were pioneering contributions to parameter-efficient adaptation, paving the way for the rich PEFT ecosystem that now enables fine-tuning of frontier models with consumer hardware.
