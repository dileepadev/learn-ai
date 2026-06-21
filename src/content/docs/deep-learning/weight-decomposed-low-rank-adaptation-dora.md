---
title: Weight-Decomposed Low-Rank Adaptation (DoRA)
description: Explore DoRA (Weight-Decomposed Low-Rank Adaptation), a parameter-efficient fine-tuning technique that decomposes weights into magnitude and direction components, matching or exceeding full fine-tuning performance.
---

**Weight-Decomposed Low-Rank Adaptation (DoRA)** is a Parameter-Efficient Fine-Tuning (PEFT) technique introduced to bridge the performance gap between Standard Low-Rank Adaptation (LoRA) and full fine-tuning. 

By decomposing pre-trained weights into magnitude and directional components and updating only the direction using low-rank matrices, DoRA achieves performance that matches or sometimes surpasses full fine-tuning while maintaining LoRA's inference efficiency.

---

## The Core Concept: Weight Decomposition

A weight matrix $W_0 \in \mathbb{R}^{d \times k}$ can be decomposed into a magnitude vector $m \in \mathbb{R}^{1 \times k}$ and a directional matrix $V \in \mathbb{R}^{d \times k}$:

$$W = m \cdot \frac{V}{\|V\|_c}$$

Where $\|\cdot\|_c$ denotes the vector $L_2$ norm of each column. 

In full fine-tuning, both $m$ and $V$ are updated freely. In standard LoRA, the weight matrix is updated as:

$$W = W_0 + \Delta W = W_0 + B A$$

Where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are low-rank matrices. However, LoRA updates magnitude and direction together in a coupled manner, which differs fundamentally from the learning dynamics of full fine-tuning.

DoRA solves this by applying LoRA updates **only to the directional component** while keeping the magnitude component parameterized separately.

---

## How DoRA Works

DoRA initializes its magnitude vector $m$ from the pre-trained weights $W_0$:

$$m = \|W_0\|_c$$

The directional matrix is initialized as $V = W_0$. During fine-tuning, the directional matrix $V$ is updated using a standard low-rank LoRA adapter $\Delta V = B A$:

$$W = m \cdot \frac{W_0 + B A}{\|W_0 + B A\|_c}$$

Here:
- $m$ (magnitude) is a learnable vector of size $1 \times k$.
- $B$ and $A$ are learnable low-rank matrices (initialized such that $BA = 0$).
- $W_0$ remains frozen.

By splitting the updates into magnitude and direction, DoRA allows the optimizer to adjust magnitude adjustments and direction adjustments independently. Empirical studies show that full fine-tuning tends to make updates that are highly decoupled (large changes in direction with minimal changes in magnitude, or vice versa). DoRA replicates this decoupling behavior, whereas standard LoRA shows a strong positive correlation between magnitude and direction updates.

---

## Key Advantages of DoRA

1. **Closer to Full Fine-Tuning:** DoRA matches full fine-tuning performance on complex tasks (e.g., math reasoning, instruction following) where standard LoRA often falls short.
2. **No Extra Inference Latency:** Just like LoRA, the adapters ($B$ and $A$) and the magnitude vector ($m$) can be folded back (merged) into the base weight matrix $W_0$ before deployment.
3. **Robustness to Rank Size:** DoRA performs exceptionally well even at lower ranks ($r = 4$ or $r = 8$), reducing the memory overhead of training.

---

## Implementing DoRA with PEFT

Hugging Face's `peft` library supports DoRA natively. You simply need to set `use_dora=True` in the `LoraConfig`.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load base model and tokenizer
model_id = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Define LoRA Config with DoRA enabled
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True,  # <-- Activates DoRA instead of standard LoRA
)

# 3. Wrap model with PEFT
dora_model = get_peft_model(model, peft_config)

# Print trainable parameters to verify efficiency
dora_model.print_trainable_parameters()
```

---

## Tuning Tips for DoRA

- **Learning Rate:** DoRA can benefit from slightly higher learning rates compared to standard LoRA, as the magnitude parameters learn quickly and stabilize training.
- **Target Modules:** Just like LoRA, targeting all linear layers (Q, K, V, O, gate, up, down projections) yields the best performance.
- **Rank Selection:** If you have memory constraints, try reducing the rank $r$ to $4$ or $8$ first before sacrificing target modules, as DoRA maintains performance better than LoRA at lower ranks.
