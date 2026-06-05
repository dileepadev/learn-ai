---
title: "LoRA: Low-Rank Adaptation for Efficient Fine-Tuning"
description: "Learn about LoRA, a parameter-efficient fine-tuning method that adds trainable low-rank matrices to frozen pretrained weights."
date: "2026-03-20"
tags: ["deep-learning", "fine-tuning", "parameter-efficient", "llms"]
---

LoRA (Low-Rank Adaptation) is a technique for fine-tuning large language models efficiently by adding small, trainable low-rank matrices to pretrained weights while keeping the original weights frozen. This reduces memory requirements and allows fine-tuning on consumer hardware.

## The Problem with Full Fine-Tuning

When fine-tuning a large model like LLaMA-7B:
- The model has 7 billion parameters
- Full fine-tuning requires storing gradients, optimizer states, and activations for all parameters
- This can require 100+ GB of GPU memory

LoRA solves this by recognizing that the change in weights during fine-tuning often has low intrinsic rank.

## LoRA Core Idea

For a pretrained weight matrix W ∈ ℝ^(d×k), instead of updating W directly:

```
Updated weight: W + ΔW
```

LoRA factorizes the update:

```
ΔW = B × A  where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k)
```

- r (rank) is typically 1-32
- Total trainable parameters: d×r + r×k = r(d+k)
- For r=8: reduces parameters by ~99% compared to full fine-tuning

## LoRA Implementation

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices (initialized to zeros/random)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.register_buffer('disabled', torch.tensor(1.0))
        
        # Initialize A with small random, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, original_weights, x):
        # Compute LoRA adaptation
        lora_adapter = (x @ self.lora_A) @ self.lora_B
        lora_adapter = lora_adapter * self.scaling
        
        # Return original + adaptation
        return original_weights(x) + lora_adapter

# Example: Apply LoRA to a linear layer
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        self.original = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.lora(self.original, x)
```

## Applying LoRA to Transformers

```python
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Apply LoRA
model = peft.get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,741,717,760 || trainable%: 0.062%
```

## Practical Considerations

**Choosing the rank:**
- Lower rank = fewer parameters but less expressivity
- Higher rank = more capacity but diminishing returns
- Start with r=8 and experiment

**Target modules:**
- For attention-based models, typically target query and value projections
- Can also target down/up projections in MLP layers

**Combining with quantization:**
- LoRA works well with 4-bit quantization (QLoRA)
- Enables fine-tuning 70B models on a single GPU

LoRA has become the dominant approach for fine-tuning LLMs, enabling efficient adaptation without sacrificing much performance.