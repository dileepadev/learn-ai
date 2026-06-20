---
title: Representation Finetuning (ReFT) & LoReFT
description: Learn about Representation Finetuning (ReFT) and LoReFT, a parameter-efficient fine-tuning approach that alters representation subspaces instead of model weights.
---

Most Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA or prefix tuning, adapt language models by modifying the weights of their layers. 

**Representation Finetuning (ReFT)** takes a fundamentally different path. Rather than editing the model's weights, ReFT operates directly on the **hidden representations** (activations) flowing through the layers during the forward pass. By learning low-rank interventions on a small subspace of activations, ReFT methods like **Low-Rank Representation Tuning (LoReFT)** achieve comparable performance to LoRA while using up to 50x fewer parameters.

---

## The Core Concept: Subspace Interventions

ReFT is built on the concept of **subspace interventions** from mechanistic interpretability. When a transformer processes a prompt, it represents high-level concepts as directions or subspaces in its intermediate activation vectors (residual stream).

Instead of changing how the model computes activations (by altering weights), ReFT surgically alters the activations directly:

$$h_{\text{new}} = h + \Phi(h)$$

Where $h \in \mathbb{R}^d$ is the hidden representation at a specific layer and position, and $\Phi$ is an intervention function.

---

## Low-Rank Representation Tuning (LoReFT)

LoReFT is a specific implementation of ReFT that limits the intervention to a low-rank subspace of the activations using a linear projection.

Given a hidden vector $h \in \mathbb{R}^d$, the LoReFT intervention at layer $L$ is defined as:

$$\text{LoReFT}(h) = h + R^T \left( \sigma(R h + b) - R h \right)$$

Where:
- $R \in \mathbb{R}^{r \times d}$ is a low-rank projection matrix (rank $r \ll d$).
- $b \in \mathbb{R}^r$ is a learnable bias vector.
- $\sigma$ is an activation function (often a Sigmoid or GeLU).

### The Intervention Target
Unlike weight adapters that process every token in the sequence, LoReFT is typically applied only to **specific token positions** (e.g., the last $K$ tokens of the prompt) at a subset of layers. This highly targeted approach reduces the number of trained parameters to a fraction of LoRA's requirements.

```
Token Flow:
[Token 1] ---> Layer 1 ---> Layer 2 ---> Intervention ---> Layer 3
[Token 2] ---> Layer 1 ---> Layer 2 ---> Intervention ---> Layer 3
                                             ^
                                       Learnable (R, b)
```

During training, only the low-rank projection matrices $R$ and bias vectors $b$ are updated via gradient descent; the entire base model remains frozen.

---

## Why LoReFT is Highly Efficient

1. **Parameters Scale:** A typical LoRA setup ($r=8$) on an 8B model requires roughly 10 million parameters. A LoReFT setup ($r=4$) on the same model requires only around 50,000 parameters.
2. **Reduced Gradient Compute:** Because interventions are applied at specific layers and token positions, the gradient computation graph is significantly smaller, reducing backpropagation memory overhead.
3. **Subspace Editing:** It directly edits the "thoughts" of the model, bypassing the complex, indirect process of altering weights to achieve a target representation change.

---

## Code Concept: Simulating a LoReFT Layer

Below is a PyTorch-like illustration of how a LoReFT intervention acts on transformer hidden states during a forward pass.

```python
import torch
import torch.nn as nn

class LoReFTIntervention(nn.Module):
    def __init__(self, embed_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.embed_dim = embed_dim
        
        # Low-rank projection projection matrix R (rank x embed_dim)
        self.R = nn.Parameter(torch.randn(rank, embed_dim) * 0.02)
        # Learnable bias
        self.b = nn.Parameter(torch.zeros(rank))
        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, h):
        # h shape: [batch, seq_len, embed_dim]
        # Standard formulation: h_new = h + R^T * (act(R*h + b) - R*h)
        
        # 1. Project to low-rank subspace
        low_rank = torch.matmul(h, self.R.t()) # [batch, seq_len, rank]
        
        # 2. Apply non-linear editing
        edited = self.act(low_rank + self.b)
        
        # 3. Project back to embedding dimension
        projected_back = torch.matmul(edited - low_rank, self.R) # [batch, seq_len, embed_dim]
        
        # 4. Apply residual addition
        h_new = h + projected_back
        return h_new
```

---

## Implementing ReFT with PyReFT

Researchers have released the `pyreft` library, which integrates seamlessly with Hugging Face models to run ReFT training.

```python
import pyreft
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 2. Set up ReFT config
reft_config = pyreft.ReftConfig(
    representations={
        "layer": 15,                     # Apply intervention at layer 15
        "component": "block_output",      # Target block output (residual stream)
        "low_rank_dimension": 4,          # Subspace rank r = 4
        "intervention": pyreft.LoreftIntervention(
            embed_dim=4096, 
            low_rank_dimension=4
        )
    }
)

# 3. Wrap model with ReFT intervention hooks
reft_model = pyreft.get_reft_model(model, reft_config)

# Only ReFT parameters are trainable
reft_model.print_trainable_parameters()
```
