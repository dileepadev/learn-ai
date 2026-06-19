---
title: Mamba-2 and State Space Duality (SSD)
description: Explore the Mamba-2 architecture and State Space Duality (SSD), which bridges the gap between state space models and attention mechanisms.
---

The Mamba-1 architecture introduced structured State Space Models (SSMs) as a viable alternative to Transformers, achieving linear computational scaling $\mathcal{O}(N)$ with sequence length while maintaining high quality. 

**Mamba-2** refines this design by introducing **State Space Duality (SSD)**. SSD establishes a theoretical connection between structured SSMs and Self-Attention mechanisms. This duality allows Mamba-2 to utilize hardware-efficient matrix multiplication kernels (Tensor Cores) and support tensor parallelism, achieving up to 8x faster training speeds compared to Mamba-1.

---

## The Concept of State Space Duality (SSD)

The key insight of State Space Duality is that structured State Space Models and variants of linear attention are two representations of the **same underlying mathematical operator**.

A structured SSM maps an input sequence $x(t) \in \mathbb{R}$ to an output sequence $y(t) \in \mathbb{R}$ through a latent state $h(t) \in \mathbb{R}^N$:

$$h'(t) = A(t) h(t) + B(t) x(t)$$

$$y(t) = C(t) h(t)$$

SSD proves that if we restrict the transition matrix $A$ to a scalar diagonal structure (where $A_t = a_t I$), this SSM can be written as a dual formulation that resembles a **Linear Attention** mechanism:

$$Y = (L \circ (C B^T)) X$$

Where:
- $X, Y$ are the input and output sequences.
- $C, B$ are the projection matrices (analogous to Query and Key in attention).
- $L$ is a semi-separable matrix representing the decay factors $a_t$ over time.
- $\circ$ denotes the Hadamard (element-wise) product.

By converting the SSM computation into this dual matrix multiplication form, Mamba-2 can compute its recurrent updates using block-wise matrix multiplications instead of sequential scans.

---

## Architectural Changes in Mamba-2

Mamba-2 modifies the block structure of Mamba-1 to optimize it for modern hardware.

```
Mamba-1 Block:
Input ---> Linear ---> Conv1d ---> SSM ---> Act ---> Linear ---> Output

Mamba-2 Block (SSD Parallelized):
Input ---> Projection Matrix (Q, K, V, A) 
           |---> Conv1d ---> SSD (Matrix Mult) ---> Output
```

1. **Parallel Projections:** In Mamba-1, input projections and parameter calculations (for $A, B, C$) occurred in different stages. Mamba-2 uses a single parallel projection layer to generate $Q$ (from $C$), $K$ (from $B$), $V$ (from $x$), and the gate parameters in one step, similar to standard Attention.
2. **Tensor Parallelism:** By aligning the structured SSM with linear attention, Mamba-2 can split the channels of $Q, K, V$ across multiple GPUs using standard Megatron-style Tensor Parallelism, which was difficult to implement in Mamba-1.
3. **Larger State Dimensions:** Because SSD matrix multiplications are highly efficient, Mamba-2 can increase the latent state dimension $N$ (from $16$ in Mamba-1 to $64$ or $128$ in Mamba-2) without a computational penalty, improving the model's memory recall capabilities.

---

## Mamba-2 vs. Mamba-1 vs. Transformer

| Feature | Transformer | Mamba-1 | Mamba-2 (SSD) |
|---|---|---|---|
| **Complexity** | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ |
| **Inference Latency** | High (KV Cache grows) | Low (Constant State) | Low (Constant State) |
| **Hardware Core Fit** | High (Matrix Mult) | Low (Sequential Scan) | High (Block Matrix Mult) |
| **Parallel Training** | Easy (Attention Map) | Medium (Associative Scan) | Easy (State Space Duality) |
| **Tensor Parallelism** | Supported natively | Challenging | Supported natively |

---

## Code Concept: SSD Block-Wise Forward Pass

Below is a conceptual PyTorch snippet illustrating how State Space Duality evaluates a block of sequence tokens using matrix multiplications instead of recurrent loops.

```python
import torch
import torch.nn as nn

class SSDAttentionDual(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_state = d_state
        # Parallel projections for Q, K, V, and dt (step size)
        self.in_proj = nn.Linear(d_model, d_model * 3 + d_state)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape
        
        # 1. Project inputs
        projected = self.in_proj(x)
        q, k, v, dt = torch.split(projected, [d_model, d_model, d_model, self.d_state], dim=-1)
        
        # 2. Compute decay matrix L based on step size dt
        # In a real implementation, this utilizes the associative prefix sum
        dt_decay = torch.exp(-dt) # [batch, seq_len, d_state]
        
        # 3. Compute block-wise SSD kernel (Dual Formulation)
        # S = (K^T * V) decays over time, then Q * S
        # This is computed using highly optimized Triton GPU kernels
        k_normalized = k / torch.norm(k, dim=-1, keepdim=True)
        q_normalized = q / torch.norm(q, dim=-1, keepdim=True)
        
        # Compute linear attention output
        attention_weights = torch.matmul(q_normalized, k_normalized.transpose(-1, -2))
        # Apply decay mask representing time-series dependencies
        decay_mask = self.get_decay_mask(dt_decay, seq_len)
        attention_weights = attention_weights * decay_mask
        
        out = torch.matmul(attention_weights, v)
        return out

    def get_decay_mask(self, dt_decay, seq_len):
        # Helper to compute SSM decay matrix
        # mask[i, j] = exp(-sum(dt[j:i])) for i > j
        mask = torch.ones(seq_len, seq_len, device=dt_decay.device)
        return torch.tril(mask)
```
