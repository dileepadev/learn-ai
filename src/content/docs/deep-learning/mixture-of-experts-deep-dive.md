---
title: "Mixture of Experts: Architecture and Training"
description: "Deep dive into mixture of experts (MoE) architectures — sparse gating, load balancing, and practical implementation."
date: "2026-06-06"
tags: ["deep-learning", "transformers", "mixture-of-experts", "scaling"]
---

Mixture of Experts (MoE) is a neural network architecture that combines multiple expert subnetworks with a gating mechanism that selects which experts to activate for each input. This enables **sparse activation**: only a small fraction of parameters are used per forward pass, allowing models to scale to billions or trillions of parameters while maintaining computational efficiency.

## The MoE Concept

A standard dense transformer layer computes:

$$y = \text{FFN}(x)$$

An MoE layer replaces the dense FFN with multiple experts and a gating network:

$$y = \sum_{i=1}^{E} G(x)_i \cdot E_i(x)$$

Where:
- $E$ is the number of experts
- $G(x)$ is the gating network output (softmax over expert weights)
- $E_i(x)$ is the output of expert $i$

The key insight: **parameters scale with number of experts, but computation scales with expert usage**. This breaks the usual parameter-computation coupling in transformers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    """
    Mixture of Experts layer with top-k gating.
    
    Instead of activating all experts, we:
    1. Compute gating scores for all experts
    2. Select top-k experts (k=1 or 2 typically)
    3. Only compute forward pass for selected experts
    4. Combine outputs weighted by gating scores
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 16, k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        
        # Create expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Gating network (simple linear + softmax)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Flatten sequence for batch processing
        x_flat = x.view(-1, d_model)  # (batch * seq, d_model)
        
        # Compute gating scores
        gating_scores = self.gate(x_flat)  # (batch * seq, num_experts)
        gating_weights = F.softmax(gating_scores, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(gating_weights, self.k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)  # Renormalize
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Compute output from each expert
        for expert_idx in range(self.num_experts):
            # Get indices where this expert is in top-k
            mask = (topk_indices == expert_idx).any(dim=-1)
            if mask.sum() == 0:
                continue
            
            # Compute expert output
            expert_out = self.experts[expert_idx](x_flat[mask])
            
            # Weight by gating score (aggregate from both top-k positions)
            weights = torch.zeros(mask.sum(), device=x.device)
            for pos in range(self.k):
                pos_mask = (topk_indices[mask] == expert_idx).any(dim=-1)
                weights += topk_weights[mask][pos_mask].sum(dim=-1)
            
            output[mask] += expert_out * weights.unsqueeze(-1)
        
        # Reshape back
        return output.view(batch_size, seq_len, d_model)
```

## Gating Mechanisms

### Top-K Gating

The most common choice: select the k highest-scoring experts and softmax over them:

$$G(x)_i = \frac{\exp(\text{score}_i(x))}{\sum_{j \in \text{top-k}} \exp(\text{score}_j(x))}$$

Typically k=1 or k=2. k=1 is simpler but k=2 provides better coverage and learning signals.

```python
class TopKGating(nn.Module):
    """Top-k gating network with load balancing."""
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        # Compute raw gating scores
        scores = self.gate(x)  # (batch, num_experts)
        
        # Add noise for exploration during training
        if self.training:
            # Noise helps experts receive varied inputs during training
            noise = torch.rand_like(scores) / self.num_experts
            scores = scores + noise
        
        # Get top-k
        topk_weights, topk_indices = torch.topk(scores, self.k, dim=-1)
        
        # Softmax over selected experts only
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # Zero out non-top-k
        full_weights = torch.zeros_like(scores)
        full_weights.scatter_(-1, topk_indices, topk_weights)
        
        return full_weights, topk_indices
```

### Noisy Top-K Gating with Load Balancing

A critical problem in MoE: if the gate learns to always select the same experts, most experts never receive gradient updates. **Load balancing** adds an auxiliary loss to encourage even expert usage:

```python
class NoisyTopKGating(nn.Module):
    """Top-k gating with load balancing auxiliary loss."""
    def __init__(self, d_model: int, num_experts: int, k: int = 2, expert_temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.temperature = expert_temperature
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        scores = self.gate(x)
        
        # Add noise during training
        if self.training:
            noise = torch.rand_like(scores)
            scores = scores - self.temperature * torch.log(-torch.log(noise))
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # Compute load balancing loss
        # Ideally: each expert used ~1/num_experts fraction of time
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        expert_usage.scatter_add_(0, topk_indices.view(-1), torch.ones_like(topk_indices, dtype=torch.float))
        expert_usage = expert_usage / x.size(0)  # Normalize by batch size
        
        # Target: uniform distribution
        target = torch.ones_like(exper_usage) / self.num_experts
        
        # Load balancing loss: encourages uniform usage
        load_balance_loss = torch.sum(expert_usage * target) * self.num_experts**2
        
        return topk_weights, topk_indices, load_balance_loss


# Complete MoE layer with load balancing
class MoELayerWithLoss(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, k: int = 2):
        super().__init__()
        self.gating = NoisyTopKGating(d_model, num_experts, k)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor):
        weights, indices, load_loss = self.gating(x)
        
        # Compute expert outputs (only for selected experts)
        expert_outputs = []
        for expert_idx in range(len(self.experts)):
            mask = (indices == expert_idx).any(dim=-1)
            if mask.sum() > 0:
                expert_out = self.experts[expert_idx](x[mask])
                expert_outputs.append((mask, expert_idx, expert_out))
        
        # Combine outputs
        output = torch.zeros_like(x)
        for mask, expert_idx, expert_out in expert_outputs:
            # Get weights for this expert
            expert_weights = weights[mask, expert_idx].unsqueeze(-1)
            output[mask] += expert_out * expert_weights
        
        return output, load_loss
```

## Switch Transformer

The Switch Transformer (Fedus et al., 2022) uses k=1 routing, simplifying implementation and improving efficiency. The key insight: even with k=1, the model benefits from having many experts and selecting the best one.

```python
class SwitchTransformerLayer(nn.Module):
    """Switch Transformer: routing to single expert (k=1)."""
    def __init__(self, d_model: int, d_ff: int, num_experts: int, capacity_factor: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity = int(capacity_factor * d_ff / num_experts)  # Expert capacity
        
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Expert capacity buffer
        self.expert_usage = [0] * num_experts
        self.expert_capacity = self.capacity
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Routing
        routing_logits = self.gate(x_flat)
        routing_probs = F.softmax(routing_logits, dim=-1)
        expert_idx = torch.argmax(routing_probs, dim=-1)  # Single expert
        
        # Initialize expert buffers
        expert_outputs = [None] * self.num_experts
        expert_counts = [0] * self.num_experts
        
        # Assign tokens to experts
        for i, expert in enumerate(expert_idx):
            if expert_counts[expert] < self.capacity:
                if expert_outputs[expert] is None:
                    expert_outputs[expert] = []
                expert_outputs[expert].append((i, x_flat[i]))
                expert_counts[expert] += 1
        
        # Compute outputs
        output = torch.zeros_like(x_flat)
        routing_weights = torch.zeros(batch_size * seq_len, device=x.device)
        
        for expert_idx, tokens in enumerate(expert_outputs):
            if tokens is None:
                continue
            
            indices, token_embeds = zip(*tokens)
            tokens_batch = torch.stack(token_embeds, dim=0)
            
            expert_out = self.experts[expert_idx](tokens_batch)
            
            for j, idx in enumerate(indices):
                output[idx] = expert_out[j]
                routing_weights[idx] = routing_probs[idx, expert_idx]
        
        return output.view(batch_size, seq_len, d_model), routing_weights
```

## Expert Capacity and Dropping

In practice, we limit how many tokens each expert can process (capacity). Tokens beyond capacity are processed by the default expert (often the first or a learned fallback):

```python
class CapacityConstrainedMoE(nn.Module):
    """MoE with explicit capacity constraints and token dropping."""
    def __init__(self, d_model: int, d_ff: int, num_experts: int, k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity = int(capacity_factor * d_ff / num_experts)
        
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.size(0)
        
        # Compute routing
        routing = self.gate(x_flat)  # (num_tokens, num_experts)
        routing_weights, topk_idx = torch.topk(routing, self.k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Expert capacity tracking
        expert_counts = [0] * self.num_experts
        expert_outputs = {}
        dropped_tokens = []
        
        # Assign tokens to experts
        for token_idx in range(num_tokens):
            for pos in range(self.k):
                expert = topk_idx[token_idx, pos].item()
                weight = routing_weights[token_idx, pos].item()
                
                if expert_counts[expert] < self.capacity:
                    if expert not in expert_outputs:
                        expert_outputs[expert] = []
                    expert_outputs[expert].append((token_idx, x_flat[token_idx], weight))
                    expert_counts[expert] += 1
                else:
                    dropped_tokens.append(token_idx)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        
        for expert_idx, tokens_info in expert_outputs.items():
            indices, inputs, weights = zip(*tokens_info)
            inputs_batch = torch.stack(inputs, dim=0)
            expert_out = self.experts[expert_idx](inputs_batch)
            
            for j, (idx, w) in enumerate(zip(indices, weights)):
                output[idx] = expert_out[j] * w
        
        # Handle dropped tokens (use mean expert output)
        if dropped_tokens:
            mean_expert_out = torch.zeros_like(x_flat[0])
            for expert in self.experts:
                mean_expert_out = mean_expert_out + expert(x_flat[:1]).squeeze(0)
            mean_expert_out = mean_expert_out / len(self.experts)
            
            for idx in dropped_tokens:
                output[idx] = mean_expert_out
        
        return output.view(batch_size, seq_len, d_model)
```

## Training Considerations

### Auxiliary Losses

MoE training typically uses multiple auxiliary losses:

1. **Load balancing loss**: Encourages uniform expert usage
2. **Importance loss**: Encourages experts to have similar variance in usage
3. **Router z-loss**: Stabilizes gating by penalizing large logits

```python
def compute_moe_losses(routing_weights: torch.Tensor, indices: torch.Tensor):
    """Compute auxiliary losses for MoE training."""
    # Load balancing loss
    expert_usage = torch.bincount(indices.view(-1), minlength=routing_weights.size(-1))
    expert_usage = expert_usage / indices.size(0)
    target = torch.ones_like(expert_usage) / expert_usage.size(0)
    load_balance_loss = torch.sum(expert_usage * target) * len(expert_usage)**2
    
    # Importance loss (encourages experts to have similar variance)
    importance = routing_weights.sum(dim=0)
    importance_loss = torch.var(importance) * len(importance)
    
    return load_balance_loss, importance_loss
```

### Expert Specialization

During training, experts naturally specialize:

- Early layers: Learn general features (many experts active)
- Middle layers: Intermediate features (some specialization)
- Top layers: Task-specific features (strong specialization)

```python
class ExpertSpecializationTracker:
    """Track how much each expert is used across layers."""
    def __init__(self, num_layers: int, num_experts: int):
        self.usage = [[] for _ in range(num_layers)]
    
    def track(self, layer_idx: int, routing_weights: torch.Tensor):
        """Record expert usage for a layer."""
        # Average usage across batch
        avg_usage = routing_weights.mean(dim=0)
        self.usage[layer_idx].append(avg_usage.cpu().detach())
    
    def analyze_specialization(self):
        """Analyze how specialized experts have become."""
        for layer_idx, usages in enumerate(self.usage):
            if not usages:
                continue
            stacked = torch.stack(usages, dim=0)
            avg = stacked.mean(dim=0)
            entropy = -(avg * torch.log(avg + 1e-10)).sum()
            
            specialization = 1.0 - (entropy / torch.log(len(avg)))
            print(f"Layer {layer_idx}: specialization={specialization:.3f}")
```

## Practical Recommendations

- **Number of experts**: Start with 4-8 for small models, scale to 16-64 for large models
- **k value**: k=1 for simplicity and efficiency; k=2 for better gradient flow
- **Capacity factor**: 1.25-1.5 to handle load imbalance
- **Expert architecture**: Same as dense FFN (no need for additional complexity)
- **Initialization**: Gate should be initialized carefully to avoid collapsing
- **Auxiliary losses**: Load balancing loss is essential; others can help stability

MoE enables training trillion-parameter models with constant compute per token, making it the architecture of choice for frontier language models like GPT-4, Gemini, and Mixtral.