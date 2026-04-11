---
title: Mixture of Depths
description: Explore Mixture of Depths, a technique that enables Transformer models to dynamically allocate compute per token and per layer — processing only the most important tokens at each depth while skipping trivial ones.
---

Mixture of Depths (MoD) is a dynamic computation technique for Transformer models that allows the network to **skip transformer blocks for less important tokens**, allocating full compute only where it is most needed. Proposed by Raposo et al. (2024) at Google DeepMind, MoD complements Mixture of Experts (MoE) by adding depth-wise conditional computation to existing parameter efficiency techniques.

## The Motivation: Compute Is Not Equally Needed Everywhere

In a standard Transformer, **every token passes through every layer with the same compute cost**, regardless of whether that token requires complex reasoning or is trivially predictable.

Consider processing the sentence: *"The cat sat on the mat."*

- "The" and "on" are common function words — their next-token probabilities are easily computed
- "sat" carries semantic weight and requires more contextual integration

Standard Transformers cannot differentiate — all tokens receive equal processing at every layer. MoD makes this heterogeneous by design.

## How Mixture of Depths Works

MoD modifies standard Transformer layers to include a **routing decision** at each depth. For each token at each layer, the router decides:

1. **Process token normally** through the full attention + FFN sublayers
2. **Skip the layer** for this token using a residual bypass

The routing is implemented as a learned **top-$k$ selection** over the sequence:

```
For layer l:
  router_scores = Linear(x)         # (batch, seq_len) scalar score per token
  top_k indices = topk(router_scores, k=T)   # select top-T% of tokens
  
  for token in top_k:
      x[token] = TransformerBlock(x[token])  # full compute
  for token not in top_k:
      x[token] = x[token]                    # skip (identity)
```

The capacity $T$ is a hyperparameter specifying what fraction of tokens are processed at each layer.

## Routing Mechanisms

### Token-Choice Routing

Each token independently scores itself. The top-$k$ tokens by score are processed; the rest skip.

**Challenge:** Causal language modeling requires every token to be predictable as a future token. MoD with token-choice routing must ensure that the routing decision for token $t$ only depends on tokens $\leq t$.

### Learned vs. Fixed Capacity

- **Learned capacity:** The router trains end-to-end; it learns which tokens need processing
- **Fixed capacity:** A predetermined fraction of tokens is processed, and the router ranks them

Both approaches use an **auxiliary loss** to prevent router collapse (all tokens always routed the same way).

## Autoregressive Efficiency: Training vs. Inference

### Training

During training, MoD uses the full batch in parallel. Token routing decisions can be computed efficiently because all sequence positions are available simultaneously, making the parallel scan over tokens inexpensive.

### Inference

Autoregressive decoding generates one token at a time. The router must decide whether to skip a layer for the **current generated token** — a binary decision without batch parallelism. This can be done with minimal overhead since the router is just a small linear layer.

In practice, MoD-trained models can achieve:

- **Same quality as full-compute models** at significantly reduced FLOPs
- Or **better quality at the same FLOPs** by using the saved compute to increase model size elsewhere

## MoD vs. Related Techniques

| Technique | What it Skips | Axis |
|---|---|---|
| Mixture of Experts (MoE) | Experts within a layer | Width (parameters) |
| Mixture of Depths (MoD) | Full layers for some tokens | Depth (layers) |
| Early Exit | Remaining layers for easy inputs | Entire depth suffix |
| Layer Skipping | Fixed layers based on task | Fixed depth subset |

MoD and MoE are **orthogonal and composable**: a model can be simultaneously MoD (token routing over depth) and MoE (expert routing over width), further scaling efficiency.

## Empirical Results

The original MoD paper demonstrated:

- A 12.5% compute reduction at the same model quality (C4 language modeling perplexity)
- Learned routing naturally assigns more compute to semantic content tokens than function words
- MoD models train stably without the load-balancing challenges of MoE

Routing analysis showed that higher layers (deep in the network) were more selective — agreeing with prior work showing that early layers handle syntax while later layers handle semantics. Semantic tokens received more compute in deeper layers.

## Combination with isoFLOP Analysis

The paper introduces **isoFLOP** comparisons — comparing architectures with different designs but identical total floating-point operation budgets. Under isoFLOP constraints:

MoD models outperform standard Transformers when:

- Model size and sequence length are large enough for routing overhead to be amortized
- Sequences have heterogeneous information density (common in natural language, code)

They perform similarly when sequences are very short or highly uniform.

## Implementation Sketch

```python
class MoDTransformerBlock(nn.Module):
    def __init__(self, d_model, capacity_factor=0.5):
        self.router = nn.Linear(d_model, 1)
        self.attention = MultiHeadAttention(d_model)
        self.ffn = FeedForward(d_model)
        self.capacity = capacity_factor  # fraction of tokens to process

    def forward(self, x):
        B, S, D = x.shape
        k = int(S * self.capacity)
        
        # Compute routing scores
        scores = self.router(x).squeeze(-1)  # (B, S)
        
        # Select top-k tokens
        topk_idx = scores.topk(k, dim=-1).indices  # (B, k)
        
        # Process only selected tokens
        selected = x[:, topk_idx]  # simplified indexing
        processed = self.attention(selected) + selected
        processed = self.ffn(processed) + processed
        
        # Write back; unselected tokens pass through unchanged
        x[:, topk_idx] = processed
        return x
```

## Broader Impact: Towards Adaptive Computation

MoD is part of a broader research direction toward **adaptive computation** — models that allocate resources proportional to task difficulty:

- **Adaptive softmax:** Variable compute for frequent vs. rare vocabulary
- **Adaptive attention span:** Variable attention window per head
- **MoE:** Variable parameter activation per token
- **Test-time compute (chain-of-thought, self-consistency):** Variable reasoning depth at inference

Together, these represent a shift from static, uniform neural architectures toward **compute-on-demand intelligence**.

## Further Reading

- Raposo et al. (2024), *Mixture of Depths: Dynamically Allocating Compute in Transformer Language Models*
- Fedus et al. (2022), *Switch Transformers: Scaling to Trillion Parameter Models* (MoE)
- Graves (2016), *Adaptive Computation Time for Recurrent Neural Networks* (early adaptive compute)
- Schuster et al. (2022), *Confident Adaptive Language Modeling* (early exit)
