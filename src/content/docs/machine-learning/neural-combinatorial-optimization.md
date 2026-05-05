---
title: Neural Combinatorial Optimization
description: A comprehensive guide to neural approaches for combinatorial optimization problems, covering pointer networks, attention models, graph neural networks for routing and scheduling, and hybrid learned-heuristic methods.
---

# Neural Combinatorial Optimization

Combinatorial optimization — finding the best solution from a finite but astronomically large set of candidates — underlies logistics, scheduling, circuit design, and protein folding. Classical solvers (branch-and-bound, dynamic programming, metaheuristics) struggle to scale or require extensive domain expertise. Neural Combinatorial Optimization (NCO) trains deep learning models to directly produce high-quality solutions, learning heuristics from data rather than hand-crafting them.

## Problem Classes

NCO has been applied to:

- **Travelling Salesman Problem (TSP)**: find the shortest route visiting all cities exactly once
- **Vehicle Routing Problem (VRP)**: TSP with capacity constraints and multiple depots
- **Job Shop Scheduling**: assign jobs to machines to minimize makespan
- **Bin Packing**: pack items into minimum number of bins
- **Maximum Cut**: partition graph nodes to maximize cut edges
- **Knapsack**: maximize value within weight constraint

## Pointer Networks (Vinyals et al., 2015)

The foundational NCO architecture extends sequence-to-sequence models with an **attention-based output mechanism** that points to positions in the input rather than generating from a fixed vocabulary.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def attention(self, encoder_out: torch.Tensor, decoder_h: torch.Tensor) -> torch.Tensor:
        # encoder_out: (B, N, H), decoder_h: (B, 1, H)
        scores = self.v(torch.tanh(self.W1(encoder_out) + self.W2(decoder_h)))
        return scores.squeeze(-1)   # (B, N) — unnormalized pointer scores

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (B, N, input_dim) — N cities
        B, N, _ = x.shape
        enc_out, (h, c) = self.encoder(x)

        # Autoregressive decoding: select cities one by one
        tours, log_probs = [], []
        visited = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        dec_input = torch.zeros(B, 1, enc_out.size(-1), device=x.device)

        for _ in range(N):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            scores = self.attention(enc_out, dec_out)
            scores[visited] = float("-inf")         # mask visited cities
            probs = F.softmax(scores, dim=-1)

            city = probs.multinomial(1)             # sample next city
            log_probs.append(probs.gather(1, city).log())
            tours.append(city)

            visited.scatter_(1, city, True)
            dec_input = enc_out.gather(1, city.unsqueeze(-1).expand(-1, -1, enc_out.size(-1)))

        tours = torch.cat(tours, dim=1)             # (B, N)
        log_probs = torch.cat(log_probs, dim=1).sum(dim=1)   # (B,)
        return tours, log_probs
```

Training uses **REINFORCE** with baseline (rollout or critic):

```python
def reinforce_loss(log_probs: torch.Tensor, rewards: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    advantage = (rewards - baseline).detach()
    return -(advantage * log_probs).mean()
```

## Attention Model (Kool et al., 2019)

The Attention Model (AM) replaces LSTM with Transformers and achieves near-optimal TSP solutions:

```python
class AttentionModel(nn.Module):
    """Multi-head attention encoder + context-based decoder for TSP/VRP."""
    def __init__(self, node_dim: int = 2, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Linear(node_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.W_q = nn.Linear(embed_dim * 3, embed_dim)  # context: [graph, first, last]
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, embed_dim)
        self.clip = 10.0   # logit clipping

    def forward(self, coords: torch.Tensor) -> tuple:
        # coords: (B, N, 2) — 2D city coordinates
        h = self.encoder(self.embed(coords))    # (B, N, D)
        graph_embed = h.mean(dim=1)             # (B, D)
        return self._decode(h, graph_embed)

    def _decode(self, h, graph_embed):
        B, N, D = h.shape
        first = torch.zeros(B, D, device=h.device)
        last = torch.zeros(B, D, device=h.device)
        tours, log_probs = [], []
        visited = torch.zeros(B, N, dtype=torch.bool, device=h.device)

        for _ in range(N):
            ctx = torch.cat([graph_embed, first, last], dim=-1)   # (B, 3D)
            q = self.W_q(ctx).unsqueeze(1)                        # (B, 1, D)
            k = self.W_k(h)                                       # (B, N, D)
            logits = (q @ k.transpose(1, 2)).squeeze(1) / D**0.5  # (B, N)
            logits = self.clip * torch.tanh(logits)
            logits[visited] = float("-inf")
            probs = torch.softmax(logits, dim=-1)

            city = probs.multinomial(1)
            log_probs.append(probs.gather(1, city).log())
            tours.append(city)
            visited.scatter_(1, city, True)

            selected = h.gather(1, city.unsqueeze(-1).expand(-1, -1, D)).squeeze(1)
            last = selected
            if len(tours) == 1:
                first = selected

        return torch.cat(tours, dim=1), torch.cat(log_probs, dim=1).sum(1)
```

## POMO: Policy Optimization with Multiple Optima

POMO (Kwon et al., 2020) exploits the **symmetry** of combinatorial problems: a TSP tour is valid regardless of which city you start from. Training with $N$ start positions simultaneously provides $N$ gradient estimates per instance:

```python
def pomo_loss(model, coords, n_starts=None):
    B, N, _ = coords.shape
    n_starts = n_starts or N

    all_rewards, all_log_probs = [], []
    for start in range(n_starts):
        tours, log_probs = model(coords, start_node=start)
        rewards = -compute_tour_length(coords, tours)   # negative distance
        all_rewards.append(rewards)
        all_log_probs.append(log_probs)

    all_rewards = torch.stack(all_rewards, dim=1)       # (B, n_starts)
    all_log_probs = torch.stack(all_log_probs, dim=1)   # (B, n_starts)

    baseline = all_rewards.mean(dim=1, keepdim=True)    # POMO baseline
    advantage = all_rewards - baseline
    loss = -(advantage * all_log_probs).mean()
    return loss
```

## GNN for Combinatorial Optimization

Graph Neural Networks naturally represent combinatorial problem structure:

```python
import torch_geometric.nn as gnn

class TSPGraphNet(nn.Module):
    def __init__(self, node_features: int, hidden: int = 256, n_layers: int = 6):
        super().__init__()
        self.embed = nn.Linear(node_features, hidden)
        self.convs = nn.ModuleList([
            gnn.GATv2Conv(hidden, hidden // 8, heads=8, concat=True)
            for _ in range(n_layers)
        ])
        self.edge_classifier = nn.Linear(hidden * 2, 1)  # edge in tour: yes/no

    def forward(self, x, edge_index):
        h = self.embed(x)
        for conv in self.convs:
            h = F.elu(conv(h, edge_index) + h)  # residual
        # Predict edge probability for each candidate edge
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=-1)
        return torch.sigmoid(self.edge_classifier(edge_features))
```

Edge predictions can be used to guide classical solvers (LKH, Concorde) by pruning the search space — a hybrid approach that achieves near-optimal solutions on large instances.

## Benchmark Results

On TSP with 100 cities (gap to optimal Concorde):

| Method | Gap (%) | Time per instance |
|---|---|---|
| Nearest neighbor heuristic | 20–25 | <1 ms |
| LKH-3 (classical) | 0.0 | ~10 s |
| Pointer Network (2015) | 8.3 | <1 ms |
| Attention Model (2019) | 4.5 | <1 ms |
| POMO (2020) | 0.14 | <1 ms |
| POMO + EAS (2022) | 0.04 | ~0.1 s |
| Neural + LKH hybrid | 0.02 | ~1 s |

## Constructive vs. Improvement Methods

**Constructive** methods build a solution from scratch (pointer networks, attention model). **Improvement** methods start from a feasible solution and iteratively refine it:

```python
class NeuralLocalSearch(nn.Module):
    """Learn to select 2-opt moves that improve the current solution."""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Linear(4, embed_dim)   # node pair features
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tour_embeddings: torch.Tensor) -> torch.Tensor:
        # Score all O(N^2) 2-opt swap candidates
        N = tour_embeddings.size(0)
        i_idx = torch.arange(N).repeat_interleave(N)
        j_idx = torch.arange(N).repeat(N)
        pairs = torch.cat([tour_embeddings[i_idx], tour_embeddings[j_idx]], dim=-1)
        return self.scorer(pairs).squeeze(-1).view(N, N)
```

## Practical Considerations

- **Generalization**: models trained on instances of size $N$ often generalize poorly to much larger instances; active research on size-agnostic architectures
- **Optimality gap**: NCO rarely matches exact solvers on large instances; hybrid neural+classical approaches close the gap
- **Training cost**: RL training for NCO is computationally expensive; supervised learning on optimal solutions (from Concorde) converges faster but requires labeled data
- **Problem-specific priors**: encoding domain structure (e.g., time windows for VRP) in the model significantly improves performance

## Summary

Neural combinatorial optimization has progressed from Pointer Networks with ~8% optimality gap to POMO-based models achieving <0.1% gap on TSP-100 — all while running in milliseconds per instance. The most effective production systems combine neural heuristics for fast initial solutions with classical solvers for refinement, exploiting the complementary strengths of both paradigms. As problem sizes grow and real-world constraints multiply (time windows, precedence, stochasticity), NCO methods that generalize across problem families and instance sizes remain an active and important research frontier.
