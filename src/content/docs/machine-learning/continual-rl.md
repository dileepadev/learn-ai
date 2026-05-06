---
title: Continual Reinforcement Learning
description: A comprehensive guide to Continual Reinforcement Learning — enabling RL agents to sequentially learn new tasks without forgetting previously acquired skills, covering catastrophic forgetting, architectural approaches, memory-based methods, and benchmarks.
---

# Continual Reinforcement Learning

**Continual Reinforcement Learning (CRL)** addresses the challenge of training RL agents that must sequentially acquire new tasks or adapt to non-stationary environments without forgetting previously learned skills. Unlike supervised continual learning where data comes in batches, RL agents face the additional complexity that forgetting can cascade — losing an early skill may prevent learning a later task that builds upon it.

## The Catastrophic Forgetting Problem in RL

When a neural network policy is updated on a new task, gradient updates overwrite weights that encoded solutions to previous tasks. This is exacerbated in RL because:

- **Non-stationary targets**: the value function $V^\pi(s)$ changes as policy $\pi$ improves
- **Sparse rewards**: representations develop only around rewarded state-action sequences
- **Distribution shift**: the replay buffer distribution changes as the policy evolves
- **Task-induced representation drift**: new tasks may require completely different feature representations

Formally, catastrophic forgetting occurs when:

$$\mathbb{E}_{s \sim \rho_i}[R_i(\pi_\theta)] \to 0 \quad \text{after training on task } j \neq i$$

## Approaches to Continual RL

### 1. Elastic Weight Consolidation for RL

EWC (Kirkpatrick et al., 2017) penalizes changes to weights important for previous tasks, approximated by the Fisher information matrix:

```python
import torch
import torch.nn as nn


class EWCPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.fisher = {}       # Fisher diagonal per parameter name
        self.star_params = {}  # Optimal params after each task

    def compute_fisher(self, dataloader, device: str = "cuda"):
        """Estimate diagonal Fisher from logged transitions."""
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        self.train()
        for states, actions, log_probs_old in dataloader:
            states = states.to(device)
            logits = self.net(states)
            log_probs = logits.log_softmax(dim=-1)
            selected = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            loss = -selected.mean()
            self.zero_grad()
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= len(dataloader)
        self.fisher = fisher
        self.star_params = {n: p.detach().clone() for n, p in self.named_parameters()}

    def ewc_loss(self, importance: float = 1000.0) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for n, p in self.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()
        return importance * loss / 2
```

EWC adds `ewc_loss()` to the standard RL objective (e.g., PPO policy gradient loss). Its quadratic approximation of the posterior becomes inaccurate for large weight changes across many tasks.

### 2. Progressive Neural Networks

Progressive Networks (Rusu et al., 2016) freeze learned columns and add new ones for each task, with lateral connections from frozen columns:

```python
class ProgressiveColumn(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, prev_columns: list, hidden: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(state_dim, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, action_dim),
        ])
        # Lateral adapters from each previous column, per layer
        self.laterals = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden, hidden) for _ in prev_columns])
            for _ in range(len(self.layers) - 1)
        ])
        self.prev_columns = prev_columns  # frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = [x]  # activations per layer; index 0 = input
        for layer_i, layer in enumerate(self.layers):
            inp = h[-1]
            if layer_i > 0:  # add lateral inputs from previous columns
                for col_i, prev_col in enumerate(self.prev_columns):
                    lateral = self.laterals[layer_i - 1][col_i]
                    inp = inp + lateral(prev_col.h[layer_i - 1])  # skip for brevity
            h_new = torch.relu(layer(inp)) if layer_i < len(self.layers) - 1 else layer(inp)
            h.append(h_new)
        self.h = h[1:]
        return h[-1]
```

Progressive Networks guarantee no forgetting (frozen columns), but capacity grows linearly with the number of tasks — not scalable to hundreds of tasks.

### 3. PackNet: Parameter Packing with Binary Masks

PackNet (Mallya & Lazebnik, 2018) iteratively prunes and re-uses freed weights for new tasks:

```python
class PackNet:
    def __init__(self, model: nn.Module, prune_fraction: float = 0.5):
        self.model = model
        self.prune_fraction = prune_fraction
        self.masks = {}  # task_id -> {param_name -> bool tensor}
        self.task_id = 0

    def prune_and_lock(self):
        """Prune least important weights and lock current task's weights."""
        mask = {}
        for name, param in self.model.named_parameters():
            with torch.no_grad():
                # Keep top (1-prune_fraction) weights by magnitude
                threshold = param.data.abs().float().quantile(self.prune_fraction)
                keep = param.data.abs() >= threshold
                mask[name] = keep
                param.data *= keep.float()
        self.masks[self.task_id] = mask
        self.task_id += 1

    def apply_mask(self, task_id: int):
        """Apply stored mask for inference on a specific task."""
        for name, param in self.model.named_parameters():
            if name in self.masks[task_id]:
                with torch.no_grad():
                    param.data *= self.masks[task_id][name].float()
```

### 4. Episodic Memory / Experience Replay

Storing and replaying transitions from previous tasks prevents policy drift:

```python
from collections import deque
import random


class ContinualReplayBuffer:
    def __init__(self, capacity_per_task: int = 10_000):
        self.task_buffers: dict[int, deque] = {}
        self.capacity_per_task = capacity_per_task

    def add(self, task_id: int, transition: tuple):
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = deque(maxlen=self.capacity_per_task)
        self.task_buffers[task_id].append(transition)

    def sample_mixed(self, batch_size: int, current_task: int) -> list:
        """Sample from current and previous tasks uniformly."""
        all_transitions = []
        for tid, buf in self.task_buffers.items():
            all_transitions.extend(list(buf))
        return random.sample(all_transitions, min(batch_size, len(all_transitions)))
```

Rehearsal-based methods are simple and effective but require storing past data — a concern for privacy-sensitive or memory-constrained deployments.

## Benchmarks for Continual RL

| Benchmark | Tasks | Type | Key Challenge |
|---|---|---|---|
| Continual World | 10 robotic manipulation | Sequential | Diverse kinematics |
| Meta-World CL | 50 manipulation | Sequential/random | Scale |
| NetHack Learning Env | Procedurally generated | Single env, non-stationary | Long-horizon |
| Gym-MiniGrid | Curriculum grid worlds | Sequential | Compositional tasks |
| SMNIST (RL variant) | 10 classification as RL | Sequential | Plasticity test |

## Evaluation Metrics

- **Average Return** (AR): mean cumulative reward across all seen tasks
- **Backward Transfer** (BWT): change in performance on old tasks after learning new ones — negative BWT indicates forgetting
- **Forward Transfer** (FWT): improvement on unseen tasks from prior learning — positive FWT indicates useful generalization

$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})$$

where $R_{j,i}$ is the return on task $i$ after training on task $j$.

## Plasticity-Stability Tradeoff

A recurring tension in CRL:

- **Stability**: preserve performance on old tasks (low BWT)
- **Plasticity**: rapidly adapt to new tasks (high FWT and AR)

Architectural approaches (progressive networks, PackNet) favor stability at the cost of plasticity and parameter efficiency. Regularization approaches (EWC) offer a tunable tradeoff via the penalty coefficient. Replay-based methods tend to offer the best balance in practice.

## Summary

Continual Reinforcement Learning remains an open frontier. While approaches like EWC, progressive networks, PackNet, and experience replay each address catastrophic forgetting from different angles, no single method dominates across all benchmarks. The field is advancing toward agents with growing episodic memory, meta-learned optimizers with built-in continual learning priors, and world models that compress task structure for efficient lifelong learning — capabilities essential for deploying robots and AI assistants that must grow with experience rather than reset with each new deployment.
