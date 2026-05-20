---
title: GFlowNets — Generative Flow Networks
description: Explore GFlowNets — a class of generative models trained to sample objects proportionally to a reward function — covering flow matching, trajectory balance, detailed balance, applications to molecular design and combinatorial optimization, and connections to MCMC and variational inference.
---

Generative Flow Networks (GFlowNets) are a class of generative models designed to sample from a distribution proportional to a given reward function over compositional objects. Unlike maximum likelihood training (which learns to match a data distribution) or RL (which optimizes for a single high-reward outcome), GFlowNets learn to **explore** the space of objects, generating *diverse* high-reward samples rather than converging to a mode. This makes them particularly suited to scientific discovery tasks where finding many different valid solutions is more useful than finding one optimal solution repeatedly.

## The Core Problem

Consider generating molecules, protein sequences, or program structures. A reward function $R(x) \geq 0$ assigns quality scores. The goal is to learn a policy that samples objects in proportion to their reward:

$$p(x) \propto R(x)$$

This is fundamentally different from reinforcement learning, which seeks $\arg\max_x R(x)$. GFlowNets pursue **diversity**: if two molecules are equally rewarding, both should appear with equal frequency in generated samples.

## Compositional Generation as Flow

GFlowNets model generation as a sequential process through a directed acyclic graph (DAG):

- **States** $s \in \mathcal{S}$: partial objects (e.g., an incomplete molecule)
- **Actions** $a \in \mathcal{A}(s)$: extensions of the current state (e.g., adding an atom or bond)
- **Terminal states** $x \in \mathcal{X} \subseteq \mathcal{S}$: complete objects with reward $R(x)$
- **Source state** $s_0$: the empty object

A **trajectory** $\tau = (s_0 \to s_1 \to \cdots \to s_n = x)$ is a sequence of state-action pairs that constructs object $x$. Multiple trajectories may construct the same terminal object.

The GFlowNet assigns a **flow** $F(s)$ to each state — analogous to fluid flow — such that flow is conserved: the total inflow to a terminal state equals its reward $R(x)$, and the total outflow from the source equals the partition function $Z = \sum_x R(x)$.

## Training Objectives

### Flow Matching (FM)

The original objective enforces flow conservation at every non-terminal state:

$$\sum_{s' : s' \to s} F(s' \to s) = \sum_{s'' : s \to s''} F(s \to s'')$$

At terminal states: $F(s_{\text{terminal}}) = R(s_{\text{terminal}})$.

In practice, the model parameterizes edge flows $F_\theta(s \to s')$ and minimizes a flow mismatch loss:

$$\mathcal{L}_{\text{FM}} = \sum_{s \in \mathcal{S}} \left(\log \frac{\text{inflow}(s)}{\text{outflow}(s)}\right)^2$$

Flow matching is exact but requires summing over all parents of each state, which can be expensive in large DAGs.

### Trajectory Balance (TB)

**Trajectory Balance** (Malkin et al., 2022) is the most widely used training objective. It trains a forward policy $P_F(s_{t+1}|s_t)$, backward policy $P_B(s_t|s_{t+1})$, and scalar $Z_\theta \approx \log Z$:

$$\mathcal{L}_{\text{TB}}(\tau) = \left(\log Z_\theta + \sum_{t=0}^{n-1} \log P_F(s_{t+1}|s_t) - \log R(x) - \sum_{t=0}^{n-1} \log P_B(s_t|s_{t+1})\right)^2$$

At optimality, for every complete trajectory $\tau$ ending at $x$:

$$Z \cdot \prod_{t} P_F(s_{t+1}|s_t) = R(x) \cdot \prod_{t} P_B(s_t|s_{t+1})$$

The loss is computed **per trajectory** — no sum over all parents needed. This makes TB scalable and the default choice.

### Detailed Balance (DB)

Detailed Balance enforces consistency at every edge rather than every trajectory:

$$F_\theta(s) \cdot P_F(s'|s) = F_\theta(s') \cdot P_B(s|s')$$

equivalently:

$$\mathcal{L}_{\text{DB}}(s \to s') = \left(\log F_\theta(s) + \log P_F(s'|s) - \log F_\theta(s') - \log P_B(s|s')\right)^2$$

DB provides a tighter training signal than TB (more constraints, shorter credit assignment paths) but requires estimating state flows $F_\theta(s)$ for non-terminal states.

### Subtrajectory Balance (SubTB)

SubTB generalizes between TB (whole trajectory) and DB (single edge), applying the balance condition to arbitrary sub-sequences of a trajectory. In practice, $\lambda$-weighted SubTB often outperforms both TB and DB:

$$\mathcal{L}_{\text{SubTB}}(\tau) = \sum_{\text{subtrajectory } \tau'} \lambda^{|\tau'|} \mathcal{L}_{\text{TB}}(\tau')$$

## Implementation

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class GFlowNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.forward_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.backward_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_Z = nn.Parameter(torch.tensor(0.0))

    def forward(self, state, mask=None):
        logits = self.forward_policy(state)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=logits)

def trajectory_balance_loss(model, trajectories, rewards):
    """Compute TB loss over a batch of trajectories."""
    total_loss = 0.0
    for traj, reward in zip(trajectories, rewards):
        log_pf = sum(
            model.forward(s).log_prob(a)
            for s, a in traj[:-1]
        )
        log_pb = sum(
            Categorical(logits=model.backward_policy(s)).log_prob(prev_a)
            for s, prev_a in traj[1:]
        )
        loss = (model.log_Z + log_pf - reward.log() - log_pb) ** 2
        total_loss += loss
    return total_loss / len(trajectories)
```

## Exploration Strategies

GFlowNets need diverse training trajectories to learn the full reward landscape. Several exploration strategies are used:

- **$\epsilon$-greedy**: with probability $\epsilon$, sample uniformly over valid actions
- **Tempered sampling**: sample with temperature $T > 1$ to encourage exploration of low-probability paths
- **Replay buffer**: store high-reward trajectories and replay them to stabilize training on sparse rewards
- **Back-and-forth sampling**: mix forward and backward trajectories during training

```python
def sample_trajectory(model, env, temperature=1.0, epsilon=0.05):
    state = env.reset()
    trajectory = []
    while not env.is_terminal(state):
        dist = model(state)
        if torch.rand(1) < epsilon:
            action = torch.randint(env.num_actions(state), (1,)).item()
        else:
            # Tempered sampling
            logits = dist.logits / temperature
            action = Categorical(logits=logits).sample().item()
        trajectory.append((state, action))
        state = env.step(state, action)
    return trajectory, env.reward(state)
```

## Connections to MCMC and Variational Inference

GFlowNets occupy a middle ground between MCMC and variational methods:

| Property | MCMC | Variational Inference | GFlowNet |
| --- | --- | --- | --- |
| Target distribution | Exact (asymptotic) | Approximate | Approximate (proportional to $R$) |
| Sample diversity | High (in limit) | Low (mode-seeking) | High by design |
| Training signal | None — runs are samples | ELBO gradient | TB/DB loss |
| Speed | Slow mixing | Fast after training | Fast after training |
| Compositionality | General | Structured | Native |

Formally, GFlowNets are equivalent to **amortized variational inference** when the reward is $R(x) = p(x|\mathcal{D})$ (the posterior). The forward policy acts as the approximate posterior $q(x)$, and trajectory balance minimizes a form of KL divergence between $q$ and the true posterior.

## Applications

### Molecular Design

GFlowNets generate molecular graphs fragment-by-fragment. Actions add atoms, bonds, or substructures. The reward combines:

- Binding affinity prediction (docking scores, ML surrogate models)
- Drug-likeness (QED score)
- Synthesizability (SA score)

Studies show GFlowNets discover **orders of magnitude more** diverse high-reward molecules than RL baselines, which collapse to repeating slight variations of discovered hits.

### Biological Sequence Design

GFlowNets generate DNA, RNA, or protein sequences token-by-token with rewards from:

- Wet-lab oracle functions (after training on initial screens)
- Protein structure predictors (ESMFold, AlphaFold2)
- Property prediction networks trained on sequence databases

### Combinatorial Optimization

For discrete optimization problems (graph coloring, scheduling, routing), GFlowNets provide diverse candidate solutions in a single trained model — useful when multiple near-optimal solutions are needed (e.g., generating candidate plans for downstream filtering).

### Scientific Hypothesis Generation

Causal structure learning can be framed as a GFlowNet problem: generate DAGs proportionally to the posterior probability under a structural equation model. GFlowNets discover diverse causal structures consistent with data rather than a single MAP estimate.

## Multi-Objective GFlowNets

Real scientific design problems involve multiple competing objectives. **Pareto GFlowNets** (Jain et al., 2023) learn to sample from the Pareto front:

$$R(x) = \text{hypervolume contribution of } (f_1(x), \ldots, f_k(x))$$

or use scalarization with a distribution over weight vectors to cover the full front. This generates a diverse set of solutions representing different trade-offs between objectives — critical for drug design where potency must be balanced against toxicity and synthesizability.

## Conditional GFlowNets

A conditional GFlowNet takes a conditioning variable $c$ as input alongside the state:

$$P_F(s_{t+1} | s_t, c), \quad R(x, c)$$

This allows a single model to generate different distributions for different conditions (e.g., different target proteins in drug design, different constraint sets in optimization). The model amortizes the learning across conditions, generalizing to new values of $c$ at inference without retraining.

## Summary

GFlowNets provide a principled framework for amortized diversity-seeking generation:

- **Flow matching** enforces conservation of "reward fluid" through the generation DAG
- **Trajectory balance** provides a scalable per-trajectory training objective without summing over DAG parents
- **Detailed balance** gives tighter local consistency constraints for faster convergence
- Applications in molecular design, sequence optimization, and causal discovery demonstrate dramatic diversity improvements over RL baselines

Where RL finds one good solution efficiently, GFlowNets find many good solutions — making them the method of choice for scientific discovery tasks where diversity, not just optimality, is the goal.
