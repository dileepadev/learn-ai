---
title: Decision Transformers
description: Explore Decision Transformers — the architecture that reframes reinforcement learning as a sequence modeling problem, enabling offline RL with transformer models and connecting language modeling to decision-making.
---

**Decision Transformers** (Chen et al., 2021) reframe **offline reinforcement learning** as a **sequence modeling problem**, allowing standard transformer architectures to learn policies from logged data — without bootstrapping value functions or computing Bellman updates.

The core insight: if you condition a transformer on desired future returns, it can learn to generate action sequences that achieve those returns from historical trajectories.

## Reinforcement Learning Background

In standard RL, an agent learns a policy $\pi(a \mid s)$ that maximizes cumulative reward:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**Offline RL** (also called batch RL) learns from a fixed dataset of logged trajectories $(s_1, a_1, r_1, s_2, a_2, r_2, \ldots)$ collected by some behavior policy, without interacting with the environment during training. This is valuable when online interaction is expensive, dangerous, or impossible (robotics, healthcare, autonomous driving).

Classic offline RL algorithms (CQL, IQL, TD3+BC) rely on value function estimation, which suffers from distribution shift and training instability when the static dataset lacks coverage of critical states.

## The Sequence Modeling Formulation

Decision Transformer represents a trajectory as a sequence of **return-to-go (RTG)**, state, and action tokens:

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T)$$

Where $\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$ is the **return-to-go** — the sum of future rewards from time $t$.

At inference time, you specify the desired target return $\hat{R}_1$, provide the initial state $s_1$, and the transformer **auto-regressively generates actions** that would achieve that return based on patterns learned from the training data.

$$a_t = \pi_\theta(\hat{R}_t, s_t, a_{t-1}, \ldots)$$

## Architecture

The Decision Transformer uses the **GPT-2 architecture** with causal (masked) self-attention:

1. Each of the three token types (RTG, state, action) is linearly embedded to a shared $d$-dimensional space.
2. A learned positional embedding encodes the timestep $t$ (not the sequence position).
3. Causal self-attention prevents future tokens from influencing past predictions.
4. The model predicts the next action conditioned on all past and current RTG and state tokens.

```
Input:  [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ]
Output: [         a₁,          a₂, ...,          aₜ]
```

Loss is computed only on action predictions (similar to instruction tuning's loss masking):

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \|a_t - \hat{a}_t\|^2$$

For continuous actions (regression), or cross-entropy for discrete actions.

## Return Conditioning: The Key Mechanism

The critical innovation is **conditioning on return-to-go** rather than simply imitating behavior:

- In standard behavioral cloning, the model learns the average behavior in the dataset — including suboptimal actions.
- In Decision Transformer, by specifying a **high target return** at test time, the model preferentially generates action sequences associated with high-return trajectories in the training data.

This allows the model to **stitch together** good partial behaviors from different trajectories — achieving returns higher than any single trajectory if the data contains diverse, complementary experiences.

## Trajectory Transformer

**Trajectory Transformer** (Janner et al., 2021) extends the sequence modeling approach further:

- Discretizes states, actions, and rewards into tokens (treating continuous RL as a language modeling problem).
- Uses beam search over predicted trajectories at inference time to plan.
- Enables model-based-style planning without a separate dynamics model.

$$\hat{\tau} = \arg\max_\tau P_\theta(\tau \mid s_0)$$

The beam search can be conditioned on reward to find high-return trajectories.

## Advantages Over Classic Offline RL

| Property | Classic Offline RL (CQL/IQL) | Decision Transformer |
|---|---|---|
| **Value function needed** | Yes | No |
| **Distribution shift issues** | Severe | Reduced |
| **Training stability** | Variable | High (supervised) |
| **Scales with model size** | Weakly | Yes (transformer scaling) |
| **Long-horizon credit assignment** | Difficult | Natural (attention) |
| **Multi-task learning** | Difficult | Natural (return conditioning) |

## Limitations

**Return coverage dependency**: Decision Transformer cannot improve beyond the best trajectories in the dataset — it can only stitch behaviors that already exist. Classic RL can sometimes discover actions better than any in the dataset through value bootstrapping.

**Suboptimal action generation**: On tasks requiring explicit credit assignment (where a single action early in the episode determines the outcome), Decision Transformers may underperform value-based methods.

**Return specification at test time**: The user must specify a target return. Too high → the model generates unrealistic actions. Too low → the model performs conservatively. Calibrating the right target return requires domain knowledge.

## Extensions and Related Work

### Generalist Decision Transformer (GDT)

Multi-task and multi-domain extensions train a single Decision Transformer on diverse offline datasets (different environments, games, and robotic tasks), learning a single policy that can zero-shot transfer to new tasks via return conditioning.

### Online Decision Transformer

Extends Decision Transformer to the online setting by collecting new trajectories with the current policy and retraining — combining the offline starting point with online fine-tuning.

### Q-Transformer

Replaces the return-to-go conditioning with **Q-value conditioning** and trains with a TD-style objective, combining transformer architecture with value-based offline RL. Used by Google DeepMind for robotic manipulation at scale.

### GATO (Generalist Agent)

DeepMind's **GATO** (Reed et al., 2022) scales the multi-task sequence modeling approach to 604 tasks across text, images, games, and robotics — demonstrating that a single transformer can function as a generalist policy across diverse modalities when trained on diverse trajectory data.

## Connection to Language Models

The Decision Transformer framework reveals a deep connection between **language modeling** and **decision-making**:

- Both model sequences of tokens.
- Both benefit from scale (larger models learn better policies / language).
- Return-to-go conditioning is analogous to prompt conditioning in language models.
- Reward can be thought of as a "human feedback signal" — paralleling RLHF.

This convergence has driven research into using **pre-trained LLMs as planners and policy networks**, leveraging their world knowledge and reasoning capabilities for embodied and agentic AI applications.

## Further Reading

- [Decision Transformer: Reinforcement Learning via Sequence Modeling — Chen et al., 2021](https://arxiv.org/abs/2106.01345)
- [Offline Reinforcement Learning as One Big Sequence Modeling Problem — Janner et al., 2021](https://arxiv.org/abs/2106.02039)
- [A Generalist Agent (GATO) — Reed et al., 2022](https://arxiv.org/abs/2205.06175)
- [Q-Transformer: Scalable Offline RL via Autoregressive Q-Functions — Chebotar et al., 2023](https://arxiv.org/abs/2309.10150)
