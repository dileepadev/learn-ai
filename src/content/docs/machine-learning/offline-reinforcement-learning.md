---
title: Offline Reinforcement Learning
description: Learn how offline reinforcement learning trains agents from fixed datasets without environment interaction, covering Conservative Q-Learning, Implicit Q-Learning, TD3+BC, distributional shift, and the practical challenges of learning from logged data.
---

**Offline reinforcement learning** (also called batch RL or data-driven RL) trains a policy entirely from a fixed, pre-collected dataset of transitions $(s, a, r, s')$ — without any additional interaction with the environment during training. This separates it fundamentally from standard (online) RL, where the agent continuously collects new experience by acting in the environment.

The motivating insight is that large logged datasets already exist in many high-stakes domains — electronic health records, autonomous driving logs, robot manipulation demonstrations, financial trading histories — and deploying an untested policy to collect more data may be unsafe, expensive, or simply impossible. Offline RL promises to extract useful behavior from those logs.

## The Core Challenge: Distributional Shift

In online RL, the Bellman backup:

$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

is only evaluated on transitions the policy will actually encounter, because the policy generates its own data. In offline RL, the dataset $\mathcal{D}$ was collected by some behavior policy $\pi_\beta$, which may be very different from the policy being learned.

When the learned policy $\pi$ selects an action $a^* = \arg\max_a Q(s, a)$ that is **out-of-distribution** (not covered by $\mathcal{D}$), the Q-value $Q(s, a^*)$ is computed by extrapolation — and function approximation errors compound through the Bellman recursion. This leads to **overestimation of Q-values for unseen actions**, causing catastrophically overconfident policies.

Formally, for state-action pairs $(s, a) \notin \text{supp}(\pi_\beta)$, there is no data to correct the Q-estimate. The bootstrapping error grows exponentially with trajectory horizon $H$:

$$\|Q_\pi - Q^*\|_\infty \leq \frac{\gamma}{1-\gamma} \cdot \epsilon_\text{approx} + \frac{2\gamma}{(1-\gamma)^2} \cdot \epsilon_\text{OOD}$$

where $\epsilon_\text{OOD}$ is the approximation error on out-of-distribution transitions.

## Behavior Cloning: The Simplest Baseline

Before specialized offline RL algorithms, **behavior cloning (BC)** directly supervised the policy to imitate the actions in the dataset:

$$\mathcal{L}_\text{BC}(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}}[\log \pi_\theta(a \mid s)]$$

BC makes no use of reward signals and does not reason about long-horizon consequences — it simply memorizes what the behavior policy did. It is a crucial baseline because many offline RL algorithms only marginally outperform BC on suboptimal or narrow datasets.

**DAgger** and its offline variants extend BC with dataset aggregation but still require some environment access.

## Conservative Q-Learning (CQL)

**CQL** (Kumar et al., 2020) is the most widely adopted offline RL method. It adds a regularization term to the standard Bellman objective that penalizes Q-values for OOD actions and rewards Q-values for in-distribution actions:

$$\min_Q \alpha \left(\mathbb{E}_{s \sim \mathcal{D},\, a \sim \mu(a|s)}[Q(s,a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)]\right) + \frac{1}{2}\,\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[\left(Q(s,a) - \mathcal{B}^* Q(s',\cdot)\right)^2\right]$$

where $\mu$ is a distribution over actions that maximizes current Q-values (OOD candidates), and $\mathcal{B}^*$ is the Bellman operator.

The first term **pushes down** Q-values of actions not in the dataset; the second term **pushes up** Q-values of dataset actions (through the standard TD loss). Together they ensure:

$$\hat{Q}_\text{CQL}(s, a) \leq Q^\pi(s, a) \quad \forall (s, a) \in \mathcal{D}$$

This **lower bound** on Q-values provides pessimism-under-uncertainty: the policy is never tricked into selecting an unseen action because it appears to have a high Q-value.

## TD3+BC: Behavioral Regularization

**TD3+BC** (Fujimoto & Gu, 2021) takes a simpler approach — add a behavior cloning term directly to the TD3 policy update:

$$\pi = \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[\lambda Q(s, \pi(s)) - (\pi(s) - a)^2\right]$$

The $\lambda Q(s, \pi(s))$ term is the standard policy improvement objective; the $-(\pi(s) - a)^2$ term penalizes deviating from dataset actions. The weight $\lambda$ normalizes the Q-values:

$$\lambda = \frac{1}{\frac{1}{N}\sum_{(s_i, a_i)}|Q(s_i, a_i)|}$$

Despite its simplicity, TD3+BC achieves competitive or superior results to far more complex methods on the D4RL benchmark, demonstrating that behavioral regularization is a powerful and reliable strategy.

## Implicit Q-Learning (IQL)

**IQL** (Kostrikov et al., 2021) avoids ever evaluating Q-values on OOD actions entirely, by reformulating the Bellman target using **expectile regression**:

$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[L_\tau^\text{exp}\!\left(Q_{\hat\theta}(s,a) - V_\psi(s)\right)\right]$$

where the asymmetric expectile loss is:

$$L_\tau^\text{exp}(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2$$

With $\tau > 0.5$ (e.g., $\tau = 0.7$), this approximates the maximum Q-value over in-dataset actions — without ever querying the policy network for OOD actions. The Q-function is then updated with a standard in-sample Bellman backup:

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[\left(r + \gamma V_\psi(s') - Q_\theta(s, a)\right)^2\right]$$

Finally, the policy is extracted by advantage-weighted regression:

$$\mathcal{L}_\pi(\phi) = -\mathbb{E}_{(s,a) \sim \mathcal{D}}\left[\exp\!\left(\beta (Q_{\hat\theta}(s,a) - V_\psi(s))\right) \log \pi_\phi(a \mid s)\right]$$

IQL is notable for its stability and applicability to fine-tuning: the pretrained value functions can be used to initialize online RL without catastrophic overestimation.

## Decision Transformer as Offline RL

**Decision Transformer** (Chen et al., 2021) reframes offline RL as **conditional sequence modeling**. A GPT-style transformer is trained to predict the next action given a trajectory of (return-to-go, state, action) triples:

$$(R_1, s_1, a_1, R_2, s_2, a_2, \ldots, R_T, s_T, a_T)$$

where $R_t = \sum_{t'=t}^T r_{t'}$ is the return-to-go at timestep $t$. At inference, the desired return is specified as a prompt — the model generates actions to achieve that target.

Decision Transformer avoids the distributional shift problem entirely by not using Bellman backups at all — it is purely supervised on in-dataset trajectories. This makes it stable and easy to train, but also means it cannot improve over the best trajectory in the dataset (it can only imitate, not "stitch" good sub-trajectories together).

## Dataset Quality and Stitching

A key capability that distinguishes offline RL from imitation is **trajectory stitching**: combining sub-optimal trajectory fragments to construct a better policy than any single trajectory in the dataset.

For example, if the dataset contains:

- Trajectory A: good first half, bad second half
- Trajectory B: bad first half, good second half

An offline RL algorithm using value function bootstrapping can learn to follow A's actions in the first half and B's actions in the second half — achieving higher return than either. Behavior cloning and Decision Transformer cannot do this because they condition on entire trajectories.

CQL and IQL can stitch trajectories because their value estimates propagate reward signals backward across trajectory boundaries through the Bellman operator.

## The D4RL Benchmark

**D4RL** (Fu et al., 2020) is the standard benchmark for offline RL, providing four dataset types for each environment:

| Dataset Type | Description | Expected Difficulty |
| --- | --- | --- |
| `random` | Uniformly random actions | Very hard — low return, sparse signal |
| `medium` | Suboptimal policy (SAC at 1/3 performance) | Medium — some signal but not optimal |
| `medium-replay` | Replay buffer of medium training | Medium — diverse but suboptimal |
| `expert` | Near-optimal policy | Easy — imitating an expert |
| `medium-expert` | Mix of medium + expert trajectories | Hard — requires stitching |

Environments include locomotion (HalfCheetah, Hopper, Walker), AntMaze (long-horizon navigation), Adroit (dexterous manipulation), and FrankaKitchen (multi-task robot).

## Offline-to-Online Fine-Tuning

A practical workflow combines offline pretraining with online fine-tuning:

1. Pretrain Q-function and policy offline using CQL or IQL on a logged dataset.
1. Initialize online RL (SAC, TD3) with pretrained weights.
1. Fine-tune with live environment interaction, using the offline data as a replay buffer.

The pretrained value function provides a strong initialization that prevents the policy from exploring dangerously, while online experience allows recovery from the limitations of the fixed dataset. IQL is particularly well-suited to this workflow because its value estimates are never optimistic about OOD actions.

## Limitations

- **Dataset coverage is a hard ceiling:** If the dataset never visits a region of the state space, offline RL cannot extrapolate reliably into it.
- **Reward quality matters:** Offline RL from data with noisy or sparse rewards is significantly harder than learning from dense, accurate reward signals.
- **Evaluation requires environment access:** The learned policy must still be evaluated in the environment; only training is offline.
- **Stitching requires good value estimation:** On long-horizon tasks (e.g., AntMaze), CQL and IQL still struggle to stitch trajectories over many steps.

## Summary

Offline reinforcement learning addresses the high cost and risk of online data collection by learning entirely from fixed logged datasets. The central challenge — distributional shift and Q-value overestimation for out-of-distribution actions — is handled through pessimism: CQL regularizes Q-values downward for OOD actions, TD3+BC penalizes policy deviation from dataset actions, and IQL avoids OOD queries entirely via expectile regression. These methods enable trajectory stitching that pure imitation cannot achieve, opening up practical deployment of RL in medicine, robotics, and other domains where online exploration is unsafe.
