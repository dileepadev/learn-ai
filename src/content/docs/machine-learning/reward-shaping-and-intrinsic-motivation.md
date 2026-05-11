---
title: Reward Shaping and Intrinsic Motivation
description: Explore reward shaping and intrinsic motivation in reinforcement learning — covering potential-based reward shaping, the PBRS consistency theorem, curiosity-driven exploration via ICM and RND, count-based methods, and RIDE for procedurally generated environments.
---

In reinforcement learning, the agent learns entirely from the reward signal. When rewards are sparse — only given at the end of a long episode, or only when a rare goal is achieved — the probability of stumbling upon a reward through random exploration is vanishingly small. A robot learning to open a door might explore for millions of steps without ever randomly achieving the correct lever-turn sequence; a game-playing agent in Montezuma's Revenge earns its first reward only after a precise sequence of 50+ actions.

**Reward shaping** and **intrinsic motivation** are two complementary strategies for making sparse reward problems tractable — either by adding auxiliary reward signals that guide the agent toward the true reward, or by generating internal curiosity-based rewards that encourage systematic exploration.

## Reward Shaping

Reward shaping adds a shaping term $F(s, a, s')$ to the environment reward $r(s, a, s')$:

$$\tilde{r}(s, a, s') = r(s, a, s') + F(s, a, s')$$

The agent is then trained on $\tilde{r}$ rather than $r$. The risk is **reward hacking**: a shaping function that accidentally provides more reward for suboptimal behavior than the true goal will cause the agent to optimize the wrong objective.

### Potential-Based Reward Shaping (PBRS)

**PBRS** (Ng, Harada, Russell, 1999) defines a shaping function as the difference of a real-valued **potential function** $\Phi: \mathcal{S} \to \mathbb{R}$:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

where $\gamma$ is the discount factor. This telescopes over a trajectory of length $T$:

$$\sum_{t=0}^{T-1} F(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$

Since $\gamma^T \Phi(s_T) \to 0$ as $T \to \infty$ for $\gamma < 1$, the total shaping reward is bounded regardless of the trajectory length. The **PBRS consistency theorem** states:

> Any shaping function of the form $F = \gamma\Phi(s') - \Phi(s)$ preserves the optimal policy: $\pi^*_{\tilde{r}} = \pi^*_r$.

Conversely, any shaping function that is **not** of this form may change the optimal policy. PBRS provides a principled way to encode domain knowledge (e.g., "being closer to the goal is better") without risking reward hacking.

### Designing the Potential

Common potential functions:

- **Distance-based:** $\Phi(s) = -d(s, s_\text{goal})$ for continuous navigation tasks. Higher potential when closer to goal.
- **Sub-goal indicator:** $\Phi(s) = \sum_i w_i \cdot \mathbf{1}[\text{sub-goal}_i \text{ achieved in } s]$ for sequential tasks.
- **Value function from a simpler problem:** $\Phi(s) = V^\pi_\text{easy}(s)$ transfers value estimates from a simplified version of the task (e.g., the same task without obstacles).

## Intrinsic Motivation

**Intrinsic motivation** generates reward signals internally — independent of the task-specific environment reward — based on properties of the agent's own learning process or uncertainty. The total reward becomes:

$$\tilde{r}(s, a, s') = r_\text{ext}(s, a, s') + \beta \cdot r_\text{int}(s, a, s')$$

where $\beta$ controls the relative weight of intrinsic reward. Importantly, intrinsic rewards can drive meaningful exploration even when $r_\text{ext} = 0$ throughout training.

### Count-Based Exploration

The earliest intrinsic motivation approaches award a bonus inversely proportional to the visit count of a state:

$$r_\text{int}(s) = \frac{1}{\sqrt{N(s)}}$$

where $N(s)$ is the number of times state $s$ has been visited. This is provably efficient in tabular settings (achieving $\tilde{O}(\sqrt{SAT})$ regret in finite MDPs), but does not scale to large or continuous state spaces where every state is visited at most once.

**Pseudo-count methods** (Bellemare et al., 2016) generalize visit counts to continuous state spaces using a density model $\rho_n(s)$ trained on observed states:

$$\hat{N}(s) = \frac{\rho_n(s)}{1 - \rho_n(s)} \cdot n, \qquad r_\text{int}(s) \propto \hat{N}(s)^{-1/2}$$

The pseudo-count tracks how familiar the density model is with state $s$: familiar states have high $\rho$ and low bonus; novel states have low $\rho$ and high bonus.

### Intrinsic Curiosity Module (ICM)

**ICM** (Pathak et al., 2017) measures curiosity as **prediction error** in a learned latent space. The key insight is that predicting in raw pixel space is confusing (TV noise is highly unpredictable but not interesting), so predictions should be made in a feature space that captures only **controllable** aspects of the environment.

ICM consists of three networks:

1. **Feature encoder** $\phi: s \to z$ encodes state $s$ into a compact representation.
1. **Inverse model** $g: (z_t, z_{t+1}) \to \hat{a}_t$ predicts the action taken from consecutive feature representations. Training the inverse model ensures $\phi$ captures only aspects of the state that the agent can influence.
1. **Forward model** $f: (z_t, a_t) \to \hat{z}_{t+1}$ predicts the next feature from current feature and action.

The intrinsic reward is the forward model prediction error:

$$r_\text{int}(s_t, a_t, s_{t+1}) = \frac{\eta}{2} \|\hat{z}_{t+1} - \phi(s_{t+1})\|^2$$

High prediction error means the outcome was surprising — the agent did something it hadn't done before. Low error means the outcome was predictable and the state has been well explored.

### Random Network Distillation (RND)

**RND** (Burda et al., 2018) replaces the forward model with a simpler construction: a random target network and a predictor trained to match it.

- **Target network** $f: s \to \mathbb{R}^d$ is a randomly initialized, fixed neural network.
- **Predictor network** $\hat{f}: s \to \mathbb{R}^d$ is trained on visited states to predict the target's output.

The intrinsic reward is the prediction error:

$$r_\text{int}(s) = \|\hat{f}(s) - f(s)\|^2$$

For states visited many times, $\hat{f}$ learns to closely match $f$ — low error, low bonus. For novel states, the predictor has not been trained on them — high error, high bonus. RND is simpler than ICM (no inverse model, no controllability constraint) yet highly effective, achieving state-of-the-art results on hard exploration Atari games like Montezuma's Revenge with just 1,200 visits to the first room (vs. millions needed for purely extrinsic RL).

RND also naturally handles the **noisy TV problem**: a random TV screen produces constant high novelty in raw pixel space, but RND's random target network produces a fixed output for random pixels (since the target is deterministic given the input), so the predictor quickly learns to match it.

### RIDE: Rewarding Impact-Driven Exploration

**RIDE** (Raileanu & Rocktäschel, 2020) addresses a key weakness of ICM and RND on procedurally generated environments: the agent may learn to visit novel states that are visually diverse (new random room layouts) but have no bearing on solving the task. RIDE defines intrinsic reward as the **impact** of the agent's action on the environment state:

$$r_\text{int}(s_t, a_t, s_{t+1}) = \frac{\|\phi(s_{t+1}) - \phi(s_t)\|_2}{\sqrt{N_\text{ep}(\phi(s_{t+1}))}}$$

The numerator measures how much the state changed (encouraging actions that have large effects). The denominator down-weights states seen many times within the current episode. RIDE outperforms ICM and RND substantially on MiniGrid, where rooms have diverse layouts but reward requires solving a key-door task.

## Combining Shaping and Intrinsic Motivation

In practice, shaped rewards and intrinsic motivation are often combined:

- Use PBRS to guide the agent toward task sub-goals (known structure of the task).
- Use RND or ICM to encourage exploration in regions where the potential function provides little guidance.

This combination is especially effective in robotics: a manipulation potential function encodes task progress (hand closer to object → higher potential), while curiosity rewards prevent the agent from getting stuck at local optima of the potential.

## Limitations and Pitfalls

- **Reward hacking with non-PBRS shaping:** Any shaping term outside the potential-based form risks changing the optimal policy. Hand-designed shaping functions should be verified against the PBRS consistency condition.
- **Diminishing intrinsic reward:** As the agent explores, intrinsic rewards naturally decrease — but the transition from intrinsic-reward-driven to extrinsic-reward-driven behavior can be unstable.
- **Interference with task reward:** A high $\beta$ coefficient on intrinsic reward can overwhelm sparse extrinsic rewards, causing the agent to optimize pure exploration rather than the task.
- **Intrinsic reward non-stationarity:** Since the intrinsic reward changes as the predictor learns, the optimization landscape is non-stationary — standard RL convergence guarantees do not apply without modification.

## Summary

Reward shaping and intrinsic motivation address the fundamental hard exploration problem in sparse-reward RL. Potential-based reward shaping provides a theoretically safe framework for encoding domain knowledge without distorting the optimal policy. ICM and RND generate curiosity bonuses from prediction error in learned feature spaces, enabling agents to systematically explore environments where extrinsic reward is absent for millions of steps. RIDE improves on both by rewarding state-change impact rather than pure novelty, handling procedurally generated environments where visual diversity is decoupled from task-relevant exploration.
