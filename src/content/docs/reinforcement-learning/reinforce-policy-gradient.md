---
title: REINFORCE - The Foundational Policy Gradient Algorithm
description: Understand the REINFORCE algorithm, the log-derivative trick behind policy gradients, and why raw REINFORCE has high variance.
---

REINFORCE is the original Monte Carlo policy gradient algorithm: rather than learning a value function and deriving a policy from it, it directly optimizes a parameterized policy `π_θ(a | s)` to maximize expected return.

## The Policy Gradient

The goal is to maximize `J(θ) = E[Return]` over trajectories sampled from the policy. The policy gradient theorem gives a way to estimate the gradient of this objective using only sampled trajectories and the log-probability of the actions taken:

```text
∇θ J(θ) = E[ Σ_t ∇θ log π_θ(a_t | s_t) * G_t ]
```

`G_t` is the return (sum of discounted future rewards) from time step `t` onward. Intuitively: increase the probability of actions that were followed by high return, decrease the probability of actions followed by low return.

## The Log-Derivative Trick

This gradient form comes from the identity `∇θ π_θ(a|s) = π_θ(a|s) ∇θ log π_θ(a|s)`, which lets you rewrite an expectation over a distribution parameterized by `θ` as an expectation you can estimate by sampling — without needing to differentiate through the environment's dynamics, which are typically unknown and non-differentiable.

## The Variance Problem

Raw REINFORCE has notoriously high variance because `G_t` is a single noisy sample of the return, and full episodes must complete before any update can be made. The standard fix is subtracting a baseline — often an estimated value function `V(s_t)` — from `G_t` without introducing bias, since the expected value of the baseline term's contribution to the gradient is zero:

```text
∇θ J(θ) ≈ E[ Σ_t ∇θ log π_θ(a_t | s_t) * (G_t - V(s_t)) ]
```

`G_t - V(s_t)` is an estimate of the advantage: how much better this action was than expected from this state.

## From REINFORCE to Actor-Critic

Replacing the Monte Carlo return `G_t` with a bootstrapped estimate from a learned critic, and updating on partial trajectories rather than waiting for full episodes, is exactly the step that leads to actor-critic methods. REINFORCE is worth learning first because it makes the core policy gradient idea and the role of the baseline explicit, before the added complexity of a critic network.

## Practical Guidance

Use REINFORCE with a baseline for small, episodic problems to build intuition, but expect slow, noisy convergence compared to modern methods like PPO. In production RL and RLHF pipelines, REINFORCE's core gradient estimator is still present conceptually inside more advanced algorithms — recognizing it there makes those algorithms much less mysterious.
