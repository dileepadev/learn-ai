---
title: Inverse Reinforcement Learning - Inferring Goals from Behavior
description: Learn how inverse reinforcement learning recovers reward functions from demonstrations and why multiple goals can explain the same behavior.
---

Inverse reinforcement learning (IRL) works backward from demonstrations. Instead of receiving a reward function and learning a policy, it observes a policy and tries to infer the reward or preference that could explain it.

```text
demonstrations -> inferred reward -> policy optimization
```

This is appealing when experts can show a task but cannot precisely write down every tradeoff they make.

## Why It Is Ambiguous

The same behavior can be optimal for many different rewards. A driver who slows at an intersection may value safety, obey a speed limit, expect a pedestrian, or simply follow traffic. IRL therefore cannot recover “the true human objective” from behavior alone.

Assumptions and additional evidence are essential: environment dynamics, feature choices, comparisons between trajectories, and constraints can narrow the set of plausible rewards.

## Maximum Entropy IRL

Maximum entropy IRL models experts as preferring high-reward trajectories while allowing stochasticity:

$$P(\tau) \propto \exp(R(\tau))$$

The method seeks a reward that makes expert trajectories likely without assuming the expert always acts perfectly. Modern approaches often learn reward features with neural networks or adversarial training.

## Uses and Cautions

IRL can support robot learning, preference modeling, and analysis of observed decisions. It should not be used to infer hidden intentions or values about people from sparse behavior. Validate the learned reward on new scenarios, expose the assumptions, and keep hard constraints separate from inferred preferences. A reward that matches past demonstrations may still produce unsafe behavior when optimized more aggressively.

