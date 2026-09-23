---
title: SARSA - On-Policy Temporal Difference Control
description: Learn how SARSA differs from Q-learning by updating toward the action actually taken, and why that makes it safer in risky environments.
---

SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference control algorithm, closely related to [[q-learning-fundamentals]] but with one crucial difference in its update target.

## The Update Rule

```text
Q(s, a) <- Q(s, a) + α [ r + γ Q(s', a') - Q(s, a) ]
```

The name comes from the five values needed for each update: the current state and action, the reward, the next state, and the next action actually selected. Unlike Q-learning's `max_a' Q(s', a')`, SARSA uses `Q(s', a')` for the action `a'` the agent's current policy actually chooses next.

## Why This Makes SARSA On-Policy

Because the update target depends on the action the behavior policy actually takes (including exploratory random actions under epsilon-greedy), SARSA learns the value of the policy it is currently following, exploration and all — not the value of the purely optimal policy. This is the definition of on-policy learning: the policy being evaluated and the policy generating data are the same.

## The Cliff-Walking Illustration

The classic example showing the practical difference is a gridworld with a cliff running along one edge, where falling off ends the episode with a large penalty. Q-learning learns the value of the optimal (risky) path hugging the cliff edge, since its target assumes the best possible next action even though the exploring agent will occasionally still fall off. SARSA learns the value of a safer path further from the edge, because its target accounts for the real possibility that the epsilon-greedy policy will sometimes take a random, dangerous action near the cliff.

## When to Prefer SARSA

SARSA is generally preferable when exploration mistakes carry real cost during learning — physical robots, safety-critical control, or any setting where you cannot separate a risk-free training phase from deployment. Q-learning is preferable when you can explore freely in simulation and only care about the value of the final, near-deterministic policy.

## Practical Guidance

Both algorithms converge to reasonable policies under standard conditions, but they can produce meaningfully different behavior mid-training. If an agent trained with Q-learning behaves recklessly during a live exploration phase, consider SARSA or a more conservative exploration schedule rather than assuming the reward function is at fault.
