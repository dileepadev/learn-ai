---
title: Q-Learning Fundamentals - Off-Policy Value Learning
description: Learn the tabular Q-learning update rule, why it is off-policy, and how it scales up to deep Q-networks.
---

Q-learning learns the optimal action-value function `Q*(s, a)` directly from experience, without needing a model of the environment's transition dynamics.

## The Update Rule

```text
Q(s, a) <- Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]
```

After observing a transition `(s, a, r, s')`, Q-learning updates `Q(s, a)` toward the reward plus the discounted value of the best action available in the next state, `max_a' Q(s', a')`.

## Why Q-Learning Is Off-Policy

The update target uses `max_a' Q(s', a')` — the value of the best possible next action — regardless of which action the agent actually takes next. This means Q-learning can learn the optimal policy's values while behaving according to a different, more exploratory policy (commonly epsilon-greedy: mostly greedy, occasionally random). This separation between the "behavior policy" that generates experience and the "target policy" being learned is what makes an algorithm off-policy, and it lets Q-learning reuse old experience freely, which is why it pairs naturally with experience replay.

## Convergence Guarantees and Their Limits

Tabular Q-learning provably converges to the optimal `Q*` under standard conditions (every state-action pair visited infinitely often, appropriately decaying learning rate). These guarantees do not carry over once `Q` is approximated with a neural network instead of a table — function approximation combined with bootstrapping and off-policy updates can diverge, a known instability sometimes called the "deadly triad."

## From Tabular Q-Learning to DQN

Tabular Q-learning requires a table entry per state-action pair, which is infeasible for continuous or high-dimensional state spaces like raw pixels. Deep Q-Networks replace the table with a neural network approximator and add stabilizing techniques (target networks, experience replay) specifically to counter the instability that naive function approximation introduces.

## Practical Guidance

Use tabular Q-learning to build intuition on small, discrete environments before moving to deep RL — the update rule and off-policy reasoning transfer directly, but the stability tricks needed for neural approximators do not exist in the tabular case because they aren't needed there. Compare against [[sarsa-algorithm]], the on-policy sibling of Q-learning, to see concretely how the choice of update target changes learned behavior in stochastic or risky environments.
