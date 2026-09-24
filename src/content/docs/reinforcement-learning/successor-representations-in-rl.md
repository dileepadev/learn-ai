---
title: Successor Representations in Reinforcement Learning
description: Learn how successor representations decompose value estimation into a state-visitation model and a reward model, enabling fast adaptation to changing rewards.
---

The successor representation is an approach to value estimation that separates "how likely am I to visit each future state" from "how much reward is each state worth," making it possible to adapt quickly when only the reward function changes without needing to relearn state-transition dynamics from scratch.

## Decomposing Value

The standard value function bundles transition dynamics and reward together into a single learned quantity. The successor representation instead explicitly factors value into two separate pieces:

```text
V(s) = Σ_s' M(s, s') * R(s')
```

`M(s, s')`, the successor representation, captures the expected discounted number of times the agent will visit state `s'` starting from state `s` under its current policy — essentially a summary of the environment's transition dynamics under that policy, independent of any particular reward function. `R(s')` is simply the reward associated with each state.

## Why the Separation Matters

If only the reward function changes — a robot's goal location moves, or a foraging agent's food source relocates, while the environment's layout and the agent's movement dynamics stay exactly the same — the successor representation `M` remains valid and only `R` needs to be relearned, which is typically much faster than relearning both dynamics and reward jointly from scratch as a standard value function would require after any reward change. This decomposition mirrors a documented feature of biological spatial and reward learning in neuroscience, which is part of why the successor representation has drawn interest as a plausible model of how animals reuse learned spatial knowledge across changing goals.

## Deep Successor Representations

Extending successor representations to large or continuous state spaces uses learned successor features: instead of a table over raw states, a neural network learns a vector-valued successor feature representation, and value is computed as a linear combination of these features weighted by a learned reward vector, allowing the same general decomposition and fast reward-adaptation benefit to apply in high-dimensional settings like pixel-based control.

## Practical Guidance

Consider successor representation-based methods specifically for problems with a fixed or slowly-changing environment but a reward function (or goal) that changes more frequently than the dynamics do — multi-task or multi-goal settings are the clearest fit, since the shared dynamics only need to be learned once. For single-task problems with a fixed reward throughout training, the added complexity of maintaining a separate successor representation offers little benefit over a standard value function.
