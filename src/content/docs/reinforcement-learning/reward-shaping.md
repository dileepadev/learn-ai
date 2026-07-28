---
title: Reward Shaping - Guiding Agents with Auxiliary Signals
description: Learn how reward shaping speeds up reinforcement learning without changing the optimal policy, and where it can silently go wrong.
---

Many reinforcement learning environments give sparse rewards—an agent might act for thousands of steps before receiving any signal at all. Reward shaping adds an extra term to the reward function to give the agent denser feedback while learning.

```text
r'(s, a, s') = r(s, a, s') + F(s, a, s')
```

`F` is the shaping function, chosen by the designer to reflect progress toward the goal, such as decreasing distance to a target.

## Potential-Based Shaping

Naively adding shaping rewards can change which policy is optimal—an agent might learn to exploit the shaping signal instead of solving the real task. **Potential-based reward shaping** avoids this by defining the shaping term as the difference of a potential function `Φ` over states:

$$F(s, a, s') = \gamma\,\Phi(s') - \Phi(s)$$

Ng, Harada, and Russell (1999) proved that this form preserves the optimal policy: any policy that is optimal under the shaped reward is also optimal under the original reward, because the shaping terms telescope and cancel out over a full episode. Choosing `Φ` to be a rough estimate of state value (e.g., negative distance to goal) still speeds up learning, since it turns a sparse terminal reward into a dense per-step signal.

## Where It Goes Wrong

- **Reward hacking**: if the shaping function only approximates progress, an agent can find shortcuts that maximize `F` without achieving the true goal—for example, circling near a target without ever reaching it if the potential function does not strictly increase near the actual goal state.
- **Non-potential-based terms**: ad hoc bonuses (like a fixed reward for taking a specific action) generally do change the optimal policy and should be used cautiously.
- **Scale mismatch**: a shaping term that is much larger than the true reward can dominate learning and mask the objective actually being optimized.

## Practical Use

Reward shaping is common in robotics (rewarding decreasing distance to a target), game-playing agents (rewarding score proxies before a match ends), and curriculum-style training where early shaping is gradually annealed out as the agent becomes more capable, letting it transition to optimizing the true, sparse objective directly.
