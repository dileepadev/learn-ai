---
title: Monte Carlo Methods in Reinforcement Learning
description: Learn how Monte Carlo methods estimate value functions from complete episodes, covering first-visit and every-visit prediction, exploring starts, and epsilon-soft control.
---

Monte Carlo (MC) methods learn value functions and policies from sampled experience, without needing a model of the environment's dynamics. They learn from complete episodes, using the actual observed returns.

## Prediction: Estimating V(s)

The value of a state is the expected return. MC estimates it by averaging returns observed after visiting the state:

```text
G_t = r_{t+1} + gamma * r_{t+2} + gamma^2 * r_{t+3} + ...
V(s) = average of G_t over visits to s
```

- **First-visit MC**: average returns only from the first time s appears in each episode
- **Every-visit MC**: average returns from every occurrence

Both converge to the true value as visits grow.

```python
from collections import defaultdict

def mc_prediction(policy, env, episodes, gamma=1.0):
    returns = defaultdict(list)
    V = defaultdict(float)
    for _ in range(episodes):
        episode = generate_episode(env, policy)      # [(s, a, r), ...]
        G, seen = 0.0, set()
        for t in reversed(range(len(episode))):
            s, _, r = episode[t]
            G = gamma * G + r
            if s not in [x[0] for x in episode[:t]]:  # first visit
                returns[s].append(G)
                V[s] = sum(returns[s]) / len(returns[s])
    return V
```

## Control: Finding a Good Policy

Estimating action values Q(s, a) allows policy improvement without a model, since the greedy policy is `argmax_a Q(s, a)`.

Exploration is required so all state-action pairs get sampled:

- **Exploring starts**: begin episodes from random state-action pairs (often impractical)
- **Epsilon-soft policies**: choose a random action with probability epsilon, keeping exploration in every state

## On-Policy versus Off-Policy

Off-policy MC learns about a target policy while following a different behavior policy, correcting with **importance sampling** ratios. This is flexible but can have very high variance.

## Properties Compared with TD Learning

| Property | Monte Carlo | Temporal Difference |
|----------|-------------|---------------------|
| Needs complete episodes | Yes | No |
| Bootstrapping | No | Yes |
| Bias | Unbiased | Biased |
| Variance | High | Lower |
| Continuing tasks | Not directly applicable | Applicable |

## When to Use

MC is simple and effective for episodic tasks with short episodes, such as games. Its returns are unbiased but noisy, which motivated TD methods and intermediate approaches like n-step returns and TD(lambda).
