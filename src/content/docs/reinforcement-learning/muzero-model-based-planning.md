---
title: MuZero - Planning with a Learned Model
description: Learn how MuZero combines tree search with a learned latent dynamics model, achieving AlphaZero-level performance without being given the rules of the environment.
---

MuZero extends AlphaZero by learning its own model of the environment. It plans with tree search, yet it is never told the game rules or the transition function, and it works in domains like Atari that AlphaZero's perfect simulator assumption excluded.

## Key Insight: Value-Equivalent Models

A model does not have to reconstruct observations. It only needs to predict the quantities that planning uses: the policy, the value, and the immediate reward. MuZero learns a latent state that is useful for these predictions and ignores irrelevant details.

## Three Learned Functions

```text
representation:  h(o_1..o_t)       -> s_0           # encode observations into a latent state
dynamics:        g(s_k, a_{k+1})   -> (r_{k+1}, s_{k+1})   # predict reward and next latent state
prediction:      f(s_k)            -> (p_k, v_k)    # policy and value
```

## Planning

MCTS runs entirely in latent space:

1. Encode the current observation into `s_0`
2. During simulation, expand nodes by applying the dynamics function to a latent state and an action
3. Evaluate leaves with the prediction function
4. Back up values and select the action from visit counts

## Training

The model is unrolled for K steps from sampled trajectory positions, and losses compare predictions to targets at every step:

- Policy targets: MCTS visit distributions
- Value targets: n-step bootstrapped returns
- Reward targets: observed rewards

```text
L = sum over k=0..K of [ L_policy(p_k, pi_{t+k}) + L_value(v_k, z_{t+k}) + L_reward(r_k, u_{t+k}) ]
```

Gradients flow back through the dynamics function across the unrolled steps, so the latent state is shaped by what planning needs.

## Results

MuZero matched AlphaZero in Go, chess, and shogi and set state-of-the-art results on Atari at the time of publication.

## Related Developments

- **EfficientZero** improves sample efficiency with self-supervised consistency losses
- **Stochastic MuZero** handles environments with randomness
- **Dreamer** learns world models for imagination-based policy learning without tree search

## Trade-offs

Search over a learned model is expensive and errors in the model can compound over deep simulations, so most gains come from combining learned models with shallow-to-moderate search depth.
