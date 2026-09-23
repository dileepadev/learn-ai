---
title: Markov Decision Processes - The Formal Foundation of RL
description: Understand the states, actions, rewards, and transitions that define a Markov Decision Process and why nearly all of RL builds on this formalism.
---

Almost every reinforcement learning algorithm assumes the environment is a Markov Decision Process (MDP). Understanding the formalism clarifies what RL algorithms are actually solving.

## The Five Components

An MDP is defined by a tuple `(S, A, P, R, γ)`:

```text
S: set of states
A: set of actions
P(s' | s, a): transition probability to next state given state and action
R(s, a, s'): reward received for that transition
γ: discount factor, 0 ≤ γ < 1
```

The Markov property requires that `P(s' | s, a)` depends only on the current state and action, not on the history of how the agent arrived there. This is a modeling assumption, not always literally true, but it makes the problem tractable and is often approximately satisfied by including enough recent history in the state representation.

## Policies and Value

A policy `π(a | s)` maps states to a distribution over actions. The value function `V^π(s)` is the expected discounted return from state `s` under policy `π`; the action-value function `Q^π(s, a)` is the expected return from taking action `a` in state `s` and then following `π`. Solving an MDP means finding a policy that maximizes expected return, and the optimal value function satisfies the Bellman optimality equation:

```text
V*(s) = max_a  sum_s' P(s'|s,a) [ R(s,a,s') + γ V*(s') ]
```

## Why the Discount Factor Matters

`γ` controls how much the agent values future reward relative to immediate reward. A `γ` near 0 makes the agent myopic, chasing immediate reward; a `γ` near 1 makes it weight distant future reward almost as heavily as immediate reward, which can slow convergence and increase variance in return estimates.

## Practical Guidance

When designing an RL environment, resist state representations that violate the Markov property by hiding information the policy needs — if the optimal action genuinely depends on history beyond the current state, either enlarge the state to include that history or use a recurrent policy. Nearly every concept in later RL topics, including [[temporal-difference-learning]], [[q-learning-fundamentals]], and policy gradient methods, is derived from solving or approximating the Bellman equations defined here.
