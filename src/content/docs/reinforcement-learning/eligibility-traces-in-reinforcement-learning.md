---
title: Eligibility Traces - Bridging Monte Carlo and TD Learning
description: Learn how eligibility traces let temporal difference errors propagate credit to recently visited states, unifying TD(0) and Monte Carlo methods.
---

Eligibility traces provide a mechanism for assigning credit to states or state-action pairs visited in the recent past, not just the single most recent one, when a temporal difference error occurs.

## The Problem They Solve

Plain TD(0), described in [[temporal-difference-learning]], updates only the value of the immediately preceding state when a TD error occurs. But if a reward arrives several steps after the action that actually caused it, TD(0) needs many repeated passes over similar trajectories before that credit properly propagates backward through the chain of state updates.

## How Traces Work

Each state (or state-action pair) has an eligibility trace, a number tracking how recently and how often it was visited:

```text
e(s) <- γλ e(s)              for all states, each step (decay)
e(s_t) <- e(s_t) + 1         for the just-visited state (bump)

V(s) <- V(s) + α * δ_t * e(s)    for all states, every step
```

`δ_t` is the TD error at the current step, and `λ` (0 to 1) controls the trace decay rate. When a TD error occurs, every state with a nonzero trace gets updated in proportion to its trace value, so recently visited states receive most of the credit and states visited long ago receive very little.

## TD(λ): Unifying Two Extremes

Setting `λ = 0` makes the trace vanish immediately after each step, recovering plain TD(0), which only updates the most recent state. Setting `λ = 1` makes the trace persist for the whole episode, making the algorithm behave like Monte Carlo, updating every visited state fully whenever a reward is observed. Intermediate values of `λ` interpolate between TD(0)'s low variance/high bias and Monte Carlo's high variance/low bias, often outperforming both extremes.

## Practical Guidance

Eligibility traces (and their generalization, n-step returns, used heavily in modern deep RL algorithms like A3C and PPO's generalized advantage estimation) are the standard way to control the bias-variance tradeoff in value estimation. When an RL agent learns correct behavior too slowly despite a well-shaped reward, consider whether increasing `λ` (or the n-step horizon in a deep RL context) would help credit propagate back through delayed rewards faster.
