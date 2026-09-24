---
title: Temporal Difference Learning - Bootstrapping Value Estimates
description: Learn how temporal difference learning updates value estimates from partial experience without waiting for an episode to end.
---

Temporal difference (TD) learning updates a value estimate using the difference between predicted and observed reward at each step, without waiting for the final outcome of an episode. It combines ideas from Monte Carlo methods (learning from actual experience) and dynamic programming (bootstrapping off existing estimates).

## The TD Update

The simplest form, TD(0), updates the value of a state toward a "TD target" made of the immediate reward plus the discounted value estimate of the next state:

```text
V(s) <- V(s) + α [ r + γ V(s') - V(s) ]
```

The bracketed term is the TD error: the difference between the current estimate `V(s)` and the bootstrapped target `r + γ V(s')`. `α` is the learning rate controlling how much each observation shifts the estimate.

## Why Bootstrapping Matters

Monte Carlo methods wait until an episode ends to compute the actual return, then update value estimates toward that full return — this works but requires episodic tasks and has high variance since a single long trajectory determines the whole update. TD learning updates after every single step using a value estimate of the next state as a stand-in for the true remaining return, which introduces bias (the estimate can be wrong) but sharply reduces variance and works for continuing, non-episodic tasks.

## TD(λ) and Eligibility Traces

TD(0) only credits the immediately preceding state for each TD error. TD(λ) generalizes this by propagating the TD error backward to recently visited states, weighted by how recently they were visited — see [[eligibility-traces-in-reinforcement-learning]] for the mechanism. Setting `λ = 0` recovers TD(0); setting `λ = 1` recovers a Monte Carlo-like update.

## Practical Guidance

TD learning is the backbone of value-based methods like [[q-learning-fundamentals]] and [[sarsa-algorithm]], and understanding the bias-variance tradeoff between TD and Monte Carlo helps explain why deep RL algorithms mix multi-step returns (n-step TD) rather than using pure TD(0) or pure Monte Carlo targets. When debugging an RL agent that learns unstably, check whether bootstrapped targets are diverging — a poorly initialized or updated value function can produce a feedback loop where bad estimates reinforce themselves.
