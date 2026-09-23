---
title: Policy Iteration and Value Iteration - Classic Dynamic Programming for RL
description: Learn the two foundational dynamic programming algorithms for solving MDPs exactly when the environment's model is fully known.
---

Policy iteration and value iteration are the two classic dynamic programming algorithms for solving a Markov Decision Process exactly, assuming the environment's transition and reward model is fully known — a strong assumption that most later RL algorithms, including [[q-learning-fundamentals]], are designed specifically to relax.

## Policy Iteration

Policy iteration alternates between two steps until the policy stops changing: policy evaluation, computing the value function for the current policy exactly by solving the Bellman equations for that fixed policy, and policy improvement, updating the policy to be greedy with respect to that value function.

```text
repeat:
    policy evaluation:   solve V^π(s) for all s, given current policy π
    policy improvement:  π(s) <- argmax_a  sum_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
until policy is unchanged
```

Each policy improvement step is guaranteed to produce a policy at least as good as the previous one, and since there are finitely many deterministic policies in a finite MDP, the algorithm is guaranteed to converge to the optimal policy in a finite number of iterations.

## Value Iteration

Value iteration instead directly iterates the Bellman optimality equation as an update rule, without waiting to fully evaluate a fixed policy at each step:

```text
V(s) <- max_a  sum_s' P(s'|s,a) [ R(s,a,s') + γ V(s') ]
```

This combines evaluation and improvement into a single update, converging to the optimal value function `V*` in the limit, from which the optimal policy is recovered by acting greedily with respect to `V*` once it has converged.

## Why This Matters Even Though It's Rarely Used Directly

Both algorithms require full knowledge of the transition probabilities `P(s'|s,a)`, which is unavailable in most real-world RL problems — you don't know the exact probability of every outcome for every action in advance. Despite this, they matter conceptually: temporal difference methods like [[q-learning-fundamentals]] and [[sarsa-algorithm]] can be understood as sample-based, model-free approximations of exactly this same Bellman backup, replacing the exact expectation over all possible next states with a single observed sample.

## Practical Guidance

Use policy iteration and value iteration to build intuition on small, fully known toy MDPs (gridworlds, simple games) before moving to model-free methods, since these algorithms make the underlying Bellman equation and convergence guarantees explicit in a way sample-based methods obscure. In practice, reach for these methods directly only when you genuinely have a full, accurate model of the environment's dynamics, which is rare outside of planning problems with known rules.
