---
title: Partially Observable MDPs (POMDPs)
description: Learn how POMDPs extend MDPs to settings where the agent cannot directly observe the true state, and the belief-state approach to solving them.
---

A Partially Observable Markov Decision Process (POMDP) extends the standard MDP formalism described in [[markov-decision-processes]] to settings where the agent cannot directly observe the true underlying state, only an observation that gives partial, possibly noisy information about it.

## Why Standard MDPs Aren't Enough

Many real problems violate the assumption that the agent knows the exact current state: a robot with noisy sensors doesn't know its exact position, a poker-playing agent doesn't know its opponents' hidden cards, and a customer service agent doesn't directly observe a user's true underlying intent, only their typed message. Treating these as standard MDPs by pretending the observation is the true state can produce badly suboptimal policies, since the same observation might correspond to genuinely different underlying states requiring different optimal actions.

## The POMDP Formalism

A POMDP adds an observation space `O` and an observation function `Z(o | s, a)` to the standard MDP tuple, describing the probability of receiving observation `o` after taking action `a` and landing in state `s`:

```text
POMDP: (S, A, P, R, Ω, Z, γ)
Ω: set of possible observations
Z(o | s', a): probability of observing o given the resulting state s' and action a
```

The agent never sees `s` directly, only a stream of observations, and must act based on its history of observations and actions rather than the true state.

## Belief States

The standard approach to solving POMDPs maintains a belief state: a probability distribution over which underlying state the agent is actually in, updated via Bayesian filtering each time a new observation and action occur. A POMDP can then be reformulated as a standard MDP over this continuous belief-state space — solving the "belief MDP" gives the optimal POMDP policy, though the belief space is continuous and typically much harder to solve exactly than the original discrete-state MDP.

```text
belief_t(s) = P(state = s | history of observations and actions up to time t)
belief update: incorporate new observation via Bayes' rule after each action
```

## Practical Approaches

Exact POMDP solutions are computationally intractable beyond small problems, so practical approaches include approximating the belief state with a simpler summary (a fixed-size sufficient statistic, or a recurrent neural network's hidden state trained end-to-end to summarize observation history), or simply conditioning a policy directly on a window of recent observations rather than maintaining a full Bayesian belief. Recurrent or transformer-based policies trained with standard deep RL algorithms have become the dominant practical approach to partial observability, implicitly learning to track whatever history-dependent information the task requires.

## Practical Guidance

Recognize partial observability explicitly when designing an RL environment — if two different underlying situations can produce the identical observation but require different optimal actions, a memoryless (feedforward) policy conditioned only on the current observation cannot solve the task correctly regardless of how much it's trained. In that case, use a recurrent or otherwise history-aware policy architecture rather than assuming more training will fix what is fundamentally an architectural limitation.
