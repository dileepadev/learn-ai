---
title: Counterfactual Regret Minimization (CFR)
description: Learn how CFR finds approximate Nash equilibria in imperfect-information games, and how it powered superhuman poker-playing agents.
---

Counterfactual Regret Minimization (CFR) is an algorithm for finding approximate Nash equilibrium strategies in imperfect-information games — games like poker where players don't observe the full game state, unlike perfect-information games such as chess or Go.

## Why Standard RL Struggles Here

Algorithms like [[q-learning-fundamentals]] or [[monte-carlo-tree-search]] work well for single-agent problems or perfect-information games, but in imperfect-information multiplayer games, the "optimal" strategy against a specific opponent isn't well-defined in the same way — the right notion of optimality is instead a Nash equilibrium: a strategy that performs as well as possible assuming the opponent also plays optimally against it, which requires reasoning about strategies over entire information sets (all game states indistinguishable to a player given what they've observed) rather than single fully-known states.

## The Regret Minimization Idea

CFR works by having each player repeatedly play against itself (self-play) and, after each iteration, compute "counterfactual regret" — how much better each alternative action at each decision point would have performed, weighted by the probability of reaching that decision point at all:

```text
for each information set I and action a:
    regret(I, a) += counterfactual_value(I, a) - counterfactual_value(I, actual_action_taken)

strategy(I) <- proportional to positive accumulated regret across actions
```

Over many self-play iterations, the average strategy across all iterations (not the strategy from any single iteration) provably converges to a Nash equilibrium in two-player zero-sum games, a guarantee that comes from regret-minimization theory in game theory rather than from anything specific to reinforcement learning.

## From Poker to Broader Impact

CFR and its variants (particularly Monte Carlo CFR, which samples rather than exhaustively traversing the game tree, making it tractable for games as large as no-limit poker) powered the first AI systems to defeat top human professionals at heads-up no-limit Texas hold'em, a landmark result specifically because poker's imperfect information and enormous strategy space made it resistant to the search-based techniques that had already conquered perfect-information games like chess and Go.

## Practical Guidance

Reach for CFR-family algorithms specifically for imperfect-information, adversarial, multi-agent settings — negotiation, auctions, security games, or any problem genuinely modeled as a zero-sum or general-sum game rather than a single-agent MDP. For cooperative or single-agent problems, standard RL algorithms are a better fit, since CFR's machinery is specifically built around adversarial game-theoretic guarantees that don't apply outside a multi-agent competitive setting.
