---
title: AlphaZero - Self-Play with Neural-Guided Tree Search
description: Learn how AlphaZero combines Monte Carlo tree search with a policy-value network trained entirely by self-play to master board games without human data.
---

AlphaZero is a reinforcement learning algorithm that learned to play chess, shogi, and Go at superhuman level starting only from the rules of each game. It generalizes the earlier AlphaGo approach by removing human game data and game-specific engineering.

## Core Components

1. **A single neural network** `f(s) -> (p, v)` that outputs a move probability vector p (policy) and a scalar value v estimating the game outcome from state s
2. **Monte Carlo Tree Search (MCTS)** guided by that network
3. **Self-play** generating training data

## MCTS with Network Guidance

Each simulation walks down the tree selecting actions that maximize:

```text
a = argmax_a [ Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a)) ]
```

- `Q`: mean value from past simulations
- `P`: network prior for the move
- `N`: visit counts

At a leaf, the network evaluates the position; there are no random rollouts. The value is backed up the path. After many simulations, the visit counts form an improved policy `pi`, which is stronger than the raw network output.

## The Training Loop

```text
repeat:
    play games using MCTS-guided moves, recording (state, pi, game_result)
    train the network:
        policy head  -> match pi (cross-entropy)
        value head   -> predict game result z (mean squared error)
    use the updated network for the next self-play batch
```

The loss combined with regularization:

```text
L = (z - v)^2  -  pi^T log p  +  c * ||theta||^2
```

## Why It Works

MCTS acts as a **policy improvement operator**: search produces better move choices than the network alone, and training distills this into the network, which then makes search stronger. This virtuous cycle drives learning.

## Exploration

During self-play, early moves are sampled in proportion to visit counts (with temperature) and Dirichlet noise is added at the root, ensuring diversity.

## Requirements and Limits

- Needs a perfect simulator of the environment
- Designed for deterministic, perfect-information, two-player zero-sum games
- Very compute-intensive in the original setup

Successors addressed some of these: MuZero removes the need for known rules, and variants extend search to other domains such as program synthesis and algorithm discovery.
