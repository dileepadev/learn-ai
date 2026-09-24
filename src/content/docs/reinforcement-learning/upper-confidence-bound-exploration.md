---
title: Upper Confidence Bound Exploration
description: Learn how UCB balances exploration and exploitation using confidence bounds on estimated action values, and why it outperforms naive epsilon-greedy.
---

Upper Confidence Bound (UCB) is an exploration strategy that selects actions based on both their estimated value and how uncertain that estimate is, addressing a key weakness of simpler strategies like epsilon-greedy exploration.

## The Problem with Epsilon-Greedy

Epsilon-greedy exploration picks a random action with probability epsilon and the currently best-known action otherwise, but this random exploration treats all non-greedy actions as equally worth trying, regardless of whether an action has been tried many times already (and is therefore well-understood) or barely tried at all (and therefore highly uncertain). This wastes exploration budget re-testing actions that are already confidently known to be mediocre.

## The UCB Formula

UCB instead selects the action that maximizes an upper confidence bound on its estimated value — the estimated value plus a bonus term that grows with uncertainty:

```text
a_t = argmax_a [ Q(a) + c * sqrt( ln(t) / N(a) ) ]
```

`Q(a)` is the current estimated value of action `a`, `N(a)` is how many times it's been selected so far, `t` is the total number of steps taken, and `c` controls the exploration strength. Actions tried rarely (`N(a)` small) get a large bonus, encouraging exploration of undertried actions; as `N(a)` grows, the bonus shrinks, and selection converges toward the action with the highest estimated value.

## Why the Bonus Term Makes Sense

The bonus term comes from a statistical confidence bound (related to Hoeffding's inequality) on how far an estimated average can plausibly be from the true expected value given a certain number of samples — an action's true value is, with high probability, no larger than `Q(a)` plus this bonus, given how many times it's been tried. Selecting the action with the highest such bound means the agent optimistically assumes the best about actions it's less certain of, which naturally directs exploration toward promising, undertried options rather than uniformly random exploration.

## Beyond Multi-Armed Bandits

UCB is most cleanly defined for the multi-armed bandit setting (see [[multi-armed-bandits]] for the broader problem framing), but the same "optimism under uncertainty" principle extends into full RL through algorithms that apply UCB-style bonuses to state-action value estimates, and it directly inspired the exploration mechanism in tree-search algorithms like Monte Carlo Tree Search, where UCB1 governs which branches of the search tree to expand further.

## Practical Guidance

Prefer UCB over epsilon-greedy when action-value estimates naturally come with a meaningful notion of confidence or visit count, since UCB directs exploration much more efficiently toward genuinely uncertain, potentially valuable actions rather than wasting samples on well-understood ones. Tune the exploration constant `c` empirically — too small and UCB behaves almost greedily, too large and it over-explores actions the current data already rules out as poor.
