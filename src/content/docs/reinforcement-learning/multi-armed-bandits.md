---
title: Multi-Armed Bandits - Learning While Choosing
description: Learn the exploration-exploitation tradeoff, common bandit algorithms, regret, and when bandits are a better fit than full reinforcement learning.
---

A multi-armed bandit models repeated choices with uncertain rewards. Each “arm” might be a recommendation, notification time, or interface variant. After choosing an arm, the system observes a reward and uses it to make the next choice better.

## The Core Tradeoff

The system must balance:

- **exploration:** try uncertain options to learn their value
- **exploitation:** choose the best option known so far

Always exploiting can lock in an early bad estimate. Always exploring wastes reward. Unlike full reinforcement learning, a basic bandit has no evolving state and each reward is immediate.

## Common Algorithms

### Epsilon-Greedy

Choose the current best arm most of the time; choose a random arm with probability $\epsilon$. It is simple but explores without considering uncertainty.

### Upper Confidence Bound

UCB chooses the arm with the highest optimistic estimate:

$$\hat{\mu}_a + c\sqrt{\frac{\log t}{n_a}}$$

An arm sampled fewer times receives a larger uncertainty bonus.

### Thompson Sampling

Maintain a probability distribution over each arm's reward, sample a plausible reward from each distribution, and choose the largest sample. For binary rewards, Beta distributions provide a convenient model.

## Contextual Bandits

Real decisions often depend on context: user language, device, time, or item features. A contextual bandit chooses an action from a context vector while still observing only the reward for the action taken. This is useful for personalization but makes logging and fairness analysis more important.

## Measuring Performance

**Cumulative regret** compares the reward earned with the reward an oracle would have earned by always choosing the best arm. In production, also track user-facing outcomes, exploration exposure, and outcomes across relevant groups.

## Responsible Use

Define a reward that represents the real objective, not just clicks. Add guardrails that prevent harmful options from being explored, cap exposure to experimental variants, and log the decision context, action probabilities, and outcome. A bandit can adapt quickly, but it will optimize exactly the signal it receives.

