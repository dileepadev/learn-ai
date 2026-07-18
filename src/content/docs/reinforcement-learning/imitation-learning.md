---
title: Imitation Learning - Learning Policies from Demonstrations
description: Explore behavior cloning, dataset aggregation, distribution shift, and the practical role of expert demonstrations in training agents.
---

Imitation learning trains an agent from demonstrations instead of asking it to discover behavior through trial and error. A demonstration dataset contains state-action pairs from an expert:

```text
(state, expert action) -> supervised policy training
```

It is useful when rewards are difficult to specify or unsafe exploration is unacceptable.

## Behavior Cloning

Behavior cloning treats action selection as supervised learning:

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s,a) \sim D}[\log \pi_\theta(a|s)]$$

For continuous actions, the loss is often mean squared error between the predicted and demonstrated action. It is simple, stable, and a strong baseline for robotics, driving, and interface automation.

## Covariate Shift

Small mistakes can move the learned policy into states absent from the demonstration data. In those unfamiliar states, it makes larger mistakes, causing errors to compound.

DAgger addresses this by repeatedly collecting states visited by the learner, asking an expert for the correct action, and adding them to the training set. It improves coverage but requires a safe way to query the expert.

## Demonstration Quality

More data is not always better. A useful dataset should cover normal operation, recovery from errors, and meaningful variation in environments. Record who performed demonstrations, the conditions, and any policy or interface changes that affect labels.

## Practical Guidance

Start with behavior cloning, evaluate closed-loop behavior rather than only held-out action accuracy, and compare against a simple scripted baseline. Keep an operator override for deployment. Imitation learning reproduces both skill and bias in the demonstrations, so review for unsafe shortcuts and unequal performance before allowing an agent to act autonomously.

