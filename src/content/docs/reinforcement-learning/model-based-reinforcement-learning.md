---
title: Model-Based Reinforcement Learning - Planning with Learned Dynamics
description: Understand how model-based RL learns or uses environment dynamics, plans actions, and manages model error.
---

Model-based reinforcement learning uses a model of the environment to predict what will happen after an action. The model can be known in advance, such as a simulator, or learned from data.

```text
state + action -> dynamics model -> next state and reward
```

The agent can test possible futures inside the model before acting in the real environment, making it potentially far more sample-efficient than model-free RL.

## Planning

Model predictive control (MPC) repeatedly plans a short action sequence:

1. generate candidate action sequences
2. predict their trajectories with the dynamics model
3. choose the sequence with highest predicted return
4. execute only its first action
5. observe reality and plan again

Replanning limits the damage from imperfect long-horizon predictions.

## Learning the Model

A learned dynamics model might predict state deltas, rewards, termination probabilities, or a distribution over possible next states. Ensembles of models estimate uncertainty: when members disagree, the agent should be cautious about trusting imagined outcomes.

World-model methods learn compact latent states rather than predicting raw pixels. This enables planning from high-dimensional observations, but the latent model must still preserve information relevant to control.

## The Main Risk: Model Exploitation

An optimizer can discover action sequences that look excellent in an imperfect model but fail in reality. This is sometimes called model bias or model exploitation.

Useful mitigations include:

- short planning horizons and frequent replanning
- uncertainty penalties for unfamiliar states
- training on diverse data
- mixing real experience with imagined rollouts
- validating policies in progressively more realistic environments

## When to Use It

Model-based RL is attractive when real interactions are expensive, such as robotics, process control, and energy systems. It is not automatically safer: a learned model is an approximation. Keep constraints outside the reward function, test under distribution shift, and require a conservative fallback when model uncertainty is high.

