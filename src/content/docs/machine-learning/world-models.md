---
title: "World Models: AI that Understands Physics"
description: "An overview of World Models in Reinforcement Learning and their ability to simulate environments internally."
---

To interact effectively with the world, an agent needs more than just a list of actions—it needs to understand how the world will react. **World Models** allow an AI to build an internal simulation of its environment.

## The Components of a World Model

- **Vision Model**: Compresses high-dimensional sensory input (like image frames) into a compact latent representation.
- **Memory Model**: Predicts the next latent state based on the current state and a chosen action.
- **Controller**: Selects the best action to take based on the predictions from the memory model.

## Learning in a "Dream"

One of the most powerful aspects of World Models is that the agent can train entirely within its own internal simulation (or "dream"), significantly reducing the need for expensive or dangerous real-world trials.
