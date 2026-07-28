---
title: Curriculum Learning in Reinforcement Learning - Ordering Tasks for Faster Training
description: Understand how presenting easier tasks before harder ones can make reinforcement learning agents converge faster and generalize better.
---

Curriculum learning trains an agent on a sequence of tasks ordered by difficulty rather than throwing it directly at the full problem. The idea mirrors how humans learn: master simple cases first, then progress to harder ones.

```text
stage 1: short episodes, few obstacles
stage 2: longer episodes, more obstacles
stage 3: full task difficulty
```

## Why It Helps

Many environments have reward landscapes that are nearly flat for a randomly initialized policy—the agent almost never stumbles into success, so gradients carry little useful signal. Starting on an easier variant of the task increases the chance of early success, giving the agent a reward signal to build on. As competence grows, task difficulty increases, keeping the agent in a regime where it succeeds often enough to keep learning but is still challenged enough to improve.

## Automatic Curriculum Generation

Manually designing a curriculum does not scale to complex environments, so several automatic approaches exist:

- **Self-play**: an agent competes against past versions of itself, which naturally produces an opponent of matching skill level as both sides improve together.
- **Goal-conditioned curricula**: a separate mechanism proposes goals of intermediate difficulty—not so easy that the agent has already mastered them, not so hard that it always fails—often measured through success-rate feedback.
- **Population-based approaches** (e.g., POET): environments and agents co-evolve, with new environment variations generated as agents solve existing ones.

## Risks and Pitfalls

- **Catastrophic forgetting**: an agent trained heavily on later, harder stages may lose competence on earlier, easier ones unless earlier tasks are periodically revisited.
- **Curriculum mismatch**: a poorly ordered curriculum can teach shortcuts that fail to transfer, for instance if early tasks share a spurious feature absent from the final task.
- **Automated curricula collapsing**: automatic difficulty selection can get stuck proposing tasks that are either too easy (no learning signal) or too hard (also no learning signal) if the difficulty estimator is inaccurate.

Curriculum learning is especially valuable in robotics and multi-agent settings where the final task is too sparse or too complex to solve from a random initialization, but it requires careful monitoring to ensure the curriculum tracks genuine capability growth rather than overfitting to easier stages.
