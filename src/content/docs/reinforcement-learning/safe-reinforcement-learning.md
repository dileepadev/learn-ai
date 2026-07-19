---
title: Safe Reinforcement Learning - Optimizing Under Constraints
description: Learn why rewards alone are not enough for safety, and explore constrained policies, shields, and safer evaluation practices.
---

Safe reinforcement learning aims to improve a policy without violating defined safety constraints. Standard RL maximizes expected reward; safe RL recognizes that an agent may find a high-reward path that is unacceptable because it damages equipment, exceeds a limit, or harms people.

## Rewards Are Not Constraints

Adding a large negative reward for unsafe behavior is often insufficient. The agent may still accept the penalty if another reward is large enough, or it may exploit a gap in the reward specification.

A constrained formulation separates the goals:

$$\max_\pi J_R(\pi) \quad \text{subject to} \quad J_C(\pi) \leq d$$

Here $J_R$ is expected reward, $J_C$ is expected cost such as collisions or emissions, and $d$ is a maximum permitted cost.

## Safety Techniques

### Constrained Optimization

Algorithms such as constrained policy optimization maintain a cost estimate alongside reward and update the policy within a safety budget.

### Action Shields

A shield filters or replaces actions that violate known rules:

```text
proposed action -> safety check -> allowed action
```

Shields are valuable when constraints can be specified independently, such as joint limits or forbidden zones.

### Risk-Sensitive Objectives

Expected reward can hide rare catastrophes. CVaR and related objectives focus on poor-tail outcomes, encouraging policies that avoid severe losses even when their average reward is attractive.

## Evaluation

Test violations, near misses, recovery behavior, and worst-case performance—not only average return. Simulators are useful but do not prove real-world safety. Define stop conditions, runtime monitors, human override paths, and an incident process before live deployment.

Safe RL is a useful research and engineering toolkit, not a safety certification. In high-consequence systems, combine it with domain expertise, formal controls, testing, and accountable human oversight.

