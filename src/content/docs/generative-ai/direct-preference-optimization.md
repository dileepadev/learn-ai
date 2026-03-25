---
title: Introduction to Direct Preference Optimization (DPO)
description: Learn about DPO, a simpler and more stable alternative to RLHF for aligning LLMs with human preferences.
---

Direct Preference Optimization (DPO) is a technique for fine-tuning large language models to align with human preferences without the complexity of traditional Reinforcement Learning from Human Feedback (RLHF).

## Why DPO?

Traditional RLHF requires training a separate reward model and using complex reinforcement learning algorithms like PPO, which can be unstable and computationally expensive. DPO simplifies this by treating the alignment problem as a simple classification task.

## How DPO Works

1. **Dataset:** Uses a dataset of pairs (prompt, winning response, losing response).
2. **Optimization:** Directly optimizes the policy model (the LLM) to increase the likelihood of the winning response relative to the losing one.
3. **No Reward Model:** Eliminates the need for a separate reward model or RL loop.

## Benefits

- **Stability:** Much easier to train than RLHF/PPO.
- **Efficiency:** Requires less computational overhead.
- **Performance:** Often achieves comparable or superior results in aligning models to human values.
