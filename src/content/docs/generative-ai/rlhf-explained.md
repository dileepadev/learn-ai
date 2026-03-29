---
title: Reinforcement Learning from Human Feedback (RLHF)
description: Learn how RLHF aligns large language models with human values and preferences through reward modeling and policy optimization.
---

Reinforcement Learning from Human Feedback (RLHF) is the training technique responsible for transforming a raw pre-trained language model into a helpful, harmless, and honest AI assistant. It bridges the gap between a model that predicts text and one that genuinely behaves the way humans prefer.

## Why Pre-trained Models Need Alignment

A language model trained via next-token prediction learns to mimic the statistical patterns of its training corpus. This makes it capable, but not inherently safe or helpful — it might:

- Give factually wrong answers confidently.
- Generate harmful, biased, or toxic content.
- Follow the literal prompt rather than the user's true intent.

RLHF steers the model's behavior toward responses humans actually prefer.

## The Three Stages of RLHF

### Stage 1: Supervised Fine-Tuning (SFT)

The base model is fine-tuned on a curated dataset of high-quality (prompt, ideal response) demonstration pairs written by human annotators. This gives the model a solid behavioral foundation before reinforcement learning begins.

### Stage 2: Reward Model Training

Human raters compare multiple model-generated responses to the same prompt and rank them by quality. These preferences form a dataset of `(prompt, winner, loser)` triplets.

A separate **Reward Model (RM)** — typically a smaller LLM with a scalar output head — is trained on this preference data to predict which response a human would prefer:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]$$

where $r(x, y)$ is the reward score for response $y$ given prompt $x$, $y_w$ is the preferred response, and $y_l$ is the less preferred one.

### Stage 3: RL Fine-Tuning with PPO

The SFT model (now acting as a **policy**) is further fine-tuned using Proximal Policy Optimization (PPO), a stable RL algorithm. The reward model provides a scalar reward signal guiding the policy toward higher-ranked responses.

A critical addition is the **KL divergence penalty** to prevent the policy from drifting too far from the SFT model (reward hacking):

$$r_{\text{total}} = r_{\theta}(x, y) - \beta \cdot \text{KL}[\pi_\theta(y|x) \| \pi_{\text{SFT}}(y|x)]$$

The policy is updated to maximize this penalized reward.

## The RLHF Pipeline at a Glance

```
Pre-trained LLM
      ↓
  SFT on demos
      ↓
  SFT Model ──────────────── Human preference data
                                      ↓
                              Reward Model (RM)
      ↓                               ↓
  PPO Fine-tuning ← reward signal from RM
      ↓
  Aligned Model (e.g., ChatGPT, Claude, Gemini)
```

## Key Challenges

### Reward Hacking

The policy can learn to exploit weaknesses in the reward model — generating responses that receive high scores but are actually low quality or even nonsensical. Mitigated by:

- The KL penalty term.
- Periodically retraining the reward model with fresh preference data.

### Scalability of Human Labeling

Collecting high-quality human preference data is expensive and slow. Constitutional AI (Anthropic) and AI Feedback (RLAIF) address this by using AI models to generate feedback instead of humans.

### Training Instability

PPO on top of large language models is notoriously sensitive to hyperparameters. A poorly tuned run can degrade model quality.

## RLHF vs. DPO

[Direct Preference Optimization (DPO)](./direct-preference-optimization.md) was introduced as a simpler alternative that skips the reward model entirely and optimizes preferences directly. RLHF remains relevant, however, when:

- Online feedback is available (rewards computed dynamically during training).
- The task requires complex reward shaping beyond pairwise preferences.
- Maximum alignment performance justifies the engineering complexity.

| | RLHF | DPO |
|---|---|---|
| Reward Model Required | Yes | No |
| Training Complexity | High (PPO) | Low (classification) |
| Stability | Lower | Higher |
| Flexibility | High | Moderate |

## Real-World Applications

- **ChatGPT (OpenAI):** InstructGPT paper formally introduced RLHF to LLM alignment and underlies ChatGPT's helpful behavior.
- **Claude (Anthropic):** Uses Constitutional AI, an RLHF variant guided by a set of written principles.
- **Gemini (Google DeepMind):** Employs RLHF as a core component of its alignment pipeline.
- **Llama fine-tunes:** Open-source variants like Llama-2-Chat and Mistral-Instruct are fine-tuned using RLHF or DPO.

## Summary

RLHF is the dominant paradigm for aligning large language models with human intent. By combining supervised fine-tuning, a learned reward model, and policy optimization via PPO, it enables the creation of models that are not just capable — but genuinely helpful, safe, and preferred by humans. Its complexity has motivated simpler successors like DPO, but RLHF remains a foundational technique in modern AI development.
