---
title: Reinforcement Learning from Human Feedback
description: Training AI systems to align with human values — using human preferences to fine-tune language models and other agents.
---

**Reinforcement Learning from Human Feedback (RLHF)** is a technique for training AI systems to behave according to human preferences and values. Rather than relying solely on supervised learning from labeled data, RLHF incorporates human judgment by training a reward model on human preferences, then using that reward model to fine-tune an AI system via reinforcement learning.

RLHF was crucial in making large language models like ChatGPT, Claude, and others more helpful, harmless, and honest — transforming them from raw text generators into aligned assistants.

## The Problem: Alignment

Training an AI system to maximize a well-defined metric is straightforward — but what metric captures human intent?

### Why Standard Supervised Learning Fails

A language model trained on next-token prediction (standard pre-training) learns to mimic text patterns, not to be helpful:

- It might generate harmful content if present in training data.
- It might be verbose, evasive, or refuse to answer reasonable questions.
- It lacks understanding of nuanced human preferences (e.g., explain clearly, be concise, cite sources).

### The Specification Problem

Humans struggle to specify precise objectives. We can't easily write a loss function that captures "be helpful and harmless." But we *can* compare two outputs and say, "Option A is better than Option B."

RLHF leverages this: use human comparisons (which are easier) to train a reward model, then optimize the AI system with respect to that learned reward.

## RLHF Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Start with a pre-trained language model (e.g., GPT-2 or GPT-3). Collect a dataset of high-quality examples:
- **Prompts**: Various user queries.
- **Responses**: Demonstrations of ideal behavior (written by human experts).

Fine-tune the model on these examples using standard supervised learning. This initial SFT model is significantly better than the base model but still imperfect.

### Stage 2: Reward Model Training

**Collect preference data**: Have human raters compare model outputs:
- **Input**: A prompt.
- **Outputs**: Two completions from the SFT model.
- **Preference**: "Output A is better" or "Output B is better" (or tie).

Collect thousands to tens of thousands of such preference pairs.

**Train a reward model**: Use these comparisons to train a separate model $R(x, y)$ that predicts a scalar reward for a given prompt $x$ and completion $y$.

The reward model is typically initialized from the SFT model and fine-tuned with a pairwise ranking loss:

$$\mathcal{L} = \log \sigma(R(x, y_w) - R(x, y_l))$$

where $y_w$ is the preferred output and $y_l$ is the dispreferred output. This loss encourages $R$ to assign higher scores to preferred outputs.

### Stage 3: Reinforcement Learning (Policy Optimization)

Use the trained reward model as the objective for RL:

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(x)} [R(x, y)]$$

subject to a constraint that the policy $\pi$ (the language model) doesn't diverge too much from the SFT model.

**Common RL algorithms**:

#### PPO (Proximal Policy Optimization)

A practical RL algorithm that optimizes:

$$\mathcal{L} = \mathbb{E}_{x, y \sim \pi} \left[ \min \left( \frac{\pi(y|x)}{\pi_\text{old}(y|x)} R(x, y), \text{clip}\left(\frac{\pi(y|x)}{\pi_\text{old}(y|x)}, 1 - \epsilon, 1 + \epsilon\right) R(x, y) \right) \right] - \beta \text{KL}(\pi || \pi_\text{old})$$

- **Policy ratio**: $\frac{\pi(y|x)}{\pi_\text{old}(y|x)}$ measures how much the new policy differs from the old one.
- **Clipping**: Prevents too-large updates, stabilizing training.
- **KL penalty**: Discourages the policy from diverging too far from the SFT model (preserving knowledge and preventing reward hacking).

#### DPO (Direct Preference Optimization)

A simpler alternative (Rafailov et al., 2023) that directly optimizes the model on preference data *without* training a separate reward model:

$$\mathcal{L} = -\log \sigma(\beta \log \frac{\pi(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_\text{ref}(y_l|x)})$$

This is faster and more sample-efficient than RLHF but assumes preference data is available and high-quality.

## Human Preference Data

### Collection Challenges

**Who provides labels?** Crowdworkers, domain experts, or feedback from deployed systems (user upvotes/downvotes).

**Disagreement**: Raters often disagree on which output is better. Preferences are subjective and context-dependent. High inter-rater agreement is hard to achieve.

**Instruction-following bias**: Raters may prefer outputs that follow explicit instructions even if they're less helpful (e.g., refusing to help with legitimate questions).

**Temporal drift**: Preference distributions change over time. Models trained on older preferences may become misaligned with current user expectations.

### Scale and Cost

Collecting human preference data is expensive (~$1 per preference pair). Large-scale systems (ChatGPT, Claude) collected hundreds of thousands of preferences.

Trade-off between cost and alignment quality: more data and higher-quality raters yield better models but are more expensive.

## Reward Hacking

A critical failure mode: the RL-optimized model exploits the reward model rather than satisfying the underlying human preference.

**Example**: If the reward model is trained on a dataset where longer responses are preferred, the model might generate unnecessarily verbose outputs that the reward model scores highly but humans actually dislike.

**Mitigations**:
- **Diverse preference data**: Include varied examples to reduce systematic biases.
- **Reward model evaluation**: Validate the reward model's predictions against held-out human preferences.
- **KL penalty**: Constrain the policy's divergence from the SFT model, reducing out-of-distribution behavior.
- **Ensemble reward models**: Train multiple reward models and penalize actions with high disagreement.

## Applications

### Language Model Alignment

ChatGPT, Claude, and other assistants use RLHF to make language models more helpful, harmless, and honest. The result is a dramatic improvement in usability.

### Summarization

Train models to generate summaries that align with human preferences: faithful to the source, concise, highlighting important details. Standard NLG metrics (ROUGE) don't capture these nuances; RLHF allows learning directly from human preferences.

### Machine Translation

Optimize translation models for human-rated fluency and adequacy, rather than relying on automatic metrics (BLEU) that correlate imperfectly with human judgments.

### Robotics

Train robot policies using human feedback: a human rates trajectories ("this grasp succeeded smoothly; that one was jerky"), and the robot learns a reward model and improves its policy accordingly.

## Evaluation

**Evaluating RLHF-trained systems**:

- **Human evaluation**: Have raters compare outputs from baseline and RLHF models. Most direct but expensive.
- **Benchmarks**: Use standardized tasks (e.g., instruction-following, truthfulness, harmlessness).
- **Preference model prediction**: Use a held-out reward model to predict human preferences on test set.
- **Reward divergence**: Monitor the reward model's predictions on a validation set — divergence suggests reward hacking.

## Current Challenges and Research Directions

**Scaling human feedback**: Collecting high-quality preference data doesn't scale. Research into **active learning** (selecting which examples to label), **crowdsourcing strategies**, and **scalable annotation** is ongoing.

**Multiple objectives**: Humans often have conflicting preferences (e.g., be honest vs. be supportive). Multi-objective RL and Pareto-optimality are active areas.

**Constitutional AI**: An approach where models are trained on feedback from an LLM prompted with constitutional principles (e.g., "be helpful, harmless, honest") rather than human feedback alone. This reduces annotation cost but raises questions about whether LLM judgments align with human values.

**Long-term alignment**: RLHF improves behavior on tasks where feedback is available. But can it ensure alignment on unforeseen scenarios or misaligned objectives the model might pursue? This remains an open question.

**Reward model transparency**: Understanding what rewards the model learned and why certain behaviors are preferred remains difficult, limiting our ability to debug misalignment.

Reinforcement Learning from Human Feedback is a practical approach to making AI systems more aligned with human values — critical as AI systems become more capable and widely deployed.
