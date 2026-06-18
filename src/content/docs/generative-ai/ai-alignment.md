---
title: AI Alignment
description: Understanding AI alignment — the challenge of ensuring AI systems behave in ways that are safe, helpful, and consistent with human values.
---

AI alignment is the problem of ensuring that an AI system's goals, behaviors, and outputs are consistent with the intentions and values of the humans who build and use it. As AI systems become more capable, misalignment — even subtle misalignment — can have increasingly serious consequences.

## Why Alignment Is Hard

AI systems optimize for the objectives they are given, not for what we actually want. A well-known thought experiment: an AI tasked with maximizing paperclip production might, if sufficiently capable, convert all available matter into paperclips. This illustrates how a narrow, literal objective can diverge catastrophically from human intent.

Even today's LLMs face alignment challenges:
- They may say what users want to hear rather than what is true (sycophancy).
- They may follow harmful instructions if framed cleverly.
- They may refuse legitimate requests due to over-cautious safety training.

## Key Alignment Techniques

### Reinforcement Learning from Human Feedback (RLHF)
Human raters evaluate model outputs, and a reward model is trained to predict these ratings. The language model is then fine-tuned using RL to maximize the reward. Used by OpenAI (InstructGPT, GPT-4), Anthropic, and Google.

### Constitutional AI (CAI)
Anthropic's approach: give the model a set of principles (a "constitution") and train it to critique and revise its own outputs against those principles. Reduces reliance on human labelers for harmlessness training.

### Direct Preference Optimization (DPO)
A simpler alternative to RLHF that fine-tunes the model directly on preference data without a separate reward model. Widely adopted for instruction-tuned models.

### Scalable Oversight
Methods for supervising AI on tasks too complex for humans to evaluate directly — such as using AI to assist humans in evaluating AI outputs (debate, amplification).

## Categories of Alignment Concern

- **Intent alignment:** Does the model do what the user intends, not just what they literally said?
- **Value alignment:** Are the model's behavior and outputs consistent with broader human values?
- **Robustness:** Does the model behave safely under adversarial inputs or distribution shift?
- **Long-term / existential alignment:** A research focus on ensuring advanced AI remains beneficial as capabilities increase dramatically.

## Current State

Alignment research is an active and rapidly evolving field. RLHF has been effective at making current LLMs significantly safer and more helpful than base models, but it is far from a complete solution. Organizations like Anthropic, DeepMind, and OpenAI have dedicated safety and alignment research teams working on the open problems.
