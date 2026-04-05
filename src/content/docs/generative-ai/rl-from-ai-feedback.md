---
title: "Constitutional AI: RLAIF and AI Alignment"
description: "Learning about Reinforcement Learning from AI Feedback (RLAIF) and the use of a 'constitution' to align model behavior."
---

Popularized by Anthropic, **Constitutional AI** is a method for training AI models to be helpful, harmless, and honest without relying solely on intensive human labeling.

## The Two-Stage Process

### 1. Supervised Learning (Critique and Revision)

The model generates responses, critiques them based on a set of principles (the "constitution"), and then revises the response to be more aligned.

### 2. RLAIF (Reinforcement Learning from AI Feedback)

A "preference model" is trained using the AI's own critiques. This preference model then guides the final fine-tuning of the main model through Reinforcement Learning.

## Benefits

- **Scalability**: Reduces the need for thousands of human annotators.
- **Transparency**: The rules governing the model's behavior are explicitly written out in the constitution.
- **Consistency**: AI feedback is often more consistent than varying human judgments over large datasets.
