---
title: "Constitutional AI: Training Models with a 'Bill of Rights'"
description: "Understand the methodology behind Constitutional AI, pioneered by Anthropic, to create safer and more helpful AI models."
---

As AI models get more powerful, "guardrails" become more critical. **Constitutional AI** is a method for training models to be safe and helpful by giving them a written set of principles—a "constitution"—to follow.

## The Two Phases of Constitutional AI

1. **Supervised Learning (Critique and Revision)**:
   The model is asked to generate a response. Then, it is asked to **critique its own response** based on the constitution (e.g., "Was this helpful? Was it harmful?") and revise it. This creates a high-quality dataset of aligned responses.

2. **Reinforcement Learning from AI Feedback (RLAIF)**:
   Instead of humans ranking responses (which is slow and expensive), a separate "judge" model uses the constitution to rank two different outputs. The target model then learns from these AI-generated rankings.

## Why use a Constitution?

- **Scalability**: AI feedback is much faster than human feedback.
- **Transparency**: Humans can see exactly what the "Bill of Rights" is (e.g., "Do not be condescending," "Avoid providing medical advice").
- **Consistency**: Unlike human raters who may have different opinions on what is "safe," a constitution provides a single source of truth.

## Impact on the Industry

Constitutional AI has proven that we can create models that are not only smarter but also more predictable and easier to control, paving the way for safer deployment in sensitive industries like healthcare and law.
