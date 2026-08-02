---
title: Process Reward Models for Better Reasoning
description: How step-by-step verification improves mathematical and logical reasoning in language models.
---

Process Reward Models (PRMs) evaluate each step of a reasoning chain rather than just the final answer, enabling more precise feedback for complex reasoning tasks.

## Outcome vs. Process Supervision

**Outcome Reward Models (ORMs)**
Evaluate the final answer only. Simple to implement but provide sparse feedback. A wrong final answer doesn't indicate which step failed.

**Process Reward Models (PRMs)**
Evaluate each reasoning step. Provide dense feedback showing exactly where reasoning diverged from correctness.

## How PRMs Work

**Training Data**
Human annotators label each step in reasoning chains as:
- Correct step
- Incorrect step
- Neutral/ambiguous

**Model Architecture**
A language model trained to output a correctness score for each step. Often a smaller model fine-tuned on step-level labels.

**Inference Usage**
- Generate multiple candidate reasoning chains
- Score each step with the PRM
- Aggregate step scores to rank complete solutions
- Select highest-scoring chain

## Benefits for Reasoning

**Error Localization**
Precisely identify where reasoning fails, enabling:
- Better error messages for users
- Targeted model improvement
- Debugging of reasoning patterns

**Verification Signal**
Rich training signal for improving reasoning:
- Reinforcement learning from step-level feedback
- Fine-tuning on correct reasoning patterns
- Curriculum learning from easy to hard steps

**Better Selection**
More reliable selection among candidates:
- A solution with one wrong step scores lower than one with all correct steps
- Final answer correctness is noisy signal; step correctness is more reliable

## Mathematical Reasoning

PRMs are particularly effective for mathematical problems:
- Each step is verifiable
- Errors cascade through subsequent steps
- Process matters as much as answer

Results on MATH and GSM8K benchmarks show PRMs significantly outperform ORMs for the same model size.

## Code Verification

PRMs can verify code generation:
- Each line or block gets evaluated
- Test cases provide ground truth
- Execution traces provide step-level signals

## Challenges

**Annotation Cost**
Labeling every step requires more human effort than outcome labels. Solutions:
- Use teacher models to generate labels
- Active learning on uncertain steps
- Automatic verification where possible (math, code)

**Credit Assignment**
When a later step is wrong, was it a reasoning error or carried forward from an earlier mistake?

**Generalization**
PRMs trained on one domain (math) may not transfer to others (legal reasoning).

## OpenAI's Approach

The o1 model uses PRM-style verification during training, learning to generate reasoning that passes step-level checks. This enables the model to "think before answering" more effectively.
