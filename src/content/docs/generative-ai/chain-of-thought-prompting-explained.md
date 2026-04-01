---
title: Introduction to Chain of Thought (CoT)
description: How providing reasons improves AI model reasoning.
---

Chain-of-thought (CoT) is a prompting technique used to elicit better performance from large language models (LLMs) on complex logical, mathematical, and reasoning tasks.

## What is Chain-of-Thought Prompting?

At its heart, CoT asks the language model to break down its reasoning into a series of logical steps. This mimics the way a human would solve a problem by showing their work.

### Traditional Prompting vs. CoT

- **Traditional:** A direct question leading to a direct answer.
- **CoT:** A question accompanied by a request to "think step by step."

## Why Does CoT Work?

1. **Step-wise Verification:** The model can double-check its work as it moves through each part of the problem.
2. **Intermediate Representations:** The model can generate intermediate data that helps it arrive at a more accurate conclusion.
3. **Problem Decomposition:** CoT helps the model break down complex problems into smaller, more manageable sub-tasks.

## Common CoT Variations

1. **Few-shot CoT:** Providing several examples of questions and their step-by-step reasoning chains in the prompt.
2. **Zero-shot CoT:** Using a simple instruction like "Let's think step by step" to trigger the reasoning process.
3. **Least-to-Most Prompting:** Breaking down a problem into its fundamental sub-problems and solving them sequentially.

## Practical Use Cases

- **Mathematical Reasoning:** Solving complex multi-step word problems.
- **Logical Puzzles:** Answering riddles or deduction-based questions.
- **Code Generation:** Writing complex code that requires several layers of logic.
