---
title: Introduction to Chain-of-Thought
description: Understanding the reasoning abilities of LLMs.
---

Chain-of-Thought (CoT) is a prompt-engineering technique that improves the reasoning capabilities of large language models (LLMs) by prompting them to generate a sequence of intermediate reasoning steps.

## How CoT Works

Instead of asking for a direct answer, you prompt the model to "think step by step" or provide a few examples of problems with their intermediate steps.

**Normal Prompt:**
"What is the sum of all prime numbers between 1 and 10?"

**CoT Prompt:**
"Think step by step to find the sum of all prime numbers between 1 and 10."

1. The prime numbers between 1 and 10 are 2, 3, 5, and 7.
2. The sum is 2 + 3 + 5 + 7 = 17.
3. Therefore, the sum is 17.

## Benefits of CoT

- **Better Accuracy**: LLMs can solve complex mathematical or logical problems more accurately by breaking them down into simpler steps.
- **Explainability**: You can see how the model arrived at the final answer.
- **Few-Shot CoT**: Using examples of reasoning in the prompt can significantly boost the model's performance on similar tasks.

## Variations of CoT

- **Self-Consistency**: Generating multiple reasoning paths and selecting the most frequent answer.
- **Tree-of-Thoughts**: Exploring multiple branches of reasoning to find the best solution.
- **Least-to-Most Prompting**: Breaking a large problem into smaller subproblems and solving them sequentially.
