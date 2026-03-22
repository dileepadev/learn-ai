---
title: Introduction to Chain-of-Thought
description: A look at how Chain-of-Thought (CoT) prompting enables LLMs to solve complex reasoning problems.
---

Chain-of-thought (CoT) prompting is a method for guiding LLMs to reason step-by-step through complex logic or reasoning tasks.

## Why it Works

By breaking down a problem into a sequence of intermediate steps, LLMs are more likely to:

- Be more accurate with their final answer.
- Provide a clear explanation of their reasoning.
- Be easier to debug when they fail.

## Example

Instead of just asking "What is 15 multiplied by 32?", you might prompt:

"To find 15 multiplied by 32:
First, multiply 15 by 30 (15 x 30 = 450).
Next, multiply 15 by 2 (15 x 2 = 30).
Finally, add the two results (450 + 30 = 480)."
