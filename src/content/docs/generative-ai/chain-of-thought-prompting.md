---
title: Introduction to Chain-of-Thought Prompting
description: An overview of CoT prompting and how it improves the reasoning capabilities of LLMs.
---

Chain-of-Thought (CoT) prompting is a prompt engineering technique that encourages large language models (LLMs) to reason through a problem step-by-step.

## How CoT Works

Instead of asking for a direct answer, CoT prompting guides the model to break down complex tasks into manageable sub-steps.

1. **Few-Shot CoT:** Provide 2-3 examples of a problem and its step-by-step solution.
2. **Zero-Shot CoT:** Append the phrase "Let's think step by step" to the prompt, triggering the model's reasoning process.

## Why Use CoT?

- **Complex Tasks:** Essential for math word problems, symbolic reasoning, and multi-step logic.
- **Explainability:** Provides insight into how the model reached its final answer.
- **Accuracy:** Reduces "lazy" or incorrect direct answers by forcing the model to articulate its logic.

## Best Practices

- **Use with Model Size:** CoT is most effective in models with 10B+ parameters.
- **Model-Specific Prompts:** Some models respond better to specific phrasing (e.g., "Think through this carefully").
