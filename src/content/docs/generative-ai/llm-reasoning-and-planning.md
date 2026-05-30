---
title: "LLM Reasoning and Planning: From CoT to o1-Style Thinking"
description: "Explore how large language models reason through complex problems — from chain-of-thought prompting to test-time compute scaling and process reward models."
---

Reasoning is the ability to work through a problem step by step, check intermediate results, and arrive at a correct answer. For a long time, LLMs were poor at this. Recent advances in training and inference have dramatically changed the picture.

## The Reasoning Gap

Early LLMs would confidently produce wrong answers to multi-step math or logic problems. The issue wasn't knowledge — it was the inability to maintain a coherent chain of intermediate steps. A single forward pass through the network isn't enough to solve problems that require sequential reasoning.

## Chain-of-Thought Prompting

The first major breakthrough was **chain-of-thought (CoT) prompting**: instructing the model to "think step by step" before giving a final answer. This forces the model to externalize its reasoning into the context, where it can be used as a scaffold for subsequent steps.

CoT works because the model can condition on its own intermediate outputs, effectively giving itself more "compute" per problem.

## Self-Consistency

A simple but powerful extension: generate multiple CoT reasoning paths independently, then take a majority vote on the final answer. This reduces variance and improves accuracy on tasks where multiple valid reasoning paths exist.

## Process Reward Models (PRMs)

Rather than only rewarding correct final answers, **Process Reward Models** provide feedback on each reasoning step. This trains models to produce not just correct answers but correct reasoning processes — catching errors mid-chain rather than only at the end.

PRMs are a key component of OpenAI's o1 and o3 training pipelines.

## Test-Time Compute Scaling

A paradigm shift: instead of making the model bigger, give it more compute at inference time. The model generates a long internal "thinking" trace before producing an answer. This trace can include:

- Exploring multiple solution approaches.
- Backtracking when a path seems wrong.
- Verifying intermediate results.
- Reformulating the problem.

Models like **o1, o3, and DeepSeek-R1** implement this. The key insight is that reasoning quality scales with the length of the thinking trace, up to a point.

## Tree of Thoughts

**Tree of Thoughts (ToT)** extends CoT by exploring a tree of reasoning paths rather than a single chain. At each step, multiple continuations are generated and evaluated, and the most promising branches are expanded. This is essentially a search algorithm over the space of reasoning traces.

## Limitations and Failure Modes

- **Overthinking**: Models can generate long reasoning traces that arrive at wrong answers, especially on simple problems.
- **Hallucinated Reasoning**: The reasoning trace can look coherent but contain factual errors that propagate to the final answer.
- **Compute Cost**: Long thinking traces are expensive. Knowing when to think deeply vs. answer quickly is an open problem.

## The Road Ahead

The combination of better PRMs, more efficient search algorithms, and models trained to allocate thinking effort appropriately is pushing LLM reasoning toward human-level performance on formal domains like mathematics, coding, and scientific reasoning.
