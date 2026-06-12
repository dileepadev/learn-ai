---
title: Prompt Chaining
description: Learn how prompt chaining breaks complex tasks into sequences of smaller LLM calls to improve reliability, accuracy, and control.
---

Prompt chaining is a technique where the output of one LLM call becomes the input (or part of the context) for the next. Rather than trying to accomplish a complex task in a single prompt, you decompose it into a pipeline of smaller, more focused steps. This improves reliability, makes errors easier to detect, and allows for conditional logic and human review between steps.

## Why Prompt Chaining?

A single large prompt asking an LLM to simultaneously research, reason, format, and validate is asking the model to do too much at once. The result is often lower quality, harder to debug, and less controllable.

Prompt chaining addresses this by:
- Breaking complex work into discrete, verifiable steps.
- Allowing inspection and transformation of intermediate outputs.
- Enabling conditional branching based on intermediate results.
- Making it easier to swap or improve individual steps without rewriting the whole pipeline.

## A Simple Example

**Task:** Write a blog post from a rough idea.

**Chain:**
1. **Prompt 1 (Research):** Given the topic, generate 5 key points to cover.
2. **Prompt 2 (Outline):** Given the key points, create a structured outline.
3. **Prompt 3 (Draft):** Given the outline, write the full draft.
4. **Prompt 4 (Edit):** Given the draft, improve clarity, tone, and grammar.

Each step has a focused job, and the output is easy to inspect and validate.

## Common Patterns

### Sequential Chains
The simplest form — each step feeds directly into the next. Used for workflows like summarize → translate → format.

### Parallel Chains
Multiple prompts run simultaneously and their outputs are merged. Useful for gathering diverse perspectives or processing multiple documents in parallel.

### Conditional / Branching Chains
The next step is chosen based on the output of the previous step. Example: classify an input, then route to a specialized prompt for that category.

### Self-Refinement Chains
The model critiques and revises its own output. Example: generate a response → critique the response → revise based on the critique.

### Map-Reduce Chains
Process each chunk of a large document individually (map), then synthesize the results (reduce). Commonly used for summarizing long texts that exceed the context window.

## Prompt Chaining vs. Agents

Both prompt chaining and agents involve multiple LLM calls. The key difference is **control flow**:
- **Prompt chaining:** The control flow is predefined and deterministic. A developer specifies the exact sequence of steps.
- **Agents:** The LLM itself decides what action to take next, using tools and reasoning dynamically.

Prompt chaining is more predictable and auditable; agents are more flexible but harder to control.

## Best Practices

- **Keep each step focused:** One job per prompt.
- **Validate intermediate outputs:** Check that each step produced the expected format before passing it forward.
- **Use structured outputs:** Ask the model to return JSON so downstream steps can parse results reliably.
- **Log everything:** Store intermediate outputs for debugging.
- **Design for failure:** Handle cases where a step returns unexpected content.

## Tools and Frameworks

- **LangChain** and **LlamaIndex** provide chain abstractions and pipeline utilities.
- **LangGraph** adds state management and conditional routing for complex chains.
- **DSPy** lets you optimize chains by automatically tuning prompts based on end-to-end performance.
