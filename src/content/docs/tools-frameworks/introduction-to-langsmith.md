---
title: Introduction to LangSmith
description: Learn how LangSmith helps you trace, evaluate, and improve LLM applications in production.
---

LangSmith is a developer platform for debugging, evaluating, and monitoring LLM applications. It is designed to answer critical questions like: "Why did this response fail?", "Which prompt version performs best?", and "How is model quality changing over time?"

## What LangSmith Solves

LLM apps are harder to debug than traditional software because output quality depends on many moving parts:

- Prompt templates
- Retrieval context
- Model configuration
- Tool calls
- User input variability

LangSmith gives visibility into this full chain so teams can improve reliability instead of guessing.

## Core Capabilities

### Tracing

LangSmith captures step-by-step traces for each request:

- Prompt inputs
- Retrieved context
- Model responses
- Tool invocations
- Latency and token usage

This makes root-cause analysis much faster when behavior is unexpected.

### Evaluation

You can run offline or online evaluations against datasets of examples and compare runs across:

- Accuracy or task-specific correctness
- Grounding and factuality
- Safety constraints
- Cost and latency

Evaluations help teams make prompt and model changes with measurable evidence.

### Dataset Management

LangSmith supports test datasets that represent real user scenarios. Teams can curate edge cases and regression tests to ensure updates do not break previously working behaviors.

### Monitoring

In production, LangSmith helps track quality drift and operational metrics. If response quality degrades after a model or prompt change, traces and comparison views make rollbacks and fixes more direct.

## Typical Development Loop

1. Build a baseline LLM chain or agent
2. Send traffic and collect traces
3. Identify failure patterns
4. Add representative examples to evaluation datasets
5. Iterate prompts/retrieval/model settings
6. Re-run evaluations before release
7. Monitor quality after deployment

This loop turns subjective prompt tuning into a structured engineering workflow.

## Benefits for Teams

- **Faster debugging:** Trace-level observability
- **Safer iteration:** Eval-backed changes
- **Regression protection:** Reusable datasets
- **Cross-functional visibility:** Shared quality metrics for product, engineering, and AI teams

## Best Practices

- Track prompt and model versions for each release
- Keep eval datasets aligned with real user behavior
- Separate smoke checks from deep quality evaluations
- Define clear go/no-go thresholds for deployment

## When to Use LangSmith

LangSmith is particularly useful once your LLM app has:

- Multiple prompts or chains
- Retrieval and tool use complexity
- User-facing reliability requirements
- A need for repeatable release criteria

For prototypes, lightweight logging may be enough. For production systems, LangSmith helps operationalize quality in a way that scales with team and product complexity.
