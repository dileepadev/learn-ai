---
title: Introduction to Opik
description: A quick introduction to Opik for tracing, evaluating, and improving LLM applications.
---

Opik is an open-source platform for LLM observability and evaluation. It helps teams capture traces, run structured evaluations, and monitor quality over time so model and prompt changes can be made safely.

## What Opik Helps With

LLM systems can fail in subtle ways that are hard to debug from logs alone. Opik helps by making it easier to inspect:

- Prompt/response chains
- Tool calls
- Retrieval context
- Latency and cost behavior

## Key Features

### Tracing

Opik records request-level traces so teams can understand how a final response was produced.

### Evaluation Workflows

You can define evaluation datasets and score outputs with reusable criteria, enabling side-by-side comparison of runs.

### Prompt and Model Iteration

By comparing experiments across versions, teams can identify which changes improve quality and which introduce regressions.

### Monitoring in Production

Opik can be used to track quality-related trends after deployment and alert on degradation patterns.

## Typical Workflow

1. Instrument your app for tracing
2. Collect representative examples
3. Run evaluations for baseline quality
4. Iterate prompts/models
5. Re-run evaluations before release
6. Monitor post-release drift

## Best Practices

- Keep a stable benchmark dataset
- Track both quantitative and human-review metrics
- Include edge cases in evaluations
- Tie evaluation results to release decisions

## Where Opik Fits

Opik works well alongside existing LLM frameworks and model providers. It does not replace your app stack; it strengthens your development and reliability process.

For teams moving from prototype to production, Opik helps turn prompt tuning into a measurable engineering discipline.
