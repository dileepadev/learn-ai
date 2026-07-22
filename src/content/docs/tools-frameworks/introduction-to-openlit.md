---
title: Introduction to OpenLIT
description: An overview of OpenLIT for observing LLM and GenAI workloads with open telemetry standards.
---

OpenLIT is an observability framework focused on LLM and GenAI applications. It helps teams instrument AI workloads to collect traces, metrics, and logs, so they can monitor performance, quality signals, and costs in one place.

## Why OpenLIT Is Useful

Traditional application monitoring often misses AI-specific concerns, such as:

- Token consumption per request
- Prompt and completion latency
- Model/provider-level cost trends
- Tool call success/failure rates
- Retrieval quality impacts on final output

OpenLIT bridges this gap by adding AI-aware telemetry to standard observability workflows.

## Core Ideas

### OpenTelemetry-Aligned Instrumentation

OpenLIT is designed to integrate with existing observability stacks that use OpenTelemetry. This reduces lock-in and lets teams route AI telemetry to systems they already operate.

### LLM-Centric Metrics

OpenLIT surfaces metrics that matter for AI systems:

- Input/output token counts
- End-to-end request latency
- Per-model request volume
- Error and timeout rates
- Estimated cost per workflow or endpoint

### Traceability Across Components

When an AI app includes retrieval, reranking, tools, and orchestration, OpenLIT helps connect events across those steps so bottlenecks and failure points are easier to detect.

## Common Use Cases

- Monitoring production chatbots and assistants
- Tracking cost spikes after prompt or model changes
- Diagnosing latency regressions in RAG pipelines
- Improving reliability of agentic workflows with tool dependencies

## Typical Integration Flow

1. Instrument your LLM application with OpenLIT
2. Export telemetry via OpenTelemetry pipelines
3. Visualize traces and metrics in your monitoring backend
4. Set alerts for cost, latency, and error thresholds
5. Use dashboards to guide optimization work

## Best Practices

- Define service-level objectives for latency and error rates
- Track token budgets by endpoint and user segment
- Add trace attributes for prompt version and model version
- Correlate quality evaluations with runtime telemetry

## Benefits

- **Unified observability:** AI telemetry alongside normal app telemetry
- **Operational control:** Better visibility into performance and spend
- **Faster incident response:** Better context for debugging failures
- **Scalable governance:** Supports standards-based observability practices

## When to Adopt OpenLIT

OpenLIT is most valuable when an AI workload moves beyond experimentation and requires production reliability, clear cost accountability, and structured observability across multiple models or providers.

By adopting OpenLIT early, teams can avoid blind spots and build AI systems that are easier to operate, optimize, and trust.
