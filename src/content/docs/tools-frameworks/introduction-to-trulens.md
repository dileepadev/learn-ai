---
title: Introduction to TruLens
description: Learn how TruLens helps evaluate and monitor LLM applications with feedback functions and groundedness checks.
---

TruLens is an open-source evaluation and observability framework for LLM applications. It helps teams measure response quality, track regressions, and improve retrieval-augmented generation (RAG) systems with repeatable feedback signals.

## Why TruLens Matters

Traditional software metrics cannot fully capture LLM quality. Teams need additional checks for:

- Relevance to user intent
- Groundedness in retrieved sources
- Hallucination risk
- Context quality in RAG

TruLens provides a structured way to compute and track these signals.

## Core Capabilities

### App Instrumentation

TruLens can instrument LLM apps to record prompts, context, model outputs, and metadata for analysis.

### Feedback Functions

You define feedback functions to score quality dimensions such as relevance, coherence, and groundedness.

### RAG Evaluation

For RAG workflows, TruLens can evaluate whether answers are supported by retrieved evidence and whether retrieval itself is useful.

### Dashboarding and Tracking

Run histories and scores can be reviewed over time to detect quality drift after prompt/model changes.

## Common Use Cases

- Prompt regression testing
- Retrieval pipeline tuning
- Launch-readiness checks for assistants
- Production quality monitoring

## Best Practices

- Build evaluation datasets from real user queries
- Track both quality and latency/cost together
- Use thresholds for release gating
- Re-evaluate after any prompt, chunking, or model update

## Getting Started

Start with a small set of high-value feedback metrics (for example relevance + groundedness). Once baseline quality is visible, add deeper domain-specific checks.

TruLens is most effective when evaluation becomes part of every release cycle, not a one-time audit step.
