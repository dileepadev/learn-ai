---
title: "LLM Caching Strategies"
description: "How caching reduces latency and cost in applications built on large language models."
---

Caching is one of the simplest ways to make AI systems faster and cheaper. If users ask the same question repeatedly, or if prompts share long static prefixes, there is no reason to pay the full inference cost every time.

## Common Cache Types

- **Response caching:** return a previous answer for repeated requests.
- **Semantic caching:** match new queries to similar past queries using embeddings.
- **Prompt prefix caching:** reuse computation for long shared prompt prefixes.
- **Retrieval caching:** store expensive search or ranking results.

## Tradeoffs

Caching improves speed and cost, but stale answers can become a problem. Teams need invalidation rules for time-sensitive data, user-specific context, and prompts whose behavior changes across versions.

## Best Fit

Caching works best when traffic contains repetition, retrieval is expensive, or prompts include large static instructions. In production AI systems, good caching can matter as much as model optimization.
