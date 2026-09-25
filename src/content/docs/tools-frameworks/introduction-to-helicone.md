---
title: Introduction to Helicone - Observability for LLM Applications
description: Learn how Helicone captures LLM request logs, cost, latency, and caching with a lightweight proxy or async logging integration.
---

Helicone is an observability platform for LLM applications, capturing every request and response to give visibility into cost, latency, errors, and prompt behavior across an application's LLM traffic.

## Proxy-Based Integration

The simplest integration routes LLM calls through Helicone's proxy by changing the base URL and adding an auth header, requiring no code changes to how requests are constructed:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://oai.helicone.ai/v1",
    default_headers={"Helicone-Auth": f"Bearer {HELICONE_API_KEY}"}
)
```

Every request and response then gets logged automatically, including token counts, latency, and cost, without instrumenting individual call sites in application code.

## What It Surfaces

Helicone dashboards break down cost and latency by model, user, or custom properties you attach to requests (such as a feature name or customer ID), which is essential for answering questions like "which feature is driving our LLM spend" or "which customer's usage pattern is causing latency spikes" in a multi-tenant application. It also supports session tracking to group multi-turn conversations or multi-step agent traces into a single coherent view rather than a flat list of disconnected requests.

## Caching and Rate Limiting

Helicone can cache identical requests at the proxy level, cutting cost and latency for repeated queries without any application-side caching logic, and supports rate limiting per user or API key to prevent runaway usage from a single client or bug from affecting the whole application's budget.

## Practical Guidance

Add LLM observability before scaling traffic, not after a cost or latency incident forces the question — request-level visibility is hard to reconstruct retroactively from provider-side billing dashboards alone, since those rarely break down cost by feature or user. Attach custom properties (feature, user, environment) to requests from day one so cost and latency breakdowns are actionable immediately rather than requiring a later data migration.
