---
title: Introduction to Portkey - An AI Gateway for Production LLM Traffic
description: Learn how Portkey's AI gateway adds caching, fallbacks, load balancing, and observability as a single control layer in front of multiple LLM providers.
---

Portkey is an AI gateway that sits between an application and its LLM providers, adding reliability, cost control, and observability features as infrastructure rather than application code.

## Gateway-Based Integration

```python
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

client = OpenAI(
    base_url=PORTKEY_GATEWAY_URL,
    api_key="dummy",
    default_headers=createHeaders(api_key="PORTKEY_API_KEY", virtual_key="openai-prod")
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Draft a follow-up email."}]
)
```

Virtual keys let Portkey manage actual provider credentials centrally, so application code and individual developers never need direct access to raw provider API keys, which simplifies key rotation and access control across a team.

## Configurable Routing and Fallbacks

Portkey's config system lets you define routing logic declaratively — retry with exponential backoff, fall back to a secondary provider or model if the primary fails or times out, load-balance traffic across multiple API keys or providers by weighted percentage — all as configuration rather than code embedded in the application:

```json
{
  "strategy": { "mode": "fallback" },
  "targets": [
    { "provider": "openai", "override_params": { "model": "gpt-4o" } },
    { "provider": "anthropic", "override_params": { "model": "claude-sonnet-4-5" } }
  ]
}
```

## Caching and Guardrails

Portkey supports simple and semantic response caching to reduce cost and latency for repeated or highly similar queries, and can enforce guardrails (PII detection, content filtering, schema validation on structured output) at the gateway level, rejecting or modifying requests and responses before they reach the application or the end user.

## Practical Guidance

Introduce an AI gateway like Portkey once an application depends on LLM calls in its critical path and needs production-grade reliability — provider outages, rate limits, and cost spikes are operational realities that a gateway handles centrally rather than requiring every service to reimplement retry and fallback logic independently. For early prototypes with a single provider and low traffic, the added infrastructure is usually unnecessary until reliability requirements increase.
