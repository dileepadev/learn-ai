---
title: Introduction to OpenRouter - A Unified API Across LLM Providers
description: Learn how OpenRouter provides one API and one bill for many LLM providers, with automatic fallback and provider routing.
---

OpenRouter is a unified API gateway that gives access to models from many different providers (OpenAI, Anthropic, Google, Meta, Mistral, and dozens of open-weight model hosts) through a single consistent API and a single bill.

## Unified Access

Instead of integrating separate SDKs and API keys for each provider, applications send requests to OpenRouter's OpenAI-compatible endpoint and specify the desired model by name:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="OPENROUTER_API_KEY"
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.5",
    messages=[{"role": "user", "content": "Draft a short release note."}]
)
```

Switching models — including switching providers entirely — often requires changing only the model string, which makes A/B testing across providers and models straightforward.

## Automatic Fallback and Routing

OpenRouter supports specifying a prioritized list of models or providers for a single request, so if the primary choice is unavailable, rate-limited, or too slow, the request automatically falls back to the next option. This is valuable for production systems that need resilience against a single provider's outages or capacity limits without building custom fallback logic themselves.

## Cost and Usage Visibility

Because all requests flow through one platform, OpenRouter provides consolidated spend and usage tracking across providers and models, which is otherwise fragmented across each provider's separate billing dashboard when integrating multiple providers directly.

## Practical Guidance

OpenRouter adds a thin latency and cost overhead compared to calling a provider directly, so weigh that against the operational simplicity of unified billing, fallback, and multi-model access — the tradeoff usually favors OpenRouter for prototyping, multi-model applications, and smaller-scale production use, while very high-volume, latency-sensitive, single-provider deployments may prefer direct integration.
