---
title: Introduction to the Mistral AI API
description: Learn how Mistral's API offers efficient open-weight and proprietary models, function calling, and a strong focus on European data residency.
---

Mistral AI provides both a hosted API and openly licensed model weights, positioning itself around efficient, strong-performing models relative to their parameter count and, notably, EU-based infrastructure and data governance.

## Basic Usage

```python
from mistralai import Mistral

client = Mistral(api_key="MISTRAL_API_KEY")

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Draft a project status update."}]
)
```

The API is intentionally OpenAI-compatible in structure, which keeps migration friction low for applications already built against an OpenAI-style chat completion interface.

## Open Weights Alongside a Hosted API

Unlike providers that offer only closed, API-only models, Mistral releases several model families with open weights under permissive licenses, letting teams self-host the exact same model that the hosted API serves. This gives a genuine choice between a managed API for convenience and self-hosting via a framework like vLLM for full control over latency, cost, and data locality, without changing which model you're actually running.

## Function Calling and JSON Mode

Mistral's API supports structured function calling and a JSON-constrained output mode, letting an application define a set of callable tools with typed parameters and have the model select and populate them reliably, following the same general pattern established by other providers' function-calling APIs.

## Data Residency and Governance

Mistral markets EU-based hosting and processing as a differentiator for organizations with strict data residency requirements under GDPR or sector-specific EU regulation, an increasingly relevant factor for public sector and regulated-industry deployments in Europe that need contractual guarantees about where data is processed and stored.

## Practical Guidance

Choose Mistral when open-weight flexibility, EU data residency, or a strong cost-to-performance ratio matter for your use case. Since the smaller Mistral models are specifically tuned for efficiency, benchmark them directly against your task rather than assuming a larger, more expensive model from another provider is necessary — many production tasks (classification, extraction, short-form generation) don't need frontier-scale models to perform well.
