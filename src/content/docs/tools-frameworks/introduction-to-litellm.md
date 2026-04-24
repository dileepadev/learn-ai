---
title: Introduction to LiteLLM
description: Learn how LiteLLM provides a unified Python SDK and proxy server for calling 100+ LLM providers through a standardized OpenAI-compatible interface — with built-in cost tracking, load balancing, fallbacks, and observability.
---

**LiteLLM** is an open-source Python library and proxy server that provides a unified interface for calling large language models from over 100 providers — including OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, Cohere, Mistral, Ollama, and many more — through a single, standardized API that is compatible with the OpenAI SDK format. LiteLLM eliminates the need to write and maintain provider-specific API integration code, enabling teams to switch providers, implement fallbacks, and manage costs centrally.

As organizations adopt multiple LLMs for different use cases or cost tiers, a unified LLM gateway becomes essential infrastructure. LiteLLM provides this gateway at both the Python library level (for application code) and the proxy server level (for organization-wide LLM management).

## Why LiteLLM?

Different LLM providers have different APIs, authentication schemes, request formats, and response structures. Building an application on top of multiple providers means:

- Writing provider-specific client code for each integration.
- Managing different authentication schemes (API keys, IAM roles, OAuth).
- Handling provider-specific error codes and retry logic.
- Tracking costs across providers with different pricing models.
- Implementing fallback logic when a provider is unavailable.

LiteLLM abstracts all of this — the same code path handles any provider, and switching providers or adding fallbacks requires changing configuration, not code.

## Installation and Basic Usage

```bash
pip install litellm
```

The primary interface mirrors OpenAI's Python SDK exactly:

```python
import litellm

# OpenAI
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

# Anthropic Claude — same code, different model string
response = litellm.completion(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

# Google Gemini — same code
response = litellm.completion(
    model="gemini/gemini-1.5-pro",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

# Azure OpenAI — same code, different model string format
response = litellm.completion(
    model="azure/my-gpt4-deployment",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    api_base="https://my-resource.openai.azure.com",
    api_key="my-azure-api-key"
)

# All responses share the same OpenAI-compatible response format
print(response.choices[0].message.content)
print(response.usage.total_tokens)
```

Model strings follow the convention `provider/model-name` or just `model-name` for OpenAI models.

## Async and Streaming Support

LiteLLM provides async and streaming variants for all providers:

```python
import asyncio
import litellm

# Async completion
async def async_example():
    response = await litellm.acompletion(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Write a haiku about AI."}]
    )
    return response.choices[0].message.content

# Streaming
for chunk in litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 10 slowly."}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Async streaming
async def async_stream():
    response = await litellm.acompletion(
        model="gemini/gemini-1.5-flash",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        stream=True
    )
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

## Fallbacks and Load Balancing

LiteLLM's **Router** enables sophisticated routing logic:

```python
from litellm import Router

model_list = [
    {
        "model_name": "gpt-4-equivalent",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": "sk-...",
        },
        "tpm": 100000,  # tokens per minute
        "rpm": 100,     # requests per minute
    },
    {
        "model_name": "gpt-4-equivalent",
        "litellm_params": {
            "model": "claude-3-5-sonnet-20241022",
            "api_key": "sk-ant-...",
        },
        "tpm": 80000,
        "rpm": 80,
    },
    {
        "model_name": "gpt-4-equivalent",
        "litellm_params": {
            "model": "azure/my-gpt4",
            "api_base": "https://my-resource.openai.azure.com",
            "api_key": "...",
        },
        "tpm": 150000,
        "rpm": 200,
    },
]

router = Router(
    model_list=model_list,
    routing_strategy="latency-based-routing",   # or "least-busy", "usage-based-routing"
    fallbacks=[{"gpt-4-equivalent": ["gpt-3.5-equivalent"]}],
    num_retries=3,
    timeout=30
)

response = router.completion(
    model="gpt-4-equivalent",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Routing strategies include:

- **Latency-based**: Route to the deployment with the lowest observed latency.
- **Usage-based**: Route to deployments with remaining capacity, avoiding rate limits.
- **Least-busy**: Route to the deployment with the fewest active requests.
- **Fallback chains**: Automatically retry with backup models on failures.

## Cost Tracking

LiteLLM tracks token usage and cost across all providers automatically:

```python
import litellm

# Enable verbose logging to see cost per request
litellm.success_callback = ["langfuse"]  # or "langsmith", "helicone", etc.

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

# Cost is computed from the response usage and provider pricing
cost = litellm.completion_cost(completion_response=response)
print(f"Cost: ${cost:.6f}")

# Access usage details
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

Budget limits can be enforced at the router level to prevent cost overruns across a team or application.

## The LiteLLM Proxy Server

The **LiteLLM Proxy** is an OpenAI-compatible HTTP server that acts as a central LLM gateway for an entire organization:

```bash
# Install with proxy support
pip install 'litellm[proxy]'

# Create a config file
cat > litellm_config.yaml << EOF
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: gpt-4
    litellm_params:
      model: azure/my-deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY

router_settings:
  routing_strategy: usage-based-routing

general_settings:
  master_key: sk-my-proxy-key
EOF

# Start the proxy
litellm --config litellm_config.yaml --port 8000
```

Once running, any OpenAI-compatible client can use the proxy:

```python
from openai import OpenAI

# Point the OpenAI client to the LiteLLM proxy
client = OpenAI(
    base_url="http://localhost:8000",
    api_key="sk-my-proxy-key"
)

# This automatically routes through LiteLLM with load balancing and fallbacks
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello from the proxy!"}]
)
```

The proxy enables:

- **Central authentication**: Teams use a single proxy key; the proxy manages provider keys.
- **Budget enforcement**: Per-user, per-team, or per-application spending limits.
- **Audit logging**: Every request and response is logged centrally.
- **Model aliasing**: Teams use friendly model names; the proxy maps to actual provider deployments.

## Observability Integrations

LiteLLM integrates with popular observability platforms via callbacks:

```python
import litellm

# Enable logging to multiple platforms
litellm.success_callback = ["langfuse", "langsmith"]
litellm.failure_callback = ["slack"]  # Alert on failures

# Environment variables for each platform
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGCHAIN_API_KEY"] = "ls-..."
```

Supported platforms include Langfuse, LangSmith, Weights & Biases, Helicone, Arize, Athina, and more. Each integration captures request/response pairs, latency, token counts, and costs for centralized monitoring.

## Embeddings and Other Operations

LiteLLM also provides unified interfaces for embeddings and image generation:

```python
# Embeddings — unified across OpenAI, Cohere, Mistral, etc.
embedding_response = litellm.embedding(
    model="text-embedding-3-small",
    input=["Hello world", "LiteLLM is great"]
)
embeddings = [e["embedding"] for e in embedding_response.data]

# Image generation
image_response = litellm.image_generation(
    model="dall-e-3",
    prompt="A peaceful mountain lake at sunrise"
)
```

## Comparison with Alternatives

| Feature | LiteLLM | OpenAI SDK | Provider SDKs |
| --- | --- | --- | --- |
| Multi-provider support | 100+ | OpenAI only | One per SDK |
| OpenAI-compatible API | Yes | Yes | No |
| Load balancing | Yes | No | No |
| Fallbacks | Yes | No | No |
| Cost tracking | Built-in | Partial | No |
| Proxy server | Yes | No | No |
| Observability | 20+ integrations | Partial | Minimal |

LiteLLM is ideal for teams building production LLM applications that need provider flexibility, cost visibility, and operational reliability without building custom gateway infrastructure.
