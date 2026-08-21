---
title: "Introduction to OpenTelemetry for AI"
description: Learn how OpenTelemetry applies to AI and LLM systems — tracing agent workflows, tracking token usage and latency, capturing prompt/response pairs, and building observability pipelines that give you real visibility into production AI applications.
---

A production AI application is a distributed system. A single user request might trigger a chain of LLM calls, tool invocations, database lookups, and external API requests — across multiple services and potentially multiple models. When that request fails, produces incorrect output, or takes three times longer than expected, how do you find out why?

Traditional logging captures isolated events. Traditional metrics capture aggregated counters. Neither tells you the full story of what happened to a specific request as it traveled through your AI pipeline. **OpenTelemetry (OTel)** is the open standard for distributed tracing, metrics, and logging that gives you that full picture — and the AI ecosystem is rapidly adopting it.

## What Is OpenTelemetry?

OpenTelemetry is a CNCF (Cloud Native Computing Foundation) project providing vendor-neutral APIs, SDKs, and tooling for collecting and exporting telemetry data. It emerged from the merger of OpenCensus and OpenTracing in 2019 and is now the standard for observability in cloud-native systems.

OTel defines three **signal types**:

**Traces:** Records of a request's journey through distributed components. A trace consists of **spans** — named, time-stamped operations with attributes (key-value metadata). Spans form a tree: the root span represents the end-to-end request; child spans represent sub-operations.

**Metrics:** Aggregated numerical measurements — counters, histograms, gauges. "Total tokens used per hour" is a metric. "P99 LLM call latency" is a metric.

**Logs:** Structured or unstructured event records. In OTel, logs can be attached to traces (correlated), making them far more useful than standalone log lines.

## Why AI Systems Need Specialized Observability

AI applications have observability needs beyond typical web services:

**Prompt and response capture.** Understanding AI failures requires seeing exactly what was sent to the model and what it returned. This is fundamentally different from web services where request bodies are usually small and structured.

**Token tracking.** LLM costs are denominated in tokens. Understanding and optimizing costs requires per-call, per-model token usage tracked at the trace level — not just aggregate billing.

**Multi-step agent workflows.** An AI agent calling 10 tools across 5 LLM calls creates a complex trace tree. Without tracing, debugging which step failed or went wrong is guesswork.

**Evaluation and quality.** Beyond latency and errors, AI applications care about output quality — hallucination rates, relevance scores, factuality. These "AI quality metrics" need to be captured alongside standard operational metrics.

**Streaming latency.** LLMs often stream tokens. The "time to first token" (TTFT) and "tokens per second" are critical UX metrics with no equivalent in traditional web services.

## OpenTelemetry Semantic Conventions for AI

As of 2024, the OpenTelemetry community has standardized a set of semantic conventions specifically for AI/LLM calls. These conventions define standard attribute names for AI-specific concepts, enabling consistent observability across different frameworks and models.

Key conventions from `semconv.gen_ai.*`:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.system` | The AI system used | `"openai"`, `"anthropic"`, `"ollama"` |
| `gen_ai.request.model` | Requested model name | `"gpt-4o"`, `"claude-3-5-sonnet"` |
| `gen_ai.response.model` | Actual model used (can differ) | `"gpt-4o-2024-08-06"` |
| `gen_ai.usage.input_tokens` | Tokens in the prompt | `1024` |
| `gen_ai.usage.output_tokens` | Tokens in the response | `512` |
| `gen_ai.request.temperature` | Sampling temperature | `0.7` |
| `gen_ai.request.max_tokens` | Token limit | `4096` |
| `gen_ai.operation.name` | Operation type | `"chat"`, `"embeddings"`, `"completion"` |

For agents and RAG:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.tool.name` | Name of tool called by agent |
| `gen_ai.tool.call.id` | ID for a specific tool invocation |
| `db.system` | Database type for vector retrieval |
| `db.vector.query.top_k` | Number of results requested |

## Instrumenting an LLM Application

### Manual Instrumentation

For custom code, you use the OTel Python SDK directly:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("my-ai-app")

# Instrument an LLM call
def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")
        
        # Optionally capture prompt (be careful with sensitive data)
        span.set_attribute("gen_ai.prompt", prompt[:1000])
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Record token usage
        span.set_attribute(
            "gen_ai.usage.input_tokens", 
            response.usage.prompt_tokens
        )
        span.set_attribute(
            "gen_ai.usage.output_tokens", 
            response.usage.completion_tokens
        )
        span.set_attribute(
            "gen_ai.response.model", 
            response.model
        )
        
        return response.choices[0].message.content
```

### Auto-Instrumentation with OpenLLMetry

Manually instrumenting every LLM call is tedious. **OpenLLMetry** (by Traceloop) provides auto-instrumentation for the most common AI libraries — a few lines of setup code instruments all LLM calls automatically:

```python
from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="my-rag-service",
    api_endpoint="http://localhost:4318"
)

# That's it — all OpenAI, Anthropic, LangChain, LlamaIndex calls
# are now automatically traced with OTel semantic conventions
```

OpenLLMetry instruments: OpenAI, Anthropic, Cohere, Azure OpenAI, LangChain, LlamaIndex, ChromaDB, Pinecone, Weaviate, and more.

### Framework-Native Instrumentation

Major AI frameworks are adding native OTel support:

**LangChain:** `LangChainInstrumentor` from OpenLLMetry or the native `opentelemetry-instrumentation-langchain` package automatically traces chains, agents, and tool calls.

**LlamaIndex:** Native OTel integration via `opentelemetry-instrumentation-llamaindex`. Traces queries, retrievals, and LLM calls in RAG pipelines.

**Haystack:** Built-in OTel tracing via `pipeline.run()` spans for each component.

```python
# LlamaIndex example
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()

# All subsequent LlamaIndex calls are auto-traced
query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")
# Spans created: query, retrieval, llm.chat, etc.
```

## Tracing an Agent Workflow

For a multi-step agent, OTel traces create a tree showing exactly what happened:

```
[ROOT] user_request: "Research competitors of Acme Corp"
  [SPAN] agent.plan (12ms)
  [SPAN] tool.web_search: "Acme Corp competitors" (1.2s)
    [SPAN] llm.chat: extract_results (340ms)
      gen_ai.usage.input_tokens: 1847
      gen_ai.usage.output_tokens: 423
  [SPAN] tool.web_search: "Acme Corp market share" (890ms)
    [SPAN] llm.chat: extract_results (290ms)
  [SPAN] llm.chat: synthesize_report (1.1s)
    gen_ai.usage.input_tokens: 4201
    gen_ai.usage.output_tokens: 1205
  [SPAN] format_output (8ms)
Total: 4.1s | Total tokens: 7,723
```

This trace immediately reveals where time was spent, which tool calls were made, how many tokens each step consumed, and where to look if something went wrong.

## Metrics for AI Systems

Beyond tracing, key metrics to instrument:

**Latency metrics:**
- `gen_ai.client.operation.duration` — histogram of end-to-end LLM call durations
- `gen_ai.client.time_to_first_token` — TTFT for streaming responses
- `gen_ai.client.token_rate` — tokens per second generation rate

**Cost metrics:**
- `gen_ai.usage.input_tokens` — counter (by model, by service)
- `gen_ai.usage.output_tokens` — counter (by model, by service)
- Derived: `gen_ai.cost_usd` — computed from token counts and per-model pricing

**Quality metrics:**
- `gen_ai.evaluation.score` — attached evaluation scores (relevance, faithfulness, etc.)
- `gen_ai.tool.call.error_rate` — fraction of tool calls that failed
- Cache hit rate, retry rate, fallback rate

**Application metrics:**
- Request rate and error rate (standard RED metrics)
- Agent step count distribution — how many steps does a typical agent task require?

## Exporters and Backends

OTel is vendor-neutral — data can be sent to many backends:

**Commercial:**
- **Datadog:** Full OTel support with AI-specific dashboards (LLM Observability)
- **Dynatrace:** AI observability features built on OTel
- **New Relic:** Native OTel integration
- **Honeycomb:** Excellent for trace-based debugging

**Open source:**
- **Jaeger:** Distributed tracing UI
- **Zipkin:** Lightweight tracing
- **Prometheus + Grafana:** Metrics visualization
- **OpenSearch/Elasticsearch:** Log storage and search

**AI-specific:**
- **Langfuse:** Open-source LLM observability with OTel ingestion, prompt management, and evaluation
- **Phoenix (Arize):** Open-source AI observability with OTel support
- **Helicone:** LLM proxy with built-in OTel-compatible observability
- **Openlit:** Lightweight OTel-native AI observability

The OTel Collector is a standalone agent that receives telemetry from your application, applies transformations and filtering, and exports to one or more backends:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 5s
  # Redact prompt content for sensitive applications
  attributes:
    actions:
      - key: gen_ai.prompt
        action: delete

exporters:
  otlphttp/langfuse:
    endpoint: https://us.cloud.langfuse.com/api/public/otel
  prometheus:
    endpoint: 0.0.0.0:8889

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp/langfuse]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

## Privacy Considerations

AI observability introduces a specific privacy challenge: **prompts and responses contain user data**. An LLM call's input might include the user's personal information, medical history, or confidential business data.

Best practices:
- **Default to not logging prompts** in production. Add explicit opt-in for debugging sessions.
- **Sanitize before export:** Run a PII detection model on prompts/responses before sending to observability backends. Libraries like Microsoft Presidio can detect and mask names, emails, addresses, etc.
- **Store prompts in secure, short-retention storage** separate from long-term metrics
- **Use OTel processor pipelines** to drop or mask sensitive attributes before export
- **Consider sampling:** Instead of tracing 100% of requests, sample 1-5% and log full prompt/response only for sampled traces

## Getting Started Quickly

The fastest path to AI observability:

```bash
pip install opentelemetry-sdk traceloop-sdk
```

```python
from traceloop.sdk import Traceloop
import openai

Traceloop.init(
    app_name="my-app",
    # Export to local Langfuse or other backend
    api_endpoint="http://localhost:3000/api/public/otel",
)

client = openai.OpenAI()

# This call is automatically traced
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

With this setup, you immediately see LLM call latency, token usage, model metadata, and errors in your observability backend — without writing a single trace instrumentation line manually.

Observability is often the last thing teams think about when building AI applications and the first thing they wish they had when debugging production issues. Starting with OTel instrumentation from day one is one of the highest-ROI investments in your AI engineering stack.
