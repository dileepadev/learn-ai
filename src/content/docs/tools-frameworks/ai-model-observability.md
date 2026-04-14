---
title: AI Model Observability
description: Understand how to monitor and debug AI and LLM systems in production using observability tools like LangSmith, Weights & Biases, Arize Phoenix, and OpenTelemetry — covering traces, metrics, drift detection, and evaluation pipelines.
---

**AI model observability** is the practice of instrumenting, monitoring, and debugging machine learning and large language model systems in production. Where traditional MLOps monitoring tracks prediction distributions and data drift, modern AI observability adds **trace-level visibility** into multi-step reasoning chains, tool calls, and retrieval pipelines — enabling faster debugging and regression detection.

## Why Observability Has Become Critical

Production AI systems — especially those built on LLMs with RAG pipelines, tool use, and multi-agent workflows — fail in ways that are qualitatively different from classical ML systems:

- **Non-determinism:** The same input can produce different outputs across runs
- **Prompt sensitivity:** Small prompt changes can silently degrade quality
- **Compound failures:** A failure in retrieval causes a downstream response failure; pinpointing the root cause is hard without traces
- **Hallucinations:** Hard to detect without per-response evaluation
- **Latency variability:** Token generation speed fluctuates based on model load, context length, and provider status

## The Three Pillars of AI Observability

### 1. Traces
A **trace** captures the full execution graph of one AI invocation:
- Sub-spans for each LLM call, tool use, database query, and retrieval step
- Input and output at each node
- Latency and token counts per span
- Errors and retries

Traces allow you to answer: *"Why did this user's request fail? Which step was slow? What context was retrieved?"*

### 2. Metrics
Aggregated measurements across many requests:
- **Throughput:** Requests per second
- **Latency percentiles:** p50, p95, p99 for end-to-end and per-span times
- **Token usage:** Input, output, and total tokens — directly tied to cost
- **Error rate:** Failed requests by error type
- **Quality scores:** Model-graded quality signals averaged over time

### 3. Evaluations
Automated or human assessments of response quality:
- **Faithfulness:** Does the response match the retrieved context?
- **Relevance:** Is the response relevant to the question?
- **Groundedness:** Are claims supported by cited sources?
- **Toxicity / safety:** Does the response violate content policies?

## LangSmith

LangSmith is LangChain's built-in observability platform, deeply integrated with LangChain agents, chains, and RAG pipelines.

**Key features:**
- Automatic tracing via `LANGCHAIN_TRACING_V2=true` environment variable — no code changes needed
- Trace explorer with full chain execution trees
- Dataset management: save traces as evaluation datasets
- Prompt playground: iterate on prompts and compare results
- Built-in evaluators (LLM-as-judge, custom Python)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "<your-key>"
os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"

# All subsequent LangChain calls are automatically traced
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("Explain positional encodings")
# The full trace is visible in the LangSmith dashboard
```

## Weights & Biases (W&B) — Weave

W&B's **Weave** product extends the company's experiment tracking platform to LLM applications:

- **Automatic tracing** of function calls using `@weave.op()` decorator
- **Evaluation harnesses:** Run structured eval suites on datasets with scoring functions
- **Model versioning:** Track prompt versions and model configs alongside evaluations
- **Dashboard integration:** Combine training metrics, fine-tuning experiments, and production traces in one platform

```python
import weave

weave.init("my-llm-app")

@weave.op()
def generate_answer(question: str) -> str:
    # Traced automatically
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
```

## Arize Phoenix

**Arize Phoenix** is an open-source AI observability tool focused on embedding-level analysis and LLM evaluation:

- **Embedding drift detection:** Compare embedding distributions between reference and production sets using UMAP clustering
- **Retrieval quality analysis:** Measure cosine similarity between query and retrieved context embeddings
- **LLM eval engine:** Run LLM-as-judge evals using built-in templates (faithfulness, relevance, harmfulness)
- **OpenTelemetry-compatible:** Ships spans conforming to the OpenInference semantic conventions
- **Local first:** Can run entirely on-premise without cloud connectivity

```python
import phoenix as px
from phoenix.otel import register

tracer_provider = register(project_name="my-rag-app")
# Instrument OpenAI and LangChain automatically
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

## OpenTelemetry for AI (OpenInference)

**OpenInference** is an open specification for AI/LLM traces built on OpenTelemetry. It defines semantic conventions for:
- LLM span attributes (`llm.model_name`, `llm.token_count.prompt`, `llm.token_count.completion`)
- Retrieval spans (`retrieval.documents`, `document.content`, `document.score`)
- Tool use spans (`tool.name`, `tool.parameters`)

By adopting OpenInference, applications can export traces to any backend: Arize Phoenix, Jaeger, Grafana Tempo, LangSmith, or custom collectors.

## Helicone

**Helicone** is a lightweight LLM gateway that sits between your application and model providers. It:
- Logs all requests and responses with zero code changes (just change the base URL)
- Tracks cost, latency, and error rates per model/user/prompt template
- Supports prompt caching and rate limiting
- Provides a dashboard for exploring historical requests

```python
# Change the base URL to route through Helicone
client = OpenAI(
    base_url="https://oai.helicone.ai/v1",
    default_headers={"Helicone-Auth": "Bearer <your-key>"}
)
```

## Comparison of Tools

| Tool | Open Source | LLM Traces | Eval Framework | Embedding Analysis | Best For |
|---|---|---|---|---|---|
| **LangSmith** | No | ✓ (deep LangChain integration) | ✓ | Limited | LangChain teams |
| **W&B Weave** | No | ✓ | ✓ | Limited | Teams using W&B for training |
| **Arize Phoenix** | Yes | ✓ (OpenInference) | ✓ | ✓ (UMAP drift) | RAG quality analysis |
| **Helicone** | Yes (proxy) | ✓ (gateway-level) | Limited | No | Cost monitoring |
| **Grafana + Prometheus** | Yes | Via OTel | No | No | Infrastructure teams |
| **Langfuse** | Yes | ✓ | ✓ | No | Self-hosted tracing |

## Evaluation Pipelines in Production

Evaluation should be continuous, not just a pre-release activity.

### Online Evaluation
Run a small sample of production responses through an LLM judge in real-time:
- Random sample 1-5% of production traffic
- Score for faithfulness, relevance, and safety
- Trigger alerts when quality degradation is detected

### Regression Evaluation
Before any change (prompt update, model upgrade, retrieval change):
1. Define a golden dataset of inputs and reference answers
2. Run both old and new configurations against the dataset
3. Compare quality scores — only deploy if no regressions

### A/B Testing
Route two user cohorts to different configurations and compare downstream business metrics (task completion rate, user feedback ratings).

## Alerting and SLOs

Define **service level objectives** for AI systems:
- **Latency SLO:** p95 response time < 5 seconds
- **Quality SLO:** Average faithfulness score > 0.85 on sampled requests
- **Cost SLO:** Average spend per request < $0.01

Set up alerts when SLOs are breached and route them to the on-call engineering team.

## Further Reading

- LangSmith Documentation: https://docs.smith.langchain.com
- Arize Phoenix Documentation: https://docs.arize.com/phoenix
- W&B Weave Documentation: https://weave-docs.wandb.ai
- OpenInference Specification: https://github.com/Arize-ai/openinference
- Langfuse: https://langfuse.com
