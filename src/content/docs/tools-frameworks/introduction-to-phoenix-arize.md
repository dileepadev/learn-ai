---
title: Introduction to Arize Phoenix
description: Get started with Arize Phoenix — the open-source AI observability platform for tracing, evaluating, and debugging LLM applications, RAG pipelines, and AI agents with OpenTelemetry-native instrumentation.
---

Arize Phoenix is an open-source AI observability platform designed to help engineers trace, evaluate, and debug LLM applications, RAG pipelines, and AI agents. Built on OpenTelemetry, it provides a visual interface for inspecting traces, running evaluations, and identifying performance issues in both development and production.

## Why AI Observability Matters

Traditional software monitoring (latency, error rates, uptime) is insufficient for AI systems. An LLM application can return a response with 200 OK status while:

- Hallucinating facts not present in the retrieved context
- Ignoring the user's intent
- Leaking sensitive information from the prompt
- Producing inconsistent answers to semantically equivalent questions

Phoenix provides the tooling to detect and diagnose these AI-specific failure modes.

## Core Concepts

### Traces and Spans

Phoenix uses the OpenTelemetry tracing model. A **trace** represents a single end-to-end request through your system. Each trace contains **spans** — individual units of work:

- `LLM` span: a single call to a language model
- `RETRIEVER` span: a vector store query
- `CHAIN` span: a sequence of steps in a LangChain/LlamaIndex chain
- `TOOL` span: a tool call in an agent
- `EMBEDDING` span: an embedding computation

Spans capture inputs, outputs, latency, token counts, and custom metadata, forming a tree that mirrors your application's execution structure.

### The OpenInference Standard

Phoenix uses **OpenInference** — an open standard for AI observability built on top of OpenTelemetry semantic conventions. This means traces from Phoenix are portable: you can export them to any OTel-compatible backend (Jaeger, Grafana, Honeycomb) while retaining AI-specific metadata like prompt templates, retrieved documents, and LLM parameters.

## Installation

```bash
pip install arize-phoenix
# With OpenAI auto-instrumentation
pip install arize-phoenix-otel openinference-instrumentation-openai
```

## Launching Phoenix

```python
import phoenix as px

# Start the Phoenix app (opens browser UI at http://localhost:6006)
session = px.launch_app()
```

Phoenix can also run as a standalone server:

```bash
python -m phoenix.server.main serve
```

Or via Docker:

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

## Auto-Instrumentation

Phoenix provides zero-code instrumentation for popular LLM frameworks via the `openinference-instrumentation-*` family of packages.

### OpenAI

```python
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(project_name="my-llm-app")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# All subsequent OpenAI calls are automatically traced
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

### LangChain

```python
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
# LangChain chains, retrievers, and agents are now traced automatically
```

### LlamaIndex

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
```

### Other Supported Frameworks

Phoenix has instrumentation packages for: Anthropic, AWS Bedrock, Azure OpenAI, Google Gemini, Mistral, DSPy, CrewAI, Haystack, Guardrails AI, and more.

## Manual Instrumentation

For custom code, use the OpenTelemetry SDK directly:

```python
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

tracer = trace.get_tracer(__name__)

def call_llm(prompt: str) -> str:
    with tracer.start_as_current_span("custom-llm-call") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        response = my_llm_call(prompt)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, count_tokens(response))
        return response
```

## Querying Traces Programmatically

Phoenix exposes collected traces as a pandas-compatible dataframe:

```python
import phoenix as px

# Get all traces for a project
traces_df = px.Client().get_spans_dataframe(project_name="my-llm-app")

# Filter LLM spans with high latency
slow_llm = traces_df[
    (traces_df["span_kind"] == "LLM") &
    (traces_df["latency_ms"] > 5000)
]

# Inspect retrieved documents for RAG spans
rag_spans = traces_df[traces_df["span_kind"] == "RETRIEVER"]
print(rag_spans[["input.value", "retrieval.documents"]].head())
```

## LLM Evaluations

Phoenix integrates directly with **Phoenix Evals** (`phoenix.evals`) — a library of LLM-as-judge evaluators that run against your traced data.

### Built-in Evaluators

```python
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator,
    run_evals,
)
from phoenix.evals import OpenAIModel

eval_model = OpenAIModel(model="gpt-4o-mini")

# Run evals on RAG traces
hallucination_eval = HallucinationEvaluator(eval_model)
qa_eval = QAEvaluator(eval_model)

results = run_evals(
    dataframe=rag_spans,
    evaluators=[hallucination_eval, qa_eval],
    provide_explanation=True,  # get LLM reasoning for each judgment
)
```

Each evaluator returns labels (e.g., `"hallucinated"` / `"factual"`) and optional explanations that Phoenix stores and displays alongside the original traces.

### Custom Evaluators

```python
from phoenix.evals import LLMCriteria, llm_classify

custom_template = """
You are evaluating whether a customer service response is professional.
[Response]: {response}
Answer YES or NO.
"""

results = llm_classify(
    dataframe=traces_df,
    template=custom_template,
    model=eval_model,
    rails=["YES", "NO"],
)
```

## Datasets and Experiments

Phoenix supports an **experiment** workflow for comparing prompt variants or model versions:

```python
import phoenix as px
from phoenix.experiments import run_experiment

client = px.Client()

# Upload a dataset
dataset = client.upload_dataset(
    dataframe=test_questions_df,
    dataset_name="customer-qa-v1",
    input_keys=["question"],
    output_keys=["reference_answer"],
)

# Define your task
def run_rag_pipeline(example):
    answer = my_rag_chain.invoke(example["question"])
    return {"response": answer}

# Run experiment and evaluate
experiment = run_experiment(
    dataset=dataset,
    task=run_rag_pipeline,
    evaluators=[qa_eval, hallucination_eval],
    experiment_name="prompt-v2-test",
)
```

Results are stored in Phoenix's UI, allowing side-by-side comparison across experiment runs.

## Embedding Visualization

Phoenix includes a powerful **embedding explorer** for visualizing high-dimensional embeddings:

- Upload query embeddings and document embeddings
- UMAP-based 2D/3D projection
- Color by evaluation score, latency, or custom metadata
- Identify clusters of failing queries or low-relevance retrievals

```python
px.launch_app(
    primary=px.Dataset(
        queries_df,
        schema=px.Schema(
            prediction_id_column_name="id",
            embedding_feature_column_names={
                "query_embedding": px.EmbeddingColumnNames(
                    vector_column_name="embedding",
                    raw_data_column_name="query_text",
                )
            },
        ),
    )
)
```

## Phoenix vs. Other Observability Tools

| Feature | Phoenix | LangSmith | Langfuse | Helicone |
| --- | --- | --- | --- | --- |
| Open source | Yes | No | Yes | Partial |
| Self-hostable | Yes | No | Yes | No |
| OpenTelemetry native | Yes | No | No | No |
| Built-in evals | Yes | Yes | Yes | No |
| Embedding viz | Yes | No | No | No |
| Experiment tracking | Yes | Yes | Yes | No |

## Production Deployment

For production use, Phoenix can be deployed as a persistent server with a PostgreSQL backend:

```bash
docker run \
  -e PHOENIX_SQL_DATABASE_URL=postgresql://user:pass@host:5432/phoenix \
  -p 6006:6006 \
  -p 4317:4317 \
  arizephoenix/phoenix:latest
```

Traces from your application are sent via gRPC (port 4317) or HTTP (port 4318) using the standard OTLP protocol, making it compatible with existing OpenTelemetry infrastructure.

## Summary

Arize Phoenix provides a comprehensive, open-source observability stack for LLM applications:

- **Tracing**: zero-code instrumentation via OpenInference for all major frameworks
- **Evaluation**: LLM-as-judge evaluators for hallucination, relevance, QA correctness, and custom criteria
- **Experiments**: structured prompt/model comparison with automatic evaluation
- **Embeddings**: visual debugging of retrieval quality

Its OpenTelemetry-native architecture makes it a natural fit for teams that want AI observability alongside their existing distributed tracing infrastructure.
