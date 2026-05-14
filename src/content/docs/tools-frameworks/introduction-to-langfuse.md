---
title: Introduction to Langfuse
description: Get started with Langfuse — an open-source LLM observability and evaluation platform — covering tracing, prompt management, evaluations, datasets, and integration with popular AI frameworks.
---

Langfuse is an open-source **LLM observability and evaluation platform** designed to help engineering teams monitor, debug, and improve AI applications. It captures traces of LLM calls and agentic workflows, enables prompt versioning, runs evaluations, and provides datasets for testing — all in a self-hostable package with a cloud-hosted option.

## Why LLM Observability?

Traditional application monitoring focuses on latency, errors, and throughput. LLM applications introduce additional challenges:

- **Non-deterministic outputs**: the same input can produce different outputs across runs
- **Multi-step reasoning**: chains and agents involve many LLM calls, tool calls, and retrieval steps; errors can arise anywhere
- **Prompt sensitivity**: small wording changes can significantly affect output quality
- **Cost tracking**: LLM API costs accumulate at the token level across thousands of calls
- **Regression detection**: quality may silently degrade as models or prompts change

Langfuse addresses all of these by capturing structured traces and enabling qualitative and quantitative evaluation.

## Core Concepts

### Traces

A **trace** is the top-level unit in Langfuse. It represents a single user interaction or pipeline execution and contains a tree of **spans** and **observations**:

- **Spans**: general units of work (e.g., a retrieval step, document processing)
- **Generations**: LLM calls, capturing input (prompt), output, model, token counts, latency, and cost
- **Events**: lightweight markers for noteworthy moments (e.g., cache hit, tool invocation)

Traces are organized hierarchically, making it easy to see exactly which sub-step produced a bad output in a complex agent.

### Sessions and Users

- **Sessions**: groups multiple traces belonging to the same conversation (e.g., all turns of a chat session)
- **Users**: associate traces with specific users for per-user analysis and privacy controls

### Scores

**Scores** attach evaluation signals to traces or individual generations:

- **Manual scores**: human annotators rate quality on a numeric or categorical scale
- **LLM-as-judge scores**: an evaluator LLM grades outputs on specific criteria
- **Automated scores**: computed from code (e.g., exact match, ROUGE, latency threshold violations)

Scores aggregate into dashboards showing quality trends over time.

## Getting Started: Installation

```bash
pip install langfuse
```

For cloud-hosted Langfuse, create an account at langfuse.com and obtain API keys. For self-hosting, use the Docker Compose setup:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

## Instrumenting Your Code

### Direct SDK Integration

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",  # or your self-hosted URL
)

# Create a trace
trace = langfuse.trace(
    name="rag-query",
    user_id="user-42",
    session_id="session-abc",
    input={"query": "What is retrieval-augmented generation?"},
)

# Add a generation (LLM call)
generation = trace.generation(
    name="openai-completion",
    model="gpt-4o",
    model_parameters={"temperature": 0.2, "max_tokens": 512},
    input=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is retrieval-augmented generation?"},
    ],
    usage={"input": 120, "output": 85},
)

response = "RAG combines retrieval of relevant documents with language model generation..."

generation.end(output=response)
trace.update(output=response)
langfuse.flush()
```

### OpenAI Drop-In Integration

Langfuse provides an OpenAI wrapper that automatically traces all API calls:

```python
from langfuse.openai import openai  # drop-in replacement

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain diffusion models briefly."}],
    langfuse_trace_name="explain-diffusion",  # optional metadata
    langfuse_tags=["production", "docs"],
)
```

Every call is automatically captured as a generation in Langfuse with tokens, cost, and latency.

### LangChain Integration

```python
from langfuse.callback import CallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
)

llm = ChatOpenAI(model="gpt-4o", callbacks=[langfuse_handler])
chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

result = chain.invoke({"query": "Explain RAG"})
```

All LangChain steps — retrieval, prompting, generation — appear as a structured trace.

### LlamaIndex Integration

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

langfuse_callback = LlamaIndexCallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
)

Settings.callback_manager = CallbackManager([langfuse_callback])

# All subsequent LlamaIndex operations are traced automatically
```

### Decorator API

For function-level tracing without manually managing spans:

```python
from langfuse.decorators import langfuse_context, observe

@observe()
def retrieve_documents(query: str) -> list:
    langfuse_context.update_current_span(name="retrieval", input={"query": query})
    docs = vector_db.similarity_search(query, k=5)
    langfuse_context.update_current_span(output={"num_docs": len(docs)})
    return docs

@observe()
def generate_answer(query: str, docs: list) -> str:
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}"
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

@observe()
def rag_pipeline(query: str) -> str:
    docs = retrieve_documents(query)
    answer = generate_answer(query, docs)
    return answer
```

`@observe()` automatically creates a span for each function, nesting them correctly into a trace tree.

## Prompt Management

Langfuse serves as a **prompt registry** with versioning, so prompts are managed centrally and changes are tracked:

```python
# Fetch a prompt by name (latest version or specific version)
prompt = langfuse.get_prompt("rag-system-prompt", version=3)

# Use the prompt template
compiled = prompt.compile(
    context="...",
    question="...",
)

# Link generations to the prompt for tracking which prompt version produced each output
generation = trace.generation(
    name="answer-generation",
    prompt=prompt,  # associates the prompt version with this generation
    input=compiled,
    model="gpt-4o",
)
```

The Langfuse UI shows which prompt versions are being used in production, their associated quality scores, and allows rollback or promotion of specific versions.

## Datasets and Evaluations

### Creating Datasets

Datasets in Langfuse are collections of input-expected output pairs used for evaluation:

```python
# Create a dataset
dataset = langfuse.create_dataset(name="rag-eval-set")

# Add items
langfuse.create_dataset_item(
    dataset_name="rag-eval-set",
    input={"query": "What is the capital of France?"},
    expected_output="Paris",
)
```

Dataset items can also be created from existing traces by flagging interesting examples directly from the UI.

### Running Evaluations

```python
def run_pipeline(input_item):
    return rag_pipeline(input_item["query"])

def evaluate_output(input_item, output, expected_output):
    # Simple exact-match evaluator
    return 1.0 if expected_output.lower() in output.lower() else 0.0

# Run over the dataset
dataset = langfuse.get_dataset("rag-eval-set")

for item in dataset.items:
    with item.observe(run_name="rag-v2-eval") as trace_id:
        output = run_pipeline(item.input)

    # Score the run
    langfuse.score(
        trace_id=trace_id,
        name="contains-answer",
        value=evaluate_output(item.input, output, item.expected_output),
    )
```

Results aggregate per run, enabling comparison between model or prompt versions.

### LLM-as-Judge

```python
from langfuse.evaluations import EvaluationClient

eval_client = EvaluationClient(langfuse)

# Run an LLM judge on all scored traces
eval_client.evaluate(
    trace_id=trace_id,
    template_name="hallucination-check",  # built-in or custom template
    variables={"output": output, "context": context},
)
```

Langfuse ships with built-in evaluation templates for hallucination, correctness, conciseness, and toxicity.

## Analytics and Dashboards

The Langfuse UI provides:

- **Trace explorer**: full-text search over traces with filters by model, tags, users, and time
- **Quality dashboard**: aggregate score distributions per prompt version, model, or tag
- **Cost dashboard**: token usage and estimated cost by model and time period
- **Latency dashboard**: p50/p95/p99 latency breakdowns by model and pipeline step
- **User analytics**: per-user trace counts, quality, and cost

These views enable identifying regressions (score drops) correlated with prompt or model changes.

## Self-Hosting and Data Privacy

A major advantage of Langfuse is **self-hostability**. Because LLM applications often process sensitive data (user queries, documents, PII), keeping traces on your own infrastructure may be a requirement.

Langfuse can be deployed via:

```bash
# Docker Compose (development)
docker compose up -d

# Kubernetes (production)
helm install langfuse langfuse/langfuse \
  --set postgresql.auth.password=your-password \
  --set langfuse.nextauth.secret=your-secret
```

The self-hosted version has feature parity with the cloud-hosted offering for core tracing and evaluation functionality.

## Comparison with Alternatives

| Feature | Langfuse | LangSmith | Arize Phoenix | Weights & Biases |
| --- | --- | --- | --- | --- |
| Open-source | Yes | No | Yes | Partial |
| Self-hostable | Yes | No | Yes | No |
| Prompt management | Yes | Yes | No | No |
| LLM-as-judge evals | Yes | Yes | Yes | No |
| Dataset management | Yes | Yes | Limited | Yes |
| Framework integrations | Many | Many | Many | Many |
| Focus | LLM observability | LLM observability | ML + LLM | ML experiments |

## Summary

Langfuse provides a comprehensive observability layer for LLM applications with:

- **Tracing**: structured trace trees capturing every LLM call, retrieval, and tool use
- **Prompt management**: versioned prompts with production usage tracking
- **Evaluation**: datasets, LLM-as-judge, and custom scorer integration
- **Analytics**: quality, cost, and latency dashboards

Its open-source, self-hostable design makes it a strong choice for teams with data privacy requirements or those wanting to avoid vendor lock-in. Integration with LangChain, LlamaIndex, OpenAI, and the decorator API covers the vast majority of LLM application patterns with minimal instrumentation overhead.
