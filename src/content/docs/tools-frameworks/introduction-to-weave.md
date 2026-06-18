---
title: Introduction to Weights & Biases Weave
description: A practical guide to Weights & Biases Weave — the LLM observability and evaluation platform. Covers tracing LLM calls, building evaluation pipelines, dataset management, scorers, and integrating Weave into production AI applications.
---

**Weights & Biases Weave** is an open-source LLM observability, tracing, and evaluation platform built on top of the Weights & Biases ecosystem. While W&B's flagship product (the `wandb` library) targets training-time experiment tracking — logging loss curves, hyperparameters, and model checkpoints — Weave is purpose-built for the *inference-time* lifecycle of LLM applications: tracing chains of LLM calls, evaluating outputs against human or automated criteria, and iterating on prompts and pipelines with confidence.

## Why LLM Applications Need Dedicated Observability

Traditional software monitoring (latency, error rates, CPU usage) is necessary but insufficient for LLM-powered applications. Key additional observability needs:

- **Input/output capture**: Every prompt sent and response received must be logged with full context for debugging and replay.
- **Chain tracing**: Modern LLM applications involve chains of calls — retrieval, reranking, generation, tool use — whose interactions determine final output quality. Tracing must capture the entire chain, not just individual calls.
- **Qualitative evaluation**: LLM output correctness is often not binary — it requires human judgment or LLM-as-judge evaluation. Standard metrics (latency, cost) don't capture response quality.
- **Dataset management**: Evaluation requires curated datasets of (input, expected output) pairs that must be versioned and maintained over time.
- **Prompt versioning**: Prompt changes are deployments — they change system behavior as dramatically as code changes. Weave treats prompts as versioned artifacts.

## Core Concepts

### Ops (Traced Functions)

The fundamental unit in Weave is an **Op** — a Python function decorated with `@weave.op()`. Any function decorated this way is automatically traced when called:

```python
import weave

weave.init("my-project")

@weave.op()
def call_llm(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

When `call_llm` is invoked:
- Inputs (prompt) and outputs (response text) are captured.
- Wall-clock duration is recorded.
- Token counts and cost are extracted from the OpenAI response.
- A **Call** record is created in the Weave backend, visible in the UI.

Ops can be **nested** — a parent function calling child ops creates a trace tree, making it easy to understand which sub-call caused a slow or incorrect response.

### Datasets

A **Weave Dataset** is a versioned, structured collection of examples used for evaluation:

```python
dataset = weave.Dataset(
    name="qa-benchmark",
    rows=[
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
    ]
)
weave.publish(dataset)
```

Datasets are versioned automatically — each `publish()` call creates a new version. Previous versions remain accessible, enabling reproducible evaluation across dataset updates.

### Scorers

**Scorers** evaluate LLM outputs. Weave supports:

**Function scorers** — simple Python functions returning a score:

```python
def exact_match(output: str, answer: str) -> bool:
    return output.strip().lower() == answer.strip().lower()
```

**Class-based scorers** — for stateful or LLM-based evaluation:

```python
class LLMJudge(weave.Scorer):
    model_id: str = "gpt-4o"

    @weave.op()
    def score(self, output: str, question: str, answer: str) -> dict:
        judgment = call_judge_llm(
            question=question,
            response=output,
            reference=answer
        )
        return {
            "correct": judgment.correct,
            "reasoning": judgment.explanation
        }
```

**Built-in scorers**: Weave ships with scorers for common patterns — hallucination detection, toxicity, summarization faithfulness, and RAG relevance, all implemented as LLM-as-judge prompts with structured output parsing.

### Evaluations

An **Evaluation** ties together a dataset, a model function, and a set of scorers:

```python
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[exact_match, LLMJudge()]
)

results = await evaluation.evaluate(call_llm)
```

Running an evaluation:
1. Iterates over all dataset rows.
2. Calls the model function with each row.
3. Runs each scorer on the (output, row) pair.
4. Aggregates scores and logs all traces to the Weave UI.

Results appear in the **Evaluations** tab — a table comparing model performance across scorers, with drill-down into individual examples to understand failure modes.

## Tracing Multi-Step Pipelines

Weave's trace tree shines for complex LLM pipelines. A RAG pipeline traced with Weave:

```python
@weave.op()
def retrieve(query: str, k: int = 5) -> list[str]:
    # Vector search
    return vector_store.search(query, k=k)

@weave.op()
def generate(query: str, context: list[str]) -> str:
    # LLM generation with retrieved context
    prompt = format_rag_prompt(query, context)
    return call_llm(prompt)

@weave.op()
def rag_pipeline(question: str) -> str:
    context = retrieve(question)
    answer = generate(question, context)
    return answer
```

In the Weave UI, calling `rag_pipeline` creates a trace with:
- `rag_pipeline` as the root span
- `retrieve` as a child span (with query + retrieved documents)
- `generate` as a child span (with formatted prompt + LLM response)
- Each LLM API call as a leaf span (with token counts, latency, cost)

This end-to-end visibility makes it straightforward to identify whether a wrong answer stems from poor retrieval (irrelevant documents) or poor generation (good documents, bad synthesis).

## Prompt Management

Weave introduces **Prompts as versioned artifacts** via the `weave.StringPrompt` and `weave.MessagesPrompt` classes:

```python
system_prompt = weave.StringPrompt("You are a helpful assistant specializing in {domain}.")
weave.publish(system_prompt, name="system-prompt")
```

Published prompts:
- Are versioned automatically on each `publish()`.
- Can be retrieved by version for reproducible evaluation.
- Appear in the UI with a diff view between versions.
- Are linked to the evaluation runs that used them — enabling direct comparison of two prompt versions on the same dataset.

This treats prompt engineering with the same rigor as code — changes are tracked, reviewable, and reversible.

## Integrations

Weave provides automatic instrumentation for popular LLM libraries:

| Library | Integration |
| --- | --- |
| OpenAI | `weave.integrations.openai` — all API calls traced automatically |
| Anthropic | `weave.integrations.anthropic` — messages API traced |
| LangChain | `weave.integrations.langchain` — chains and agents traced |
| LlamaIndex | `weave.integrations.llamaindex` — query engines and pipelines traced |
| DSPy | `weave.integrations.dspy` — modules and optimizers traced |
| Google Gemini | `weave.integrations.google_genai` — generate content calls traced |
| Groq | `weave.integrations.groq` — completions traced |

Enabling automatic tracing for OpenAI requires one import:

```python
import weave
from weave.integrations.openai import weave_client

weave.init("my-project")
# All subsequent OpenAI calls are traced automatically
```

## The Weave UI

The web interface provides:

- **Traces tab**: Timeline of all traced calls with latency waterfall, input/output viewer, and token cost breakdown. Filterable by time, model, and custom tags.
- **Evaluations tab**: Table of evaluation runs comparing different models and prompt versions. Click into any run to see per-example scores and full traces.
- **Datasets tab**: Versioned dataset browser with example-level viewing and export.
- **Leaderboard**: Compare multiple models or prompt variants on the same evaluation dataset side by side.
- **Cost dashboard**: Aggregate token costs by model, time period, and tagged application.

## Production Usage Patterns

### A/B Testing Prompts

```python
@weave.op()
def pipeline_v1(question: str) -> str:
    return call_llm(PROMPT_V1.format(question=question))

@weave.op()
def pipeline_v2(question: str) -> str:
    return call_llm(PROMPT_V2.format(question=question))

# Run both through the same evaluation
eval = weave.Evaluation(dataset=dataset, scorers=[LLMJudge()])
await eval.evaluate(pipeline_v1)
await eval.evaluate(pipeline_v2)
# Compare side-by-side in UI
```

### Online Evaluation

Attach scorers to production traces to continuously monitor quality:

```python
@weave.op()
def production_handler(user_input: str) -> str:
    response = rag_pipeline(user_input)
    # Async scoring in production
    score_toxicity.score(output=response)
    return response
```

### Human Annotation Workflows

Weave's UI supports **human annotation** — reviewers can open traces and attach structured labels (thumbs up/down, categorical ratings, free-text notes) directly to individual calls. These annotations are stored as first-class data and can be used to build future evaluation datasets.

## Comparing Weave with Alternatives

| Feature | Weave | LangSmith | Langfuse | Arize Phoenix |
| --- | --- | --- | --- | --- |
| Open-source | Yes (core) | No | Yes | Yes |
| Training integration | W&B native | No | No | No |
| Evaluations | Built-in | Built-in | Via experiments | Built-in |
| Dataset versioning | Yes | Yes | Yes | Limited |
| Self-hostable | Yes | No | Yes | Yes |
| LangChain integration | Yes | Native | Yes | Yes |

Weave's tightest differentiation is its native connection to the broader W&B platform — teams already using W&B for training can extend the same project to cover inference observability, linking training metrics to deployment behavior in a single workspace.

Weave represents the maturation of LLM tooling toward the rigor that production software demands. As LLM applications move from demos to critical infrastructure, systematic tracing, versioned evaluations, and reproducible benchmarking become not optional extras but foundational engineering practices — and Weave provides a coherent, open-source foundation for building them.
