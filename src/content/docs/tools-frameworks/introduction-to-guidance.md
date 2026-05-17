---
title: Introduction to Microsoft Guidance
description: Get started with Microsoft Guidance — a Python library for constrained, structured generation with LLMs — covering templates, grammars, state machines, interleaved generation-control, and integration with local and cloud models.
---

Microsoft Guidance is an open-source Python library that gives developers fine-grained control over LLM generation. Unlike prompt frameworks that simply format messages and send them to an API, Guidance **interleaves code and generation** — allowing Python logic to steer model output mid-generation, enforce structural constraints, and efficiently reuse computation. The result is a programming paradigm where the model is a programmable co-author rather than a black-box service.

## Core Idea: Programs, Not Prompts

Traditional LLM usage: construct a prompt → send to API → receive complete response.

Guidance: write a **program** that mixes fixed text, model-generated text, and Python control flow. The program executes top-to-bottom; at each `gen()` call the model generates tokens until a constraint is satisfied, then execution continues.

```python
from guidance import models, gen, select

lm = models.OpenAI("gpt-4o")

# A guidance program
with lm.chat() as lm:
    lm += "What is the capital of France?"
    lm += gen("answer", max_tokens=20, stop=".")

print(lm["answer"])  # "Paris"
```

The `gen()` call produces output that is stored in the named variable `answer` and incorporated into the context for subsequent generations.

## Installation

```bash
pip install guidance
```

For local model support via llama.cpp:

```bash
pip install guidance llama-cpp-python
```

## Supported Backends

Guidance works with a wide range of model backends:

- **OpenAI**: `models.OpenAI("gpt-4o")`, `models.OpenAI("gpt-4o-mini")`
- **Anthropic**: `models.Anthropic("claude-3-5-sonnet-20241022")`
- **Azure OpenAI**: `models.AzureOpenAI(...)`
- **Local via llama.cpp**: `models.LlamaCpp("path/to/model.gguf")`
- **Transformers (HuggingFace)**: `models.Transformers("mistralai/Mistral-7B-Instruct-v0.3")`
- **Vertex AI / Gemini**: `models.VertexAI("gemini-2.0-flash")`

## Structured Generation

### `select`: Constrained Choice

Force the model to choose from a discrete set of options:

```python
from guidance import models, select

lm = models.OpenAI("gpt-4o")

lm += "Is the following sentiment positive or negative? 'I love this product!'\nSentiment: "
lm += select(["positive", "negative", "neutral"], name="sentiment")

print(lm["sentiment"])  # "positive"
```

`select` constrains generation at the token level — the model only produces tokens that continue toward one of the listed options. This guarantees a valid output without post-processing or output parsing.

### `gen` with Regex

Use regular expressions to constrain generation to a specific format:

```python
from guidance import models, gen

lm = models.OpenAI("gpt-4o")

lm += "Extract the date from this text: 'The meeting is on 2026-05-17.'\nDate: "
lm += gen("date", regex=r"\d{4}-\d{2}-\d{2}")

print(lm["date"])  # "2026-05-17"
```

The regex constraint is enforced token-by-token during generation — the model cannot produce tokens that would make the output fail the regex. This is implemented via token masking: at each step, only tokens that keep the partial output on a valid regex path are allowed.

### JSON Schema Enforcement

```python
from guidance import models, gen
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "score": {"type": "number", "minimum": 0, "maximum": 100}
    },
    "required": ["name", "age", "score"]
}

lm = models.OpenAI("gpt-4o")
lm += "Generate a student record:\n"
lm += gen("record", json_schema=schema)

record = json.loads(lm["record"])
```

Guidance enforces the JSON schema during generation — every token produced is consistent with the schema structure. This eliminates JSON parsing failures entirely.

## Interleaved Generation and Control

The key Guidance innovation is **interleaving** — Python code can branch, loop, and call functions based on values already generated:

```python
from guidance import models, gen, select

lm = models.OpenAI("gpt-4o")

lm += "Classify this email as spam or not spam: 'Win a million dollars!'\nClassification: "
lm += select(["spam", "not spam"], name="classification")

# Branch based on what the model generated
if lm["classification"] == "spam":
    lm += "\nExplain why this is spam in one sentence: "
    lm += gen("explanation", max_tokens=50)
else:
    lm += "\nSummarize the email topic in one phrase: "
    lm += gen("topic", max_tokens=20)
```

This creates a structured pipeline where downstream generation is conditioned on upstream model decisions — all within a single coherent context.

## Guidance Grammars

For complex structured outputs, Guidance supports **context-free grammars** as constraints:

```python
from guidance import models, gen
from guidance.library import json, select

# Define a grammar for arithmetic expressions
lm = models.LlamaCpp("mistral-7b.gguf")

# Generate a valid Python function signature
lm += "Write a Python function signature for sorting a list:\n"
lm += "def " + gen("func_name", regex=r"[a-z_][a-z_0-9]*")
lm += "(" + gen("params", regex=r"[a-zA-Z_, :]*") + "):\n"
lm += "    " + gen("body", max_tokens=100, stop="\n\n")
```

Guidance compiles grammar rules into token masks applied at each generation step — producing outputs that are grammatically valid by construction.

## Token Healing

A subtle but important feature: standard LLM APIs tokenize the entire prompt, which can cause **boundary artifacts** where the last token of the prompt partially occupies a token that should be in the completion. Guidance implements **token healing** — when a completion begins at a partial token boundary, it backs up by one token and allows the model to re-generate from a clean token boundary.

This is particularly important for constrained generation where token-level constraints must align with tokenization boundaries.

## KV Cache Reuse

For local models (llama.cpp, Transformers), Guidance reuses the **KV cache** across multiple calls that share a common prefix:

```python
lm = models.LlamaCpp("model.gguf")
base = lm + "You are a helpful assistant.\n"

# Each query reuses the cached key-value pairs for the system prompt
for question in questions:
    response = base + f"Q: {question}\nA: " + gen("answer", max_tokens=100)
    print(response["answer"])
```

This is equivalent to prompt caching on cloud APIs, but implemented locally — critical for batch inference efficiency with repeated system prompts.

## Roles and Chat Templates

```python
from guidance import models, gen, system, user, assistant

lm = models.OpenAI("gpt-4o")

with system():
    lm += "You are a concise assistant that always answers in exactly three bullet points."

with user():
    lm += "What are the main benefits of containerization?"

with assistant():
    lm += gen("response", max_tokens=200)

print(lm["response"])
```

The `system()`, `user()`, and `assistant()` context managers automatically apply the correct chat template for the loaded model — ensuring correct formatting without manual template management.

## Recursive and Looping Programs

```python
from guidance import models, gen, select

lm = models.OpenAI("gpt-4o")

lm += "Generate a numbered list of 3 tips for writing clean code:\n"

tips = []
for i in range(1, 4):
    lm += f"{i}. "
    lm += gen(f"tip_{i}", max_tokens=30, stop="\n")
    lm += "\n"
    tips.append(lm[f"tip_{i}"])

print(tips)
```

The loop builds context incrementally — each iteration's generated tip becomes part of the context for the next, enabling coherent multi-item list generation with structural guarantees.

## Comparison with Other Structured Generation Libraries

| Feature | Guidance | Outlines | Instructor | LangChain |
| --- | --- | --- | --- | --- |
| Token-level constraints | Yes | Yes | No (post-parse) | No |
| Interleaved control flow | Yes | No | No | Partial |
| JSON schema enforcement | Yes | Yes | Yes (Pydantic) | Partial |
| Regex constraints | Yes | Yes | No | No |
| KV cache reuse | Yes (local) | No | No | No |
| Chat role management | Yes | Partial | No | Yes |
| Local model support | Yes (llama.cpp) | Yes | No | Yes |

## When to Use Guidance

Guidance is best suited for:

- **Structured extraction**: enforcing JSON, regex, or schema constraints at the token level — zero parsing failures
- **Decision trees**: branching generation logic based on model outputs
- **Multi-step programs**: complex pipelines where each step conditions on previous outputs
- **Local model efficiency**: KV cache reuse for batch inference with shared prefixes
- **Guaranteed output formats**: when downstream systems require exact formats and cannot tolerate retry logic

It is less suited for:

- Simple single-turn API calls without structural requirements (overhead not justified)
- Streaming applications (interleaved control requires synchronous execution)
- Cloud models without token-masking support (falls back to post-processing constraints)

## Summary

Microsoft Guidance reimagines LLM programming as interleaved generation and control: programs execute code and model generation in lockstep, with token-level constraints enforcing structural validity at every step. Its key capabilities — `select`, regex/JSON/grammar constraints, role management, and KV cache reuse — enable reliable, structured LLM outputs without fragile post-processing. For applications that require guaranteed output formats or complex conditional generation logic, Guidance provides a principled programming model that treats the LLM as a programmable component rather than an opaque service.
