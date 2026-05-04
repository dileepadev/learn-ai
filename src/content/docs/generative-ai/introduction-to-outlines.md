---
title: "Introduction to Outlines: Structured Text Generation"
description: "A practical guide to Outlines, an open-source Python library for constrained structured text generation that ensures LLM outputs conform to JSON schemas, regex patterns, Pydantic models, and custom grammars."
---

## What Is Outlines?

**Outlines** is an open-source Python library developed by .txt (dottxt-ai) that enables **structured generation** from large language models. Instead of hoping the model produces well-formed JSON or valid code, Outlines mathematically guarantees that the output conforms to a specified structure — a JSON schema, regex pattern, Pydantic model, or context-free grammar — by constraining the token sampling process at each step.

Outlines works with most open-weight models (via `transformers`, `vLLM`, or `llama.cpp`) and is designed for production use cases where output reliability is non-negotiable.

---

## The Problem: LLM Outputs Are Unreliable

LLMs generate text token-by-token, sampling from a probability distribution over the vocabulary. Without constraints, nothing prevents the model from:
- Producing malformed JSON (missing brackets, unquoted strings).
- Hallucinating field names not in the target schema.
- Generating outputs that fail Pydantic validation.
- Wrapping JSON in markdown code fences when you need raw output.

Prompt-based approaches ("respond only with valid JSON") are unreliable — models break format rules, especially under distribution shift or when asked to follow complex schemas.

---

## How Outlines Works: Guided Generation

Outlines uses **logit biasing** to constrain generation at each token step:

1. Given a target structure (JSON schema, regex, grammar), Outlines computes a **finite-state automaton** (FSA) that accepts all valid strings in that structure.
2. At each generation step, Outlines determines which tokens in the vocabulary are valid next steps in the automaton given the current state.
3. A **mask** is applied to the logits: invalid tokens get $-\infty$ logit (zero probability), valid tokens are unaffected.
4. Sampling proceeds over the constrained distribution — the model still chooses freely among valid tokens.

This approach:
- Guarantees structural validity (by construction, not heuristically).
- Preserves the model's semantic choices within the constrained space.
- Adds minimal latency (the FSA state is updated in $O(1)$ per token).

---

## Installation and Setup

```bash
pip install outlines
# With transformers backend:
pip install outlines[transformers]
# With vLLM backend (for production serving):
pip install outlines[vllm]
```

---

## Core Usage Patterns

### Generating Structured JSON

Define a Pydantic model, and Outlines guarantees the output is a valid instance:

```python
import outlines
from pydantic import BaseModel
from typing import Literal

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

class MovieReview(BaseModel):
    title: str
    rating: float           # 0.0 to 10.0
    sentiment: Literal["positive", "negative", "mixed"]
    summary: str

generator = outlines.generate.json(model, MovieReview)

review = generator(
    "Review: 'Inception' is a mind-bending thriller with stunning visuals "
    "and a complex narrative. Rating: 9/10."
)

print(review)
# MovieReview(title='Inception', rating=9.0, sentiment='positive',
#             summary='A mind-bending thriller with stunning visuals.')
print(type(review))  # <class 'MovieReview'>
```

The output is a native Python `MovieReview` instance — not a string that needs parsing.

### Generating with JSON Schema Directly

If you don't use Pydantic, pass a JSON schema dict or string:

```python
import json
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "email": {"type": "string", "format": "email"},
        "is_active": {"type": "boolean"}
    },
    "required": ["name", "age", "is_active"]
}

generator = outlines.generate.json(model, json.dumps(schema))
result = generator("Extract user info: John Doe, 34 years old, active user.")
print(result)  # {"name": "John Doe", "age": 34, "is_active": true}
```

### Regex-Constrained Generation

Force the model to produce output matching a specific regex pattern:

```python
import outlines

model = outlines.models.transformers("meta-llama/Meta-Llama-3-8B-Instruct")

# Extract phone numbers in a specific format
phone_pattern = r"\+1-\d{3}-\d{3}-\d{4}"
generator = outlines.generate.regex(model, phone_pattern)

phone = generator("What is the company's support number?")
print(phone)  # "+1-800-555-1234" (guaranteed to match the regex)
```

Regex constraints are useful for:
- Phone numbers, zip codes, dates in specific formats
- Product codes, identifiers, serial numbers
- Simple yes/no or true/false extraction

### Constrained Choice (Classification)

Force the model to choose from a fixed set of options:

```python
import outlines

model = outlines.models.transformers("HuggingFaceH4/zephyr-7b-beta")

generator = outlines.generate.choice(model, ["positive", "negative", "neutral"])

sentiment = generator(
    "Classify the sentiment of: 'The product exceeded my expectations!'"
)
print(sentiment)  # "positive" (guaranteed — no postprocessing needed)
```

This is equivalent to a multi-class classifier with LLM reasoning, but with guaranteed valid output.

### Integer and Float Generation

```python
import outlines

model = outlines.models.transformers("Qwen/Qwen2-7B-Instruct")

int_gen = outlines.generate.format(model, int)
rating = int_gen("Rate this product from 1 to 10: 'Amazing quality, fast shipping.'")
print(rating)  # 9 (a Python int, guaranteed)

float_gen = outlines.generate.format(model, float)
score = float_gen("Give a confidence score between 0 and 1 for this classification.")
print(score)  # 0.87 (a Python float)
```

---

## Context-Free Grammar Generation

For complex structured outputs (SQL, Python code, formal languages), Outlines supports context-free grammars via the EBNF (Extended Backus-Naur Form) notation:

```python
import outlines

arithmetic_grammar = """
    ?start: expr
    ?expr: term ("+" term)*
    ?term: factor ("*" factor)*
    ?factor: NUMBER | "(" expr ")"
    NUMBER: /[0-9]+/
    %ignore /\s+/
"""

model = outlines.models.transformers("meta-llama/Meta-Llama-3-8B-Instruct")
generator = outlines.generate.cfg(model, arithmetic_grammar)

expression = generator("Write a valid arithmetic expression: 3 plus 4 times 2")
print(expression)  # "(3+4)*2" or "3+4*2" — always valid arithmetic
```

Grammar-constrained generation is particularly valuable for:
- Generating valid SQL queries
- Producing syntactically correct code in any language
- Generating structured logical formulas
- Creating markup in specific dialects (LaTeX, custom DSLs)

---

## Integration with vLLM for Production

For high-throughput production serving, Outlines integrates with vLLM:

```python
from vllm import LLM, SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from pydantic import BaseModel

class OrderItem(BaseModel):
    product_id: str
    quantity: int
    unit_price: float

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
logits_processor = JSONLogitsProcessor(OrderItem, llm)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    logits_processors=[logits_processor]
)

outputs = llm.generate(
    ["Extract order: 3 units of SKU-12345 at $29.99 each."],
    sampling_params
)
print(outputs[0].outputs[0].text)
```

vLLM with Outlines enables concurrent structured generation at thousands of tokens/second with guaranteed output validity.

---

## Comparison with Alternatives

| Approach | Validity Guarantee | Latency Overhead | Schema Support | Model Compatibility |
|----------|-------------------|-----------------|----------------|---------------------|
| Prompt engineering | None | None | N/A | Any |
| Instructor (retry-based) | Probabilistic | High (retries) | Pydantic | API models |
| LangChain output parsers | Probabilistic | Low | Pydantic, custom | Any |
| **Outlines (guided)** | **Mathematical** | **Minimal** | **Schema, regex, CFG** | **Open-weight models** |
| LMQL | Mathematical | Moderate | Custom query lang | Some |
| Guidance | Mathematical | Low | Custom templates | Open-weight models |

---

## Advanced: Batched Structured Generation

```python
import outlines
from pydantic import BaseModel
from typing import List

model = outlines.models.transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    device="cuda"
)

class Entity(BaseModel):
    name: str
    entity_type: str  # PERSON, ORG, LOCATION, etc.
    confidence: float

generator = outlines.generate.json(model, Entity)

# Process multiple inputs in a batch
prompts = [
    "Extract the main entity: 'Apple announced new iPhone models in California.'",
    "Extract the main entity: 'Elon Musk visited the Tesla factory in Texas.'",
    "Extract the main entity: 'The Amazon River flows through Brazil.'",
]

results = generator(prompts)  # Batched inference
for entity in results:
    print(f"{entity.name} ({entity.entity_type}): {entity.confidence:.2f}")
```

---

## Streaming Structured Output

Outlines supports streaming tokens while still enforcing structure:

```python
import outlines
from pydantic import BaseModel

model = outlines.models.transformers("HuggingFaceH4/zephyr-7b-beta")

class ReportSection(BaseModel):
    title: str
    content: str
    word_count: int

generator = outlines.generate.json(model, ReportSection)

# Stream token-by-token while maintaining structural validity
for token in generator.stream("Generate a summary section for a Q3 earnings report."):
    print(token, end="", flush=True)
```

---

## Practical Tips

**Use the smallest schema that captures your needs.** Complex deeply-nested schemas increase the FSA complexity and can slow generation. Flatten schemas where possible.

**Combine with prompt engineering.** Structured generation guarantees format but not content quality. Use good prompts to ensure the model fills fields with meaningful values.

**Test schema coverage.** Ensure your schema allows all valid outputs you want. Overly restrictive schemas (e.g., enums missing valid options) cause the model to produce the nearest valid alternative, which may be wrong.

**Use `temperature > 0` for creative fields.** For string fields in a schema, moderate temperature ensures diverse, natural outputs. For constrained fields (integers, enums), temperature has less effect since only valid tokens are sampled.

**Cache the compiled FSA.** Outlines pre-compiles the FSA for each schema. Reuse the `generator` object across calls — don't recreate it for each request.

---

## Summary

Outlines transforms structured LLM generation from a probabilistic hope into a mathematical guarantee. By constraining token sampling to match JSON schemas, regex patterns, Pydantic models, and context-free grammars, it eliminates a major class of production failures in LLM-powered applications. The guided generation approach adds minimal overhead, works with any open-weight model, and integrates cleanly with high-throughput serving infrastructure like vLLM. For any application where output format correctness is non-negotiable — data extraction, API response generation, classification, code generation — Outlines is an essential tool.
