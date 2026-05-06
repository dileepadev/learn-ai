---
title: Introduction to LMQL
description: A practical guide to LMQL (Language Model Query Language), a declarative programming language for LLMs that enables constraint-guided decoding, typed outputs, nested queries, and efficient batched inference with any local or API-based model.
---

# Introduction to LMQL

**LMQL** (Language Model Query Language) is an open-source programming language and runtime for interacting with large language models. It combines a Python-like syntax with **constraint-guided decoding** — expressions that constrain what tokens the model can generate, enforcing structure, types, and logical conditions during generation rather than post-processing. LMQL compiles queries into efficient token-masked beam searches, reducing wasted inference and improving reliability.

## Why LMQL?

Standard LLM APIs generate free-form text that must be parsed and validated after generation. This fails silently when the model doesn't follow instructions. LMQL solves this by:

- **Enforcing constraints at decode time**: the model physically cannot produce invalid tokens
- **Type-safe outputs**: `INT`, `FLOAT`, `BOOL`, regex patterns, enumerated values
- **Nested structure**: sub-queries, branching, loops within a single model interaction
- **Efficiency**: constraint masks prune the beam, reducing generations needed for valid outputs

## Installation

```bash
pip install lmql

# For local models (llama.cpp backend)
pip install lmql[hf]
```

## Basic Query Syntax

LMQL queries are Python functions decorated with `@lmql.query`:

```python
import lmql

@lmql.query
async def classify_sentiment(review: str):
    '''lmql
    "Classify the sentiment of this review as positive, negative, or neutral.\n"
    "Review: {review}\n"
    "Sentiment: [SENTIMENT]" where SENTIMENT in ["positive", "negative", "neutral"]
    return SENTIMENT
    '''

# Run
import asyncio
result = asyncio.run(classify_sentiment("The product exceeded all my expectations!"))
print(result)  # "positive"
```

The `where` clause constrains `SENTIMENT` to one of three values — the decoder applies a mask that zeros out all other token logits.

## Typed Outputs

LMQL provides built-in type constraints:

```python
@lmql.query
async def extract_structured(text: str):
    '''lmql
    "Extract the following from this text: {text}\n\n"
    "Name: [NAME]\n"           where len(TOKENS(NAME)) < 10
    "Age: [AGE]\n"             where INT(AGE) and int(AGE) in range(0, 150)
    "Confidence (0-1): [CONF]" where FLOAT(CONF) and float(CONF) >= 0.0 and float(CONF) <= 1.0
    return {"name": NAME, "age": AGE, "confidence": CONF}
    '''

result = asyncio.run(extract_structured("Dr. Sarah Chen is 42 years old."))
print(result)  # {"name": "Dr. Sarah Chen", "age": "42", "confidence": "0.97"}
```

## Regex Constraints

Regular expressions enforce format constraints on generated text:

```python
@lmql.query
async def extract_date(text: str):
    '''lmql
    "Extract the date mentioned in: {text}\n"
    "Date (YYYY-MM-DD): [DATE]"
    where REGEX(DATE, r"\d{4}-\d{2}-\d{2}")
    return DATE
    '''

@lmql.query
async def extract_email(text: str):
    '''lmql
    "What is the email address in: {text}?\n"
    "Email: [EMAIL]"
    where REGEX(EMAIL, r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    return EMAIL
    '''
```

## Multi-Variable Queries

LMQL can generate multiple constrained variables in a single forward pass:

```python
@lmql.query
async def analyze_code(code: str):
    '''lmql
    "Analyze this Python code snippet:\n```python\n{code}\n```\n\n"
    "Has bugs: [HAS_BUGS]"         where HAS_BUGS in ["yes", "no"]
    " | Complexity: [COMPLEXITY]"  where COMPLEXITY in ["low", "medium", "high"]
    " | Lines: [LINES]"            where INT(LINES)
    "\n\nExplanation: [EXPLANATION]" where len(TOKENS(EXPLANATION)) in range(10, 100)
    return {
        "has_bugs": HAS_BUGS == "yes",
        "complexity": COMPLEXITY,
        "lines": int(LINES),
        "explanation": EXPLANATION,
    }
    '''

result = asyncio.run(analyze_code("for i in range(10):\n    print(i)"))
```

## Branching and Control Flow

LMQL supports Python control flow within queries:

```python
@lmql.query
async def multi_step_qa(question: str, use_chain_of_thought: bool = True):
    '''lmql
    "Q: {question}\n"

    if use_chain_of_thought:
        "Let me think step by step:\n[REASONING]\n"
        where len(TOKENS(REASONING)) in range(20, 200)

    "Final answer: [ANSWER]"
    where len(TOKENS(ANSWER)) < 50

    return ANSWER
    '''
```

## Nested Queries and Composition

Queries can call other queries:

```python
@lmql.query
async def translate(text: str, target_lang: str):
    '''lmql
    "Translate to {target_lang}: {text}\n"
    "Translation: [RESULT]"
    return RESULT
    '''

@lmql.query
async def multilingual_pipeline(texts: list):
    '''lmql
    results = []
    for text in texts:
        fr = await lmql.call(translate, text, "French")
        de = await lmql.call(translate, text, "German")
        results.append({"fr": fr, "de": de})
    return results
    '''
```

## Connecting to Models

LMQL supports multiple backends:

```python
# OpenAI API
lmql.set_default_model("gpt-4o-mini")

# Local HuggingFace model
lmql.set_default_model("local:meta-llama/Llama-3.2-3B-Instruct")

# llama.cpp (GGUF)
lmql.set_default_model("llama.cpp:/path/to/model.gguf")

# Anthropic
lmql.set_default_model("anthropic/claude-3-haiku-20240307")
```

## Comparison with Similar Tools

| Feature | LMQL | Guidance | Outlines | DSPy |
|---|---|---|---|---|
| Constraint type | Declarative `where` | Template interleaving | Regex / JSON schema | Input/output signatures |
| Decoding control | Token masking at decode | Stop/continue signals | Logit bias | Post-generation |
| Typed outputs | ✅ INT, FLOAT, BOOL | Partial | ✅ JSON schema | ❌ |
| Control flow | ✅ Full Python | ✅ | Limited | ✅ |
| Batching | ✅ Async batching | Manual | ✅ | ✅ |
| Local model support | ✅ llama.cpp, HF | ✅ | ✅ | ✅ |

## Async Batching for Efficiency

LMQL's runtime batches multiple queries together for efficient GPU utilization:

```python
import asyncio
import lmql

reviews = [
    "Great product, highly recommend!",
    "Terrible experience, never again.",
    "It's okay, nothing special.",
]

@lmql.query
async def classify(review: str):
    '''lmql
    "Sentiment of '{review}': [LABEL]" where LABEL in ["positive", "negative", "neutral"]
    return LABEL
    '''

# All 3 queries batched into a single model forward pass
results = asyncio.run(asyncio.gather(*[classify(r) for r in reviews]))
print(results)  # ["positive", "negative", "neutral"]
```

## Summary

LMQL brings the rigor of typed, constraint-based programming to LLM interactions. By enforcing output constraints at token generation time rather than post-hoc parsing, it eliminates invalid outputs, reduces retry overhead, and enables reliable structured data extraction. Its Python-native syntax, async batching, and multi-backend support make it practical for both research prototyping and production pipelines where structured, type-safe LLM outputs are required. For use cases demanding guaranteed output format — information extraction, form filling, structured reasoning — LMQL provides stronger guarantees than prompt engineering alone.
