---
title: "Structured Output Generation with LLMs"
description: "Learn how to reliably extract structured data (JSON, XML, typed objects) from LLMs using constrained decoding, function calling, and schema-guided generation."
---

Getting an LLM to return free-form text is easy. Getting it to return valid, schema-conforming JSON every single time is a different challenge. **Structured output generation** is the set of techniques that make LLM outputs reliably parseable and typed.

## Why Free-Form Output Fails in Production

When you prompt an LLM to "respond in JSON," it usually does — but not always. Common failure modes include:

- Extra prose before or after the JSON block.
- Missing required fields.
- Wrong data types (a number returned as a string).
- Truncated output when the response is long.

For any production pipeline that parses LLM output, these failures cause downstream errors.

## Approach 1: Prompt Engineering

The simplest approach is careful prompting: provide a JSON schema in the system prompt, show few-shot examples, and instruct the model to output only JSON. This works reasonably well with capable models but has no hard guarantees.

## Approach 2: Function Calling / Tool Use

OpenAI, Anthropic, and Google all support structured output via function/tool definitions. You define a JSON schema for the expected output, and the model is fine-tuned to fill it. The API enforces the schema at the output level.

```json
{
  "name": "extract_person",
  "parameters": {
    "type": "object",
    "properties": {
      "name": { "type": "string" },
      "age": { "type": "integer" },
      "email": { "type": "string" }
    },
    "required": ["name", "age"]
  }
}
```

## Approach 3: Constrained Decoding

The most robust approach modifies the decoding process itself. At each token step, a grammar or schema is used to **mask out invalid tokens** — only tokens that could continue a valid output are allowed.

Libraries like **Outlines**, **Guidance**, and **LMQL** implement this. The model never has the option to produce invalid output because invalid tokens are assigned zero probability.

## Approach 4: Instructor and Pydantic

The **Instructor** library wraps LLM APIs and uses Pydantic models to define the expected output schema. It handles retries, validation, and re-prompting automatically when the model returns invalid output.

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

class Person(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-4o",
    response_model=Person,
    messages=[{"role": "user", "content": "John is 30 years old."}]
)
# person.name == "John", person.age == 30
```

## Choosing the Right Approach

| Approach | Reliability | Flexibility | Latency |
|---|---|---|---|
| Prompt engineering | Low | High | None |
| Function calling | High | Medium | Low |
| Constrained decoding | Very High | Medium | Low |
| Instructor + retries | High | High | Medium |

For most production use cases, function calling or constrained decoding is the right choice. Prompt engineering alone is rarely sufficient.
