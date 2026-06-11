---
title: Structured Outputs from LLMs
description: Techniques for getting LLMs to produce reliable, machine-readable structured outputs like JSON instead of free-form text.
---

Getting LLMs to return structured, machine-readable outputs (JSON, XML, typed objects) is essential for building reliable AI applications. Free-form text is fine for conversational responses, but downstream code needs predictable, parseable data.

## The Problem with Free-Form Output

If you ask an LLM to "return a JSON object with name and age," it might:
- Return valid JSON — but wrapped in markdown code fences.
- Omit required fields.
- Add extra fields not in the schema.
- Return text commentary before or after the JSON.
- Produce invalid JSON with trailing commas or missing quotes.

These inconsistencies break downstream parsing and require defensive code.

## Approaches to Structured Output

### Prompting
The simplest approach: explicitly specify the desired format in the prompt, provide an example, and ask the model not to include anything else. Unreliable without additional enforcement — use only as a fallback.

### JSON Mode
OpenAI, Anthropic, and Google APIs offer a JSON mode that constrains the model to always output valid JSON. This guarantees syntactic validity but not adherence to a specific schema.

### Structured Output / Function Calling
Provide a JSON Schema to the API. The model is constrained (via constrained decoding) to produce output that strictly matches the schema — correct field names, types, and required fields. Available in OpenAI (`response_format` with schema), Anthropic (tool use), and Google Gemini.

### Constrained Decoding (Local Models)
For locally hosted models (llama.cpp, vLLM, Ollama), libraries like **Outlines** and **llama.cpp grammar** enforce any context-free grammar — including JSON schemas — at the token level during generation. This is the most reliable method.

### Validation Libraries
**Pydantic** (Python) defines schemas as classes. Paired with libraries like **Instructor** or **Pydantic AI**, it automatically extracts and validates structured objects from LLM responses, retrying with error feedback if validation fails.

```python
from pydantic import BaseModel
import instructor

class Person(BaseModel):
    name: str
    age: int

client = instructor.from_openai(openai.OpenAI())
person = client.chat.completions.create(
    model="gpt-4o",
    response_model=Person,
    messages=[{"role": "user", "content": "John is 30 years old."}]
)
# person.name == "John", person.age == 30
```

## Best Practices

- Use the API's native structured output feature when available — it's the most reliable.
- Define tight schemas: mark only truly optional fields as optional.
- Validate and retry: if output fails schema validation, send the error back to the model and ask it to fix the output.
- Keep schemas simple — deeply nested structures increase error rates.
- Log raw outputs during development to catch schema violations early.
