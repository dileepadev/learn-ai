---
title: Introduction to the OpenAI API
description: A practical guide to getting started with the OpenAI API — completions, chat, embeddings, function calling, and best practices.
---

The OpenAI API provides programmatic access to OpenAI's models including GPT-4o, o1, and o3. It is the most widely used LLM API and the reference point for how most AI SDKs and frameworks are designed.

## Core Endpoints

### Chat Completions
The primary endpoint for conversational and instruction-following tasks. Accepts a list of messages with roles (`system`, `user`, `assistant`) and returns a completion.

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
```

### Embeddings
Converts text into dense vector representations for semantic search, clustering, and RAG pipelines.

```python
embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox"
).data[0].embedding
```

### Structured Outputs
Constrain the model to return output matching a JSON Schema — the most reliable way to get structured data.

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John is 30."}],
    response_format=MyPydanticModel
)
```

## Tool / Function Calling
Define functions the model can call. The model returns a structured tool call when it needs external information.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }
}]
```

## Models Overview

| Model | Best For |
|---|---|
| gpt-4o | Fast, capable, multimodal default |
| gpt-4o-mini | Cost-efficient for simple tasks |
| o3 / o4-mini | Complex reasoning, math, coding |
| text-embedding-3-small | Fast, cheap embeddings |

## Key Parameters

- **temperature (0–2):** Controls randomness. 0 for deterministic, 1 for default, higher for creative tasks.
- **max_tokens:** Limits response length.
- **top_p:** Nucleus sampling — an alternative to temperature.
- **stream:** Set to `True` for streaming token-by-token output.

## Best Practices

- Always set a `system` prompt to define the model's behavior and persona.
- Use structured outputs or function calling whenever you need parseable data.
- Implement retry logic with exponential backoff for rate limit errors.
- Log prompts and responses during development for debugging.
- Monitor token usage — cost scales linearly with tokens.
