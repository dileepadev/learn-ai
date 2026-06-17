---
title: Introduction to the Anthropic API
description: Getting started with Anthropic's API for Claude models — messages, vision, tool use, streaming, prompt caching, and best practices for building with Claude.
---

The Anthropic API provides programmatic access to Claude — Anthropic's family of large language models. It is clean, well-documented, and designed around a messages-based interface similar to OpenAI's Chat Completions API, with some unique features like prompt caching and a distinctive approach to tool use.

## Setup

```bash
pip install anthropic
```

Set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Basic Usage: Messages API

The core endpoint is `/v1/messages`. Every request requires a `model`, `max_tokens`, and a list of `messages`.

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain the attention mechanism in transformers."}
    ],
)

print(message.content[0].text)
```

### System Prompts
The system prompt is a separate top-level parameter, not a message with `"role": "system"`:

```python
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=512,
    system="You are a concise technical writer. Keep answers under 3 sentences.",
    messages=[{"role": "user", "content": "What is backpropagation?"}],
)
```

## Vision: Analyzing Images

Claude supports image inputs (JPEG, PNG, GIF, WebP) either as base64-encoded data or direct URLs (for supported models):

```python
import base64

with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
            {"type": "text", "text": "Describe this architecture diagram."},
        ],
    }],
)
```

## Tool Use (Function Calling)

Claude's tool use follows a turn-based pattern where the model requests a tool call, you execute it, and return the result.

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
}]

# First turn — model may call a tool
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)

# If Claude calls a tool, stop_reason == "tool_use"
if response.stop_reason == "tool_use":
    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = call_weather_api(tool_use.input["location"])  # your function

    # Second turn — provide the tool result
    final_response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result),
            }]},
        ],
    )
```

## Streaming

For responsive UIs, stream the response token by token:

```python
with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku about deep learning."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Prompt Caching

For large system prompts or documents reused across many requests, cache them to reduce cost and latency:

```python
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "<very long document>",
        "cache_control": {"type": "ephemeral"},
    }],
    messages=[{"role": "user", "content": "Summarize the key findings."}],
)
# Check cache usage
print(message.usage)  # Shows cache_creation_input_tokens, cache_read_input_tokens
```

Cached tokens are charged at ~10% of standard input token rates on subsequent hits.

## Model Selection

| Model | Best For |
|-------|----------|
| `claude-opus-4-5` | Complex reasoning, nuanced writing, analysis |
| `claude-sonnet-4-5` | Balanced performance and cost |
| `claude-haiku-3-5` | Speed and cost-sensitive workloads |

## Key Differences from OpenAI API

- **System prompt** is a top-level parameter, not a `system` role message.
- **Tool use** uses `tool_result` messages rather than `tool` role messages.
- **No `logprobs`:** Anthropic does not expose token probabilities.
- **`max_tokens` is required** (no default).
- **Prompt caching** via `cache_control` is explicit and detailed.
- **Content blocks:** Responses return a list of content blocks (`TextBlock`, `ToolUseBlock`) rather than a single string.

The Python SDK also supports async (`AsyncAnthropic`) for use in async frameworks like FastAPI.
