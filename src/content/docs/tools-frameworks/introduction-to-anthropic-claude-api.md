---
title: "Introduction to the Anthropic Claude API"
description: "Learn how to use the Anthropic Claude API — covering messages, vision, tool use, extended thinking, and prompt caching for building production AI applications."
---

The **Anthropic Claude API** provides access to the Claude family of models. Claude is known for strong reasoning, long context handling, and safety-focused design. This guide covers the key features you need to build production applications.

## Getting Started

Install the SDK and make your first call:

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum entanglement simply."}
    ]
)
print(message.content[0].text)
```

## The Messages API

Claude uses a messages format with alternating `user` and `assistant` turns. The system prompt is a separate parameter:

```python
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=2048,
    system="You are an expert Python developer. Be concise.",
    messages=[
        {"role": "user", "content": "How do I reverse a list in Python?"},
        {"role": "assistant", "content": "Use `my_list[::-1]` for a new reversed list, or `my_list.reverse()` to reverse in place."},
        {"role": "user", "content": "What's the difference in performance?"}
    ]
)
```

## Vision

Claude can analyze images passed as base64 or URLs:

```python
import base64

with open("chart.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
            {"type": "text", "text": "What trend does this chart show?"}
        ]
    }]
)
```

## Tool Use

Define tools with JSON schemas and Claude will call them when appropriate:

```python
tools = [{
    "name": "get_stock_price",
    "description": "Get the current stock price for a ticker symbol.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["ticker"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's Apple's stock price?"}]
)
```

## Extended Thinking

Claude's extended thinking mode lets the model reason through complex problems before responding:

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Solve this logic puzzle: ..."}]
)

# Access the thinking trace
for block in response.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Answer:", block.text)
```

## Prompt Caching

For prompts with large, repeated context (system prompts, documents), prompt caching can reduce costs by up to 90% and latency by up to 85%:

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": very_long_system_prompt,
        "cache_control": {"type": "ephemeral"}  # Cache this content
    }],
    messages=[{"role": "user", "content": "Question about the document..."}]
)
```

## Model Selection

| Model | Best For | Context |
|---|---|---|
| claude-opus-4-5 | Complex reasoning, analysis | 200K |
| claude-sonnet-4-5 | Balanced performance/cost | 200K |
| claude-haiku-3-5 | Fast, lightweight tasks | 200K |

Choose Sonnet for most production workloads — it offers the best balance of capability and cost.
