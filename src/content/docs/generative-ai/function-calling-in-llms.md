---
title: Function Calling in Large Language Models
description: Learn how LLMs use function calling (tool use) to interact with external APIs, databases, and code — enabling structured, actionable AI responses.
---

Function calling — also called **tool use** — is a capability that allows Large Language Models (LLMs) to produce structured outputs that invoke external functions, APIs, or services rather than generating plain text. It bridges the gap between language understanding and real-world action.

## Why Function Calling Matters

Standard LLMs generate freeform text. While impressive, this is limiting when you need:

- A precise database query result.
- A live weather reading.
- A calculation beyond the model's arithmetic accuracy.
- A write operation to an external system.

Function calling teaches the LLM to signal *when* and *how* to call a tool, and what arguments to provide — leaving the actual execution to the application layer.

## How It Works

The general pattern is:

1. **Define functions** — Provide the LLM with a schema (usually JSON Schema) describing available tools: their name, description, and parameters.
2. **Model decides to call** — Based on the user's message, the model outputs a structured function call JSON instead of (or alongside) a text response.
3. **Application executes** — The application extracts the function name and arguments, executes the actual function, and captures the result.
4. **Model continues** — The result is passed back to the model, which incorporates it into its final response.

### OpenAI-Style Example

```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Tokyo'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city"]
            }
        }
    }
]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "get_current_weather"
print(tool_call.function.arguments)  # '{"city": "Tokyo"}'
```

## Parallel Function Calling

Modern LLMs can invoke **multiple functions in a single turn** when queries require it. For example, "What's the weather in Paris and Berlin?" may trigger two simultaneous weather calls, reducing round-trip latency.

## Structured Output vs. Function Calling

| Feature | Structured Output | Function Calling |
|---|---|---|
| Goal | Force a specific JSON shape | Trigger an external action |
| Execution | No execution — just formatting | Application executes the call |
| Use case | Extracting info from text | Agents, workflows, live data |

The two are closely related: both instruct the model to produce structured JSON, but function calling carries the semantic meaning of *intention to act*.

## Tool Choice Strategies

Most APIs expose a `tool_choice` parameter:

- **`auto`** — The model decides whether to call a function or respond with text.
- **`required`** — The model must call one of the provided functions.
- **`none`** — The model must respond with text, ignoring tools.
- **Specific function** — Force a specific function to be called.

## Multi-Turn Tool Use

In agentic systems, function calling is often **multi-turn**: the model calls a tool, receives the result, may call another tool based on the result, and continues until the task is complete. This is the foundation of frameworks like LangChain, AutoGen, and the OpenAI Assistants API.

```
User → Model → Tool Call → Execute → Result → Model → Final Answer
```

## Best Practices

- **Write precise descriptions** — The model routes tool selection based on the description, not the name. Be specific about when to use each tool.
- **Validate arguments server-side** — Never trust model-generated arguments blindly; validate and sanitize before execution.
- **Handle tool errors gracefully** — Return errors back to the model as tool results so it can adapt its response.
- **Limit tool count** — Too many tools can confuse the model. Group related operations or prune rarely-used tools.
- **Log all tool calls** — For debugging and auditing, always log the function name, arguments, and result.

## Security Considerations

Function calling introduces a **privileged execution channel**. Key risks include:

- **Prompt injection** — Malicious content in retrieved documents could trick the model into calling unintended functions.
- **Over-permissioning** — Tools should adhere to least-privilege principles; don't expose write/delete endpoints unless necessary.
- **Argument injection** — Validate all arguments to prevent injection attacks in downstream systems (SQL, shell commands, etc.).

## Real-World Applications

- **AI assistants** — Booking calendars, sending emails, reading CRM data.
- **Data analytics** — Querying databases based on natural language.
- **Code execution** — Running code snippets in a sandbox for math or data analysis.
- **IoT control** — Issuing commands to smart devices.
- **RAG pipelines** — Calling a vector search tool to retrieve relevant documents before answering.

Function calling is foundational to the modern **agentic AI** paradigm — it is what transforms an LLM from a text predictor into an action-capable system embedded in real-world workflows.
