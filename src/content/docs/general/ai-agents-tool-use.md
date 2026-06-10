---
title: Tool Use in AI Agents
description: How AI agents use tools — APIs, code interpreters, search engines, and more — to act on the world and extend LLM capabilities.
---

Tool use is the ability of an AI agent to invoke external functions or services to gather information, perform computations, or take actions. It transforms a language model from a text generator into an active system that can browse the web, run code, query databases, send emails, and more.

## Why Agents Need Tools

LLMs have limitations that tools directly address:
- **Knowledge cutoff:** Web search provides up-to-date information.
- **Cannot compute reliably:** A code interpreter executes math and logic accurately.
- **No memory by default:** Database tools enable persistent storage and retrieval.
- **Cannot act:** API calls allow agents to create calendar events, file tickets, send messages.

## How Tool Use Works

The agent follows a loop:

1. **Reason:** The LLM decides whether a tool is needed and which one.
2. **Call:** The agent formats a tool call with the correct arguments.
3. **Execute:** The tool runs and returns a result.
4. **Observe:** The result is added back to the context.
5. **Continue:** The LLM decides to call another tool or produce a final response.

This is the **ReAct** pattern (Reason + Act), the foundation of most agent frameworks.

## Common Tool Categories

- **Search:** Web search (Bing, Tavily), document search, vector database lookup.
- **Code execution:** Python interpreter, shell, SQL runner — for reliable computation and data analysis.
- **APIs:** Weather, maps, calendars, CRM, ticketing systems, payment processors.
- **File operations:** Read/write files, parse PDFs, process spreadsheets.
- **Browser:** Navigate web pages, fill forms, click buttons (computer use).
- **Memory:** Store and retrieve facts, conversation history, user preferences.

## Defining Tools

In most frameworks, a tool is a function with a name, description, and JSON schema for its parameters. The LLM uses the description to decide when and how to invoke it.

```python
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # call weather API...
    return f"22°C, sunny in {city}"
```

## Reliability Challenges

- **Wrong tool selection:** The agent picks the wrong tool or skips tools it should use.
- **Argument errors:** Malformed tool inputs cause failures.
- **Error handling:** Tools fail; agents must recognize and recover from errors.
- **Tool overuse:** Agents that call tools when they don't need to, wasting latency and cost.

Good tool descriptions and few-shot examples in the system prompt significantly improve reliability.

## Frameworks with Tool Use

LangChain, LlamaIndex, LangGraph, CrewAI, AutoGen, and Pydantic AI all provide tool abstractions. OpenAI's function calling and Anthropic's tool use are the underlying APIs most frameworks build on.
