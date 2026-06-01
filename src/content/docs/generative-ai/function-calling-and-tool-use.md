---
title: "Function Calling and Tool Use in LLMs"
description: "Learn how LLMs call functions and use tools — from simple JSON function calls to complex tool orchestration, and how to design tools that models can invoke reliably."
---

Function calling transforms LLMs from text generators into interactive agents that can take real-world actions. When you need the model to look up real-time data, call an API, or manipulate external systems, function calling is the bridge.

## How Function Calling Works

Function calling has two components:

1. **Tool definition**: The developer describes available functions and their parameters.
2. **Tool invocation**: The model recognizes when to call a function, extracts the arguments, and the system executes it.

The model never executes code itself — it only decides *what* to call and *with what arguments*.

## API-Style Function Calling

OpenAI and Anthropic define functions via JSON schemas:

```python
functions = [{
    "name": "get_stock_price",
    "description": "Get the current stock price for a ticker symbol",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol, e.g., AAPL, GOOGL"
            }
        },
        "required": ["ticker"]
    }
}]
```

When the model decides to call the function, it returns a structured response:

```json
{
    "function_calls": [{
        "name": "get_stock_price",
        "arguments": {"ticker": "NVDA"}
    }]
}
```

The application executes the function and returns the result:

```json
{
    "function_results": [{
        "name": "get_stock_price",
        "result": {"price": 875.28, "currency": "USD"}
    }]
}
```

## Designing Effective Tools

### Tool Naming and Description
The model's ability to invoke the correct tool depends entirely on how well you describe it. Best practices:

```python
# Good: Clear, specific description
{
    "name": "search_documents",
    "description": "Search the internal knowledge base for technical documentation. "
                  "Use for questions about API usage, error messages, or configuration.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords or natural language question"},
            "filters": {"type": "object", "description": "Optional metadata filters"}
        }
    }
}

# Bad: Vague or missing context
{
    "name": "search",
    "description": "Search something"
}
```

### Parameter Design
Parameters should be:
- **Atomic**: Don't overload a single parameter with multiple concepts.
- **Typed**: Use JSON Schema types to constrain valid inputs.
- **Optional when possible**: Require only what's essential; let the model choose optional parameters.

## Tool Use Patterns

### Single Tool Call
The simplest pattern: model receives user request, calls a tool, receives result, generates response.

### Parallel Tool Calls
For independent operations, the model can call multiple tools at once:

```python
{
    "function_calls": [
        {"name": "get_weather", "arguments": {"city": "San Francisco"}},
        {"name": "get_stock_price", "arguments": {"ticker": "GOOGL"}},
        {"name": "search_calender", "arguments": {"date": "2025-06-04"}}
    ]
}
```

This is faster when tools are independent and can execute in parallel.

### Sequential Tool Calls
When the result of one tool is needed to call another:

```
User: "What's the weather in the capital of France?"
  ↓
Model: Calls get_capital(country="France") → Returns "Paris"
  ↓
Model: Calls get_weather(city="Paris") → Returns 22°C
  ↓
Model: "The current weather in Paris is 22°C."
```

This requires the model to chain tool calls correctly. Some APIs handle this automatically; others require multiple round trips.

### Tool Selection with Routing
When multiple tools are available, the model must select the right one:

```
User: "What was Q4 revenue?"
  ↓
Tool selection: sales_query_tool (not: weather_tool, calendar_tool, email_tool)
  ↓
Call: sales_query_tool(query="Q4 revenue")
  ↓
Return results
```

## Tool Execution Considerations

### Execution Environment
Tool execution happens outside the LLM. Design considerations:
- **Sandboxing**: Execute untrusted code in isolated containers.
- **Timeouts**: Set reasonable timeouts to prevent hangs.
- **Rate limits**: Respect API rate limits; queue or reject excess calls.
- **Error handling**: Return structured errors the model can interpret and respond to.

### Handling Errors
```python
def execute_tool(name: str, arguments: dict) -> dict:
    try:
        result = call_external_api(name, arguments)
        return {"status": "success", "data": result}
    except TimeoutError:
        return {"status": "error", "message": "Request timed out", "retryable": True}
    except PermissionError:
        return {"status": "error", "message": "Permission denied", "retryable": False}
    except Exception as e:
        return {"status": "error", "message": str(e), "retryable": False}
```

The model needs to know whether errors are retryable to handle them appropriately.

### Latency
Tool calls add latency to the overall response. Mitigate with:
- **Parallel execution** when possible.
- **Streaming partial results** before tool completion.
- **Prefetching** common tool calls based on context.

## Multi-Step Tool Workflows

For complex tasks, design workflows the model can navigate:

```python
workflow = {
    "name": "order_troubleshooting",
    "steps": [
        {"tool": "lookup_order", "description": "Get order status"},
        {"tool": "get_shipping_info", "description": "Track package if shipped"},
        {"tool": "initiate_return", "description": "Start return process if requested"},
        {"tool": "schedule_support", "description": "Escalate to human if needed"}
    ]
}
```

The model can then guide users through multi-step processes, calling tools as needed at each step.

## Tool Safety and Rate Limiting

- **Cost controls**: Set per-request or per-session spending limits.
- **Permission boundaries**: Restrict available tools per user role.
- **Audit logging**: Log all tool calls for security review.
- **Human-in-the-loop**: Require approval for sensitive operations (payments, data deletion).

Function calling is the foundation of agentic AI. Designing clear, reliable tools — and building robust execution infrastructure around them — is essential for building systems that can take real-world action.