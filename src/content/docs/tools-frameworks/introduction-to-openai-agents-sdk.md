---
title: "Introduction to the OpenAI Agents SDK"
description: "Get started with the OpenAI Agents SDK — a lightweight Python framework for building multi-agent workflows with tools, handoffs, and built-in tracing."
---

The **OpenAI Agents SDK** (formerly Swarm) is an open-source Python framework for building multi-agent systems. It provides a minimal but powerful set of primitives: agents, tools, and handoffs — enough to build sophisticated workflows without the overhead of heavier frameworks.

## Core Concepts

### Agents
An agent is an LLM configured with a system prompt and a set of tools. It runs in a loop: receive input, decide whether to call a tool or respond, execute tools, observe results, repeat.

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
)

result = Runner.run_sync(agent, "What is the capital of France?")
print(result.final_output)
```

### Tools
Tools are Python functions decorated with `@function_tool`. The SDK automatically generates the JSON schema from the function signature and docstring.

```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Call a weather API
    return f"Sunny, 22°C in {city}"
```

### Handoffs
Handoffs allow one agent to transfer control to another. This is the primary mechanism for building multi-agent workflows.

```python
from agents import Agent

triage_agent = Agent(
    name="Triage",
    instructions="Route the user to the right specialist.",
    handoffs=[billing_agent, technical_agent],
)
```

When the triage agent decides to hand off, the target agent takes over the conversation with full context.

## The Agent Loop

The SDK manages the agent loop automatically:

1. Call the LLM with the current messages and available tools.
2. If the response includes tool calls, execute them and append results.
3. If the response includes a handoff, switch to the target agent.
4. If the response is a final message, return it.

## Guardrails

The SDK supports input and output guardrails — validators that run before the agent processes input or after it produces output. Guardrails can reject inputs, modify outputs, or trigger tripwires that halt execution.

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def check_safe_input(ctx, agent, input):
    # Check if input is appropriate
    is_safe = await run_safety_check(input)
    return GuardrailFunctionOutput(output_info=input, tripwire_triggered=not is_safe)
```

## Tracing

Every agent run is automatically traced. The SDK integrates with the OpenAI platform's tracing UI, showing the full execution graph: which agents ran, which tools were called, what was handed off, and how long each step took.

## When to Use It

The Agents SDK is a good fit when:
- You want a lightweight framework without heavy abstractions.
- You're building on OpenAI models and want native integration.
- You need multi-agent handoffs with minimal boilerplate.

For more complex orchestration (conditional branching, cycles, state machines), LangGraph or a custom solution may be more appropriate.
