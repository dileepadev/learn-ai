---
title: Introduction to the OpenAI Agents SDK
description: Learn the core primitives of the OpenAI Agents SDK, including agents, tools, handoffs, guardrails, and tracing, for building multi-step agentic applications.
---

The OpenAI Agents SDK is a lightweight Python framework for building agentic applications. It deliberately exposes a small set of primitives instead of a large abstraction layer.

## Core Primitives

- **Agent**: a model configured with instructions and a set of tools
- **Tools**: Python functions, hosted tools (web search, file search), or other agents
- **Handoffs**: delegation from one agent to another specialized agent
- **Guardrails**: input and output validation that can halt a run
- **Tracing**: automatic recording of model calls, tool calls, and handoffs

## A Minimal Agent

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_order_status(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id} is out for delivery."

agent = Agent(
    name="Support",
    instructions="Help customers with order questions. Use tools when needed.",
    tools=[get_order_status],
)

result = Runner.run_sync(agent, "Where is order 4821?")
print(result.final_output)
```

The runner executes a loop: call the model, run any requested tools, feed results back, and repeat until the agent produces a final answer.

## Handoffs for Specialization

```python
billing = Agent(name="Billing", instructions="Handle billing questions.")
triage = Agent(
    name="Triage",
    instructions="Route the user to the right specialist.",
    handoffs=[billing],
)
```

## Guardrails and Tracing

Guardrails run checks (for example, off-topic or unsafe input detection) in parallel with the main agent. Traces make it possible to debug why an agent chose a tool or handoff.

## Design Advice

Prefer a few narrowly scoped agents with clear instructions over one agent with dozens of tools; tool selection accuracy drops as the toolset grows.
