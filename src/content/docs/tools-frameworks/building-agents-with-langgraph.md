---
title: Building Agents with LangGraph
description: A comprehensive guide to building stateful, multi-step AI agents using LangGraph, covering graph-based state machines, nodes and edges, conditional routing, human-in-the-loop workflows, multi-agent architectures, and persistence with checkpointing.
---

LangGraph is a library built on top of LangChain that enables construction of stateful, cyclical agent workflows as explicit graphs. Unlike simple chain-based pipelines, LangGraph models agents as state machines where each node is a processing step and edges define transitions — including conditional branches and cycles that allow agents to reason iteratively.

## Core Concepts

### State

LangGraph agents operate on a shared **state** object that flows through the graph and is updated by each node. State is defined as a typed dictionary:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    iteration_count: int
```

The `Annotated` type with `operator.add` tells LangGraph to **accumulate** new messages into the list rather than replace the entire field — a common pattern for chat history.

### Nodes

Nodes are Python functions (or runnables) that receive the current state and return a partial state update:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    # execute the tool requested in the last AI message
    last_message = state["messages"][-1]
    tool_result = execute_tool(last_message.tool_calls[0])
    return {"messages": [tool_result]}
```

### Edges

Edges connect nodes and can be:

- **Direct** — always transition from node A to node B
- **Conditional** — choose the next node based on a routing function

```python
from langgraph.graph import StateGraph, END

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("call_model", call_model)
workflow.add_node("call_tool", call_tool)
workflow.set_entry_point("call_model")
workflow.add_conditional_edges("call_model", should_continue)
workflow.add_edge("call_tool", "call_model")  # cycle back

graph = workflow.compile()
```

## Building a ReAct Agent

The ReAct (Reasoning and Acting) pattern interleaves reasoning steps with tool calls until the agent decides it has a sufficient answer.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_web(query: str) -> str:
    """Search the web for recent information."""
    return web_search_api(query)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

tools = [search_web, calculate]
agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [HumanMessage(content="What is the GDP of Germany multiplied by 0.15?")]
})
```

`create_react_agent` is a prebuilt convenience that internally constructs the same state graph pattern — node for LLM call, conditional edge to tool node, cycle back to LLM.

## Persistence and Checkpointing

One of LangGraph's key differentiators is built-in **persistence**. A checkpointer saves state after every node execution, enabling:

- **Resume on failure** — re-run from the last saved checkpoint
- **Human-in-the-loop** — pause, inspect, modify, and resume execution
- **Multi-turn conversations** — restore conversation history across sessions

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.db")
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user-123-session-7"}}

# First invocation
result = graph.invoke({"messages": [HumanMessage("Plan a trip to Tokyo")]}, config)

# Resume later in same thread — previous messages restored automatically
result = graph.invoke({"messages": [HumanMessage("Add one more restaurant recommendation")]}, config)
```

Supported backends: in-memory, SQLite, PostgreSQL (via `AsyncPostgresSaver`), Redis.

## Human-in-the-Loop

LangGraph supports **interrupt points** that pause execution and wait for human approval before proceeding:

```python
from langgraph.graph import interrupt

def review_action(state: AgentState):
    proposed_action = state["proposed_action"]
    # Execution pauses here; human can inspect state and approve/modify
    approved = interrupt({"action": proposed_action, "reason": state["reasoning"]})
    return {"approved_action": approved}

graph = workflow.compile(checkpointer=memory, interrupt_before=["execute_action"])
```

The interrupted graph is resumed by passing the human's response back via `graph.invoke(Command(resume=approval), config)`.

## Multi-Agent Architectures

LangGraph supports multiple agents collaborating as a network:

### Supervisor Pattern

A supervisor agent receives the task and delegates sub-tasks to specialist agents, collecting their results:

```python
from langgraph.graph import StateGraph

def supervisor(state):
    # Decide which specialist to invoke next
    next_agent = llm.invoke(supervisor_prompt + str(state))
    return {"next": next_agent.content}

def research_agent(state):
    ...

def writing_agent(state):
    ...

builder = StateGraph(MultiAgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", research_agent)
builder.add_node("writer", writing_agent)
builder.add_conditional_edges("supervisor", lambda s: s["next"], {
    "researcher": "researcher",
    "writer": "writer",
    "FINISH": END
})
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
```

### Subgraph Pattern

Each specialist agent is a compiled LangGraph graph that can be embedded as a node within a parent graph, enabling modular and reusable agent components.

## Streaming

LangGraph streams both token-level output and intermediate graph state updates:

```python
for chunk in graph.stream({"messages": [input_message]}, stream_mode="updates"):
    print(chunk)  # Each node's state update printed as it completes

# Stream tokens in real time
for chunk in graph.stream({"messages": [input_message]}, stream_mode="messages"):
    if chunk[1]["langgraph_node"] == "call_model":
        print(chunk[0].content, end="", flush=True)
```

## Comparison with Alternative Agent Frameworks

| Feature | LangGraph | CrewAI | AutoGen |
| --- | --- | --- | --- |
| State model | Explicit typed graph | Role-based crew | Conversation-based |
| Cycles / loops | Native support | Limited | Via group chat |
| Persistence | Built-in checkpointing | External | External |
| Human-in-the-loop | Interrupt / resume | Manual | Built-in |
| Streaming | Token + node events | Limited | Token-level |
| Complexity | Medium-high | Low-medium | Medium |

## Best Practices

- **Define state explicitly** — well-typed state makes debugging and testing straightforward
- **Keep nodes focused** — each node should do one thing (call LLM, execute tool, validate output)
- **Use interrupts for irreversible actions** — database writes, external API calls, and financial transactions should always include a review step
- **Set recursion limits** — prevent infinite loops with `graph.invoke(..., {"recursion_limit": 25})`
- **Test subgraphs independently** — compile and unit-test each agent subgraph before integrating into the parent workflow
- **Monitor with LangSmith** — native tracing integration provides run-level visibility into every node's inputs, outputs, and latency
