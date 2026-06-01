---
title: "Introduction to Agent-to-Agent (A2A) Protocol"
description: "Learn how Google's Agent-to-Agent (A2A) protocol enables interoperability between AI agents from different vendors and frameworks, and how it complements MCP for building multi-agent systems."
---

As AI agents proliferate, a new challenge emerges: how do agents built by different teams, using different frameworks, communicate with each other? **Agent-to-Agent (A2A)** is an open protocol developed by Google that standardizes how agents discover, communicate with, and delegate tasks to other agents.

## The Problem A2A Solves

Today's multi-agent systems are mostly siloed. A LangGraph agent can call tools, but it can't natively delegate a subtask to a CrewAI agent running on a different server. An enterprise might have dozens of specialized agents built by different teams — a research agent, a coding agent, a data analysis agent — with no standard way for them to collaborate.

A2A provides the interoperability layer: a common language for agents to talk to each other, regardless of the underlying framework or model.

## How A2A Relates to MCP

A2A and MCP are complementary, not competing:

- **MCP** connects agents to *tools and resources* (databases, APIs, file systems).
- **A2A** connects agents to *other agents* (delegation, collaboration, orchestration).

An agent might use MCP to access a database and A2A to delegate a subtask to a specialized agent.

## Core Concepts

### Agent Card
Every A2A-compatible agent publishes an **Agent Card** — a JSON document describing its capabilities, skills, and how to communicate with it. This is the agent's "business card" for discovery.

```json
{
  "name": "Data Analysis Agent",
  "description": "Analyzes structured data and produces insights",
  "url": "https://agents.example.com/data-analyst",
  "version": "1.0.0",
  "skills": [
    {
      "id": "analyze_csv",
      "name": "Analyze CSV Data",
      "description": "Performs statistical analysis on CSV files",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "data"]
    }
  ],
  "authentication": {
    "schemes": ["bearer"]
  }
}
```

### Tasks
The fundamental unit of work in A2A. A client agent sends a task to a remote agent and receives updates on its progress.

Tasks have a lifecycle: `submitted → working → completed | failed | canceled`

```python
# Sending a task to a remote agent
task = {
    "id": "task-123",
    "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Analyze the attached sales data and identify the top 3 trends."}]
    }
}
```

### Streaming with SSE
Long-running tasks stream progress updates back to the client using Server-Sent Events (SSE). The client receives incremental updates rather than waiting for the full result.

### Push Notifications
For very long tasks, agents can register a webhook URL to receive completion notifications asynchronously, freeing the client from maintaining a persistent connection.

## Multi-Agent Patterns with A2A

### Orchestrator-Worker
A central orchestrator agent breaks down a complex task and delegates subtasks to specialized worker agents. The orchestrator collects results and synthesizes the final output.

### Peer-to-Peer
Agents collaborate as equals, each contributing their specialty. Agent A handles research, passes results to Agent B for analysis, which passes to Agent C for report generation.

### Dynamic Discovery
Agents query a registry of available agents, select the most appropriate one for a subtask based on Agent Cards, and delegate dynamically — without hardcoded routing logic.

## Building an A2A Server

Using the Python SDK:

```python
from a2a.server import A2AServer
from a2a.types import AgentCard, AgentSkill, Task, TaskStatus

agent_card = AgentCard(
    name="My Specialist Agent",
    description="Handles specialized analysis tasks",
    url="http://localhost:8000",
    version="1.0.0",
    skills=[
        AgentSkill(
            id="analyze",
            name="Analyze Data",
            description="Performs deep analysis",
            inputModes=["text"],
            outputModes=["text"]
        )
    ]
)

server = A2AServer(agent_card=agent_card)

@server.task_handler
async def handle_task(task: Task):
    # Process the task
    result = await do_analysis(task.message)
    return TaskStatus(
        state="completed",
        message={"role": "agent", "parts": [{"type": "text", "text": result}]}
    )

server.run(host="0.0.0.0", port=8000)
```

## Current Ecosystem

A2A launched in early 2025 with support from over 50 technology partners including SAP, Salesforce, Atlassian, and ServiceNow. Major agent frameworks (LangGraph, CrewAI, AutoGen) are adding A2A support.

Combined with MCP for tool access, A2A is becoming a foundational protocol for enterprise multi-agent systems where agents from different vendors need to collaborate securely and reliably.
