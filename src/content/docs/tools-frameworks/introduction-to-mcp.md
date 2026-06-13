---
title: Introduction to Model Context Protocol (MCP)
description: Learn about the Model Context Protocol (MCP), the open standard that lets AI models securely connect to external tools, data sources, and services.
---

The Model Context Protocol (MCP) is an open standard developed by Anthropic that defines how AI models — particularly LLM-based assistants — can communicate with external tools, APIs, databases, and services. It provides a universal interface so that AI applications can be extended with capabilities without requiring custom integrations for every tool.

## Why MCP Exists

LLMs are powerful reasoners, but they are static — their knowledge is frozen at training time and they cannot natively access live data or perform actions in the real world. Before MCP, connecting an LLM to external tools meant writing custom integration code for every combination of model and tool.

MCP solves this by defining a standard protocol: a tool exposes itself as an MCP server, and any compatible AI client can discover and use it without bespoke integration work. Think of it as USB-C for AI tools.

## How MCP Works

MCP uses a **client-server architecture**:

- **MCP Server:** An application that exposes tools, resources, and prompts. Examples: a filesystem server, a database query server, a GitHub API server.
- **MCP Client:** The AI application (e.g., Claude Desktop, an IDE, a custom agent) that connects to servers and uses their capabilities.

Communication happens over JSON-RPC, either locally (stdio) or over a network (HTTP + SSE).

## Core Primitives

### Tools
Functions the model can call. Each tool has a name, description, and input schema. The model decides when to invoke a tool and with what arguments.

Example tools: `read_file`, `search_web`, `run_sql_query`, `send_email`.

### Resources
Structured data sources the model can read. Resources are identified by URIs and can be static (a file) or dynamic (a database query result).

### Prompts
Pre-built prompt templates that the server can expose. Useful for providing the model with domain-specific instructions.

## A Simple MCP Server Example

```python
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types

app = Server("my-server")

@app.list_tools()
async def list_tools():
    return [types.Tool(
        name="get_weather",
        description="Get current weather for a city",
        inputSchema={"type": "object", "properties": {"city": {"type": "string"}}}
    )]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        # Call weather API...
        return [types.TextContent(type="text", text=f"Weather in {city}: 22°C, sunny")]
```

## Who Uses MCP?

MCP is supported by a growing ecosystem:
- **Claude Desktop** — supports MCP servers natively.
- **Cursor, Zed, Cline** — AI-enabled code editors with MCP support.
- **Open-source community** — hundreds of community-built MCP servers for GitHub, Slack, databases, web browsers, and more.

## MCP vs. Function Calling

Both MCP and OpenAI-style function calling let models invoke external tools, but:
- **Function calling** is model-specific and stateless — each call is a one-off interaction.
- **MCP** is a persistent, bidirectional protocol with session state, resource subscriptions, and a discoverable tool registry. It is model-agnostic and enables richer, stateful interactions.

## Getting Started

Install the MCP Python SDK:

```bash
pip install mcp
```

Browse the official MCP server registry at `github.com/modelcontextprotocol/servers` for ready-to-use servers, or build your own using the SDK.
