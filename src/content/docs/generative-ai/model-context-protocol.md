---
title: Model Context Protocol (MCP)
description: Learn about the Model Context Protocol — an open standard that enables AI models and agents to securely connect to external tools, data sources, and services through a unified interface.
---

The **Model Context Protocol (MCP)** is an open standard introduced by Anthropic in late 2024 that defines a uniform interface for connecting AI language models with external tools, data sources, and services. Its goal is to solve the **M×N integration problem** — where every AI application otherwise needs custom integration code for every external service.

## The Problem MCP Solves

Before MCP, integrating an LLM with external systems required:

- Custom function schemas for every tool.
- Bespoke API wrappers for every data source.
- No shared standard between clients (Claude, GPT-based tools, open-source models) and servers (databases, APIs, file systems).

This created fragmented ecosystems where tool integrations weren't portable across AI systems.

MCP addresses this by standardizing the **protocol layer** between AI models and the world — much like how HTTP standardized web communication or LSP (Language Server Protocol) standardized IDE language support.

## Architecture

MCP follows a **client-server architecture**:

```
┌─────────────────────────────┐
│        AI Application       │
│  (Host: Claude Desktop,     │
│   VS Code, custom app)      │
│                             │
│  ┌─────────┐ ┌─────────┐   │
│  │  MCP    │ │  MCP    │   │
│  │ Client  │ │ Client  │   │
│  └────┬────┘ └────┬────┘   │
└───────┼───────────┼─────────┘
        │           │
   ┌────▼────┐ ┌────▼────┐
   │  MCP    │ │  MCP    │
   │ Server  │ │ Server  │
   │ (Files) │ │  (DB)   │
   └─────────┘ └─────────┘
```

- **Host** — The AI application (e.g., Claude Desktop, an IDE, a custom agent).
- **MCP Client** — Lives inside the host; manages connections to MCP servers.
- **MCP Server** — A lightweight process exposing capabilities (tools, resources, prompts) to the client.

## Core Primitives

MCP servers expose three types of capabilities:

### Tools

**Tools** are callable functions the AI model can invoke — analogous to OpenAI function calling, but standardized:

```json
{
  "name": "read_file",
  "description": "Read the contents of a file from the filesystem",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": { "type": "string", "description": "File path to read" }
    },
    "required": ["path"]
  }
}
```

The model decides when to call a tool based on its description; the MCP client routes the call to the correct server and returns the result.

### Resources

**Resources** expose data the model can read — files, database records, API responses — as URI-addressable content:

```
file:///home/user/documents/report.pdf
db://mydb/customers/123
```

Resources can be static (pre-defined) or dynamic (generated on request).

### Prompts

**Prompts** are reusable, parameterized message templates that servers expose for common workflows. A user or application can invoke a named prompt with arguments rather than constructing message sequences manually.

## Transport Mechanisms

MCP supports two transports:

- **stdio** — The server runs as a subprocess; client communicates over standard input/output. Ideal for local tools.
- **HTTP + SSE (Server-Sent Events)** — For remote or networked MCP servers. Uses SSE for server-to-client messages and HTTP POST for client-to-server messages.

## Security Model

MCP incorporates several security considerations:

- **Human-in-the-loop** — Hosts should prompt users for approval before executing tool calls, especially destructive ones.
- **Least-privilege servers** — Each MCP server should expose only what is needed for its domain.
- **No ambient authority** — Servers cannot initiate connections to the client; they only respond to requests.
- **Prompt injection vigilance** — Data returned by resources could attempt to hijack model behavior; applications should validate and sanitize resource content.

## Building an MCP Server

MCP servers can be built using official SDKs in Python, TypeScript, and other languages:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

app = Server("my-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        # fetch weather...
        return [types.TextContent(type="text", text=f"Weather in {city}: Sunny, 22°C")]

async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())
```

## MCP vs. Function Calling

| Feature | OpenAI Function Calling | MCP |
|---|---|---|
| Standardization | Model-specific | Open standard |
| Portability | One LLM provider | Any MCP-compatible model |
| Server reuse | No | Yes — one server, many clients |
| Resources | No | Yes |
| Discovery | Manual schema passing | Dynamic server discovery |

MCP is designed to be **model-agnostic** — the same MCP server works with Claude, a local Llama model, or any other MCP-compatible host.

## Ecosystem and Adoption

Since its release, MCP has seen rapid adoption:

- **Official servers** — Anthropic and partners maintain servers for file systems, databases (PostgreSQL, SQLite), web browsers, GitHub, Slack, Google Drive, and more.
- **IDE integration** — VS Code Copilot supports MCP servers for extending AI coding assistance.
- **Agent frameworks** — LangChain, Semantic Kernel, and AutoGen have added MCP client support.
- **Open registry** — Community-maintained directories of available MCP servers (e.g., mcp.so, Smithery).

## When to Use MCP

MCP is a good fit when:

- Building agents that need to interact with multiple external systems.
- You want portability across AI providers without rewriting tool definitions.
- You want to share tool servers across multiple applications or teams.

For a single, in-process tool in a narrow application, simpler function calling may suffice. For composable, multi-tool agentic systems with diverse backends, MCP provides significant value.
