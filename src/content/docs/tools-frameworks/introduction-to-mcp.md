---
title: "Introduction to Model Context Protocol (MCP)"
description: "Learn what the Model Context Protocol is, how it standardizes tool and resource access for AI agents, and how to build and connect MCP servers to your AI applications."
---

**Model Context Protocol (MCP)** is an open standard developed by Anthropic that defines how AI models connect to external tools, data sources, and services. It's the USB-C of AI integrations — a single protocol that works across different models and clients.

## The Problem MCP Solves

Before MCP, every AI application had to build custom integrations for every tool. A coding assistant needed custom code to connect to GitHub, Jira, Slack, and databases. When you switched models, you rebuilt the integrations. When a tool changed its API, every integration broke.

MCP standardizes this: tools expose themselves as MCP servers, and any MCP-compatible client (Claude, Cursor, your custom agent) can use them without custom integration code.

## Core Concepts

### Resources
Static or dynamic data that the model can read. Examples: file contents, database records, API responses.

```json
{
  "uri": "file:///project/README.md",
  "name": "README",
  "mimeType": "text/markdown"
}
```

### Tools
Functions the model can call to take actions or retrieve information.

```json
{
  "name": "create_github_issue",
  "description": "Create a new issue in a GitHub repository",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": {"type": "string"},
      "title": {"type": "string"},
      "body": {"type": "string"}
    }
  }
}
```

### Prompts
Reusable prompt templates that servers can expose to clients.

## Building an MCP Server

Using the Python SDK:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

server = Server("my-tool-server")

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        location = arguments["location"]
        # Call weather API
        return [types.TextContent(type="text", text=f"Sunny, 22°C in {location}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
```

## Transport Options

MCP supports two transport mechanisms:

- **stdio**: The client spawns the server as a subprocess and communicates via stdin/stdout. Simple, secure, no network exposure.
- **HTTP with SSE**: The server runs as an HTTP service. Enables remote servers and multi-client scenarios.

## Connecting to Claude Desktop

Add your server to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-tool-server": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

## The MCP Ecosystem

A growing ecosystem of pre-built MCP servers covers:
- **Development tools**: GitHub, GitLab, Jira, Linear.
- **Data sources**: PostgreSQL, SQLite, Google Drive, Notion.
- **Web**: Browser automation, web search, scraping.
- **Infrastructure**: AWS, Kubernetes, Docker.

MCP is rapidly becoming the standard integration layer for AI agents, with support from Anthropic, OpenAI, Google, and most major AI development tools.
