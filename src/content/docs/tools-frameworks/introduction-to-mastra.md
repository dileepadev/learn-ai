---
title: Introduction to Mastra - TypeScript Framework for AI Agents
description: Learn how Mastra provides agents, workflows, memory, and RAG primitives for building AI applications in the TypeScript ecosystem.
---

Mastra is an open-source TypeScript framework for building AI agents and workflows. It targets JavaScript and TypeScript developers who want agent capabilities without leaving their existing web stack.

## Key Concepts

- **Agents**: LLM-backed entities with instructions, tools, and optional memory
- **Tools**: typed functions the agent can call, described with Zod schemas
- **Workflows**: graph-based, durable orchestration with branching, parallel steps, and suspend/resume
- **Memory**: conversation history and semantic recall
- **RAG**: document chunking, embedding, and retrieval utilities
- **Evals**: scoring of agent outputs

## Defining a Tool and Agent

```typescript
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";

const weatherTool = createTool({
  id: "get-weather",
  description: "Get the current weather for a city",
  inputSchema: z.object({ city: z.string() }),
  execute: async ({ context }) => {
    return { temperature: 21, condition: "Sunny", city: context.city };
  },
});

export const assistant = new Agent({
  name: "Weather Assistant",
  instructions: "Answer weather questions using the tool.",
  model: openai("gpt-4o-mini"),
  tools: { weatherTool },
});
```

## Workflows

Workflows model multi-step processes where each step has typed inputs and outputs. Steps can run sequentially, in parallel, or conditionally, and a workflow can pause for human input and resume later, which is valuable for approval flows.

## Why TypeScript Matters

End-to-end types catch mismatched tool schemas at compile time, and the framework fits naturally into Next.js and Node services, simplifying deployment alongside existing front-end and API code.

## Considerations

The ecosystem is younger than Python-based alternatives, so check that integrations you need (vector stores, model providers) are available.
