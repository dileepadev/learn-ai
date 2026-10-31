---
title: Introduction to Dify - Open-Source LLM Application Platform
description: Learn how Dify combines a visual workflow builder, RAG pipeline, agent tooling, and observability into a self-hostable platform for building LLM applications.
---

Dify is an open-source platform for building and operating LLM applications. It bundles prompt management, retrieval-augmented generation, agent capabilities, and a visual workflow editor behind a web UI and an API, so teams can prototype and ship without writing all the orchestration code themselves.

## Main Building Blocks

- **Prompt orchestration**: design and version prompts with variables and model settings
- **Knowledge base (RAG)**: upload documents, configure chunking and embeddings, and enable hybrid retrieval with reranking
- **Workflows**: a node-based canvas combining LLM calls, retrieval, code, conditionals, HTTP requests, and loops
- **Agents**: models that call tools through function calling or ReAct-style reasoning
- **Model providers**: connect to hosted APIs or local models through a unified interface

## Application Types

| Type | Purpose |
|------|---------|
| Chatbot | Conversational assistant with memory and optional retrieval |
| Text generator | Single-shot generation from a form |
| Agent | Tool-using assistant |
| Workflow | Deterministic multi-step pipeline |

## Serving as an API

Each application is exposed through a REST API, so front ends or backend services can call it:

```bash
curl -X POST 'https://your-dify-host/v1/chat-messages' \
  -H 'Authorization: Bearer APP_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {}, "query": "Summarize our refund policy", "user": "u-123", "response_mode": "blocking"}'
```

## Operational Features

Dify records conversation logs, token usage, and user feedback, helping teams inspect failures and iterate on prompts and retrieval settings.

## Trade-offs

Platforms like Dify speed up prototyping but constrain you to their abstractions. For highly custom control flow or strict performance requirements, a code-first framework may fit better.
