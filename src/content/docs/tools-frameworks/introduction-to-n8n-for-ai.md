---
title: Introduction to n8n for AI Automation Workflows
description: Learn how n8n's node-based workflow automation combines triggers, integrations, and LLM agent nodes to automate business processes with AI.
---

n8n is a workflow automation tool with a visual node editor and hundreds of integrations. Its AI nodes make it practical for wiring LLMs into everyday business processes such as email triage, document processing, and support routing.

## Workflow Anatomy

A workflow is a graph of nodes:

1. **Trigger**: a webhook, schedule, incoming email, or form submission starts the run
2. **Processing nodes**: transform data, call APIs, branch on conditions
3. **AI nodes**: call a language model, run an agent, or query a vector store
4. **Action nodes**: write to a database, post to chat, create a ticket

## AI Capabilities

- **LLM chain nodes** for classification, extraction, and summarization with a chosen prompt
- **AI Agent node** that can decide which connected tools to call
- **Memory nodes** that keep conversation context
- **Vector store and embedding nodes** for retrieval over your own documents
- **Output parsers** that enforce structured JSON from model responses

## Example: Support Ticket Triage

```text
Webhook (new ticket)
  -> LLM: classify category and urgency (JSON output)
  -> Switch on urgency
       high   -> Notify on-call channel + create priority ticket
       normal -> Draft suggested reply -> Add to queue
```

## Good Practices

- Use structured output parsing so downstream nodes receive predictable fields.
- Add a human approval step before any irreversible action such as sending external email.
- Handle failures with error workflows and retries, since model and API calls are fallible.
- Keep credentials in the credential store rather than in node parameters.

## When to Choose It

n8n works well for integration-heavy automations with moderate AI logic. Complex agent behavior with custom state handling is usually easier in a code framework.
