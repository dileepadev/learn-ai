---
title: "Agentic Memory: Long-Term Context for AI Agents"
description: "Understand how agents maintain state and context over time to perform consistently across many tasks."
---

# Agentic Memory: Long-Term Context for AI Agents

For an AI agent to perform complex, multi-day tasks, it needs a way to remember past interactions, preferences, and results. Memory is the "glue" that holds an agent's long-term behavior together.

---

## 1. Types of Memory for AI Agents

- **Short-Term Memory**: The model's context window. This includes the current dialogue or task.
- **Long-Term Memory**: Storing information outside the context window (e.g., in a database or local file).
- **Semantics Memory**: Remembering the meaning behind past interactions, often using a vector database.

---

## 2. Implementing Memory in Agentic Workflows

- **Context Window Management**: LLMs have limited context. Summarize old conversation history or "roll over" important facts to new contexts.
- **Persistent Storage**: Use databases (like Azure Cosmos DB) to save user preferences, tool results, and history.
- **Retrieval-Augmented Memories**: Before an agent starts a task, it can query its memory for relevant past experiences to inform its current reasoning.

---

## Why Memory Matters for Agents

Memory allows for **Personalization**, **Continuity**, and **Efficiency**. An agent that remembers who you are and what you've done before is more helpful than one that starts from scratch every time.
