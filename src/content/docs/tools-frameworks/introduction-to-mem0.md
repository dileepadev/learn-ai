---
title: Introduction to Mem0 - A Memory Layer for AI Agents
description: Learn how Mem0 extracts, stores, and retrieves durable memories for AI agents so conversations persist meaningful context across sessions.
---

Mem0 is a memory layer for AI applications and agents, handling extraction, storage, and retrieval of durable facts and preferences from conversations so an agent can recall relevant context across sessions rather than starting from a blank slate every time.

## Why LLM Context Windows Aren't Enough

Feeding an entire conversation history back into a prompt every turn is expensive, eventually exceeds context window limits, and buries genuinely important facts (a user's stated preference, a decision made three sessions ago) in a flood of irrelevant chat turns. Mem0 instead extracts salient, durable facts from conversations and stores them separately, so retrieval at query time pulls only what's relevant to the current turn rather than replaying entire histories.

## Core Workflow

```python
from mem0 import Memory

memory = Memory()
memory.add("I prefer flights with no layovers and always fly economy.", user_id="alice")

relevant = memory.search("book me a flight to Chicago", user_id="alice")
# returns the stored travel preference relevant to this new request
```

Internally, Mem0 uses an LLM to decide what from a conversation is worth remembering, update or reconcile new information against existing memories (handling contradictions — a changed preference should overwrite the old one, not just add to it), and stores memories in a vector store for semantic retrieval, often alongside a graph store for capturing structured relationships between remembered facts.

## Memory Scopes

Mem0 supports memory scoped to a user, a session, or an agent, so a multi-agent system can maintain both agent-specific working memory and shared user-level long-term memory without conflating the two.

## Practical Guidance

Use a dedicated memory layer like Mem0 when an agent needs to feel consistent and personalized across sessions — remembering stated preferences, past decisions, or ongoing project context — rather than re-deriving this from scratch or manually building an ad hoc fact-extraction pipeline. For purely single-session, stateless tasks, the added complexity of a memory layer is usually unnecessary.
