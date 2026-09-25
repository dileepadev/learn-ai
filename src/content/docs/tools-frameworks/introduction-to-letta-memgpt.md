---
title: Introduction to Letta (MemGPT) - Agents with Self-Managed Memory
description: Learn how Letta, built on the MemGPT research, gives LLM agents an operating-system-style memory hierarchy they manage themselves.
---

Letta (formerly known through the MemGPT research project) gives LLM agents a tiered memory system modeled loosely on operating system virtual memory, letting an agent manage its own context rather than relying entirely on an external orchestration layer to decide what to keep or discard.

## The Memory Hierarchy

Letta agents distinguish between main context (the actual tokens visible to the model in its current prompt, analogous to RAM) and external context (a larger store of memories not currently loaded, analogous to disk). The agent itself, through function calls the framework exposes, decides when to move information between these tiers:

```text
main context (limited, always visible)
   <-> archival memory (long-term, searchable, not directly visible)
   <-> recall memory (past conversation history, searchable)
```

This is the key conceptual difference from a simple RAG-backed chatbot: instead of an external pipeline deciding what to retrieve before each turn, the agent itself calls memory-management functions (save this fact, search archival memory for X, edit this stored memory) as part of its own reasoning process.

## Self-Editing Memory

Because the agent can explicitly rewrite its own core memory blocks (for example, a persistent block holding key facts about the user or the agent's persona), it can correct outdated information or refine its own understanding over time, rather than accumulating stale or contradictory facts the way a purely append-only memory log would.

## Why This Matters for Long-Running Agents

Agents intended to run indefinitely — a persistent personal assistant, a long-term research agent — need memory management that scales far beyond what fits in any single context window, and hand-coding retrieval heuristics for every kind of information an agent might need to recall does not scale well. Letta's approach shifts responsibility for that judgment onto the agent itself, using the same reasoning capability it already applies to the task at hand.

## Practical Guidance

Letta adds meaningful complexity compared to a stateless prompt-and-response agent, so it is best suited to genuinely long-running, personalized agents where memory continuity is a core requirement rather than a nice-to-have. For simpler use cases, a lighter memory layer like [[introduction-to-mem0]] or a standard RAG pipeline over conversation history may be sufficient with far less operational overhead.
