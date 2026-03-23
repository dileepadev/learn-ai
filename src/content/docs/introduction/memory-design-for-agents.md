---
title: Memory Design for Agents
description: Patterns and practices for designing memory systems in autonomous AI agents.
---

Agent memory enables persistent context across interactions and improves long-term performance.

## Memory Types

- Short-term buffers for immediate context
- Episodic memory for past interactions
- Long-term knowledge stores for facts and user preferences

## Design Tips

- Persist only what improves agent decisions to reduce storage costs
- Use embeddings for semantic retrieval of past interactions
- Implement TTL and pruning strategies to limit growth
