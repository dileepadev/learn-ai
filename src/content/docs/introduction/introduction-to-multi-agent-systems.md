---
title: Introduction to Multi-Agent Systems
description: Collaborative AI systems working together to solve problems.
---

A multi-agent system (MAS) consists of multiple AI agents—each with its own set of skills—working together to achieve a common goal. This mirrors how human teams collaborate in the workplace.

## How Multi-Agent Systems Work

In a MAS, a "Master Agent" or a "Coordinator" breaks down a complex task into smaller subtasks and assigns them to specialized agents.

1. **Planner Agent**: Analyzes the request and defines the steps.
2. **Search Agent**: Gathers information from external sources.
3. **Drafting Agent**: Creates the final output based on the gathered data.
4. **Reviewer Agent**: Checks the output for errors or inconsistencies.

## Benefits of Multi-Agent Systems

- **Specialization**: Each agent can be fine-tuned or given specific tools for its role.
- **Autonomy**: Agents can work independently and communicate with each other.
- **Robustness**: If one agent fails, other agents can take over or provide alternative solutions.

## Common Frameworks

- **AutoGen**: A framework for building multi-agent conversations.
- **CrewAI**: Designed for orchestrating role-playing autonomous AI agents.
- **Semantic Kernel**: Microsoft's framework for integrating AI models into applications with plugins and planners.
