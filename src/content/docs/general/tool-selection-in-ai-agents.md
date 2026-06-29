---
title: "Tool Selection in AI Agents"
description: "How agents decide when to search, calculate, call an API, or answer directly."
---

An AI agent becomes more useful when it has access to tools, but only if it knows which tool to use and when. Poor tool selection leads to wasted latency, wrong actions, or answers that should have been grounded in data but were generated from memory instead.

## Common Selection Signals

- Does the task require fresh information?
- Is precise computation needed?
- Is there a structured source of truth available?
- Can the question be answered directly without external calls?

## Failure Modes

Agents often overuse tools, underuse tools, or pick the wrong one. For example, they may call a general search tool when a narrow customer database query would be faster and more accurate.

## Better Agent Design

Clear tool descriptions, examples, and explicit policies improve selection. In practice, tool choice is one of the biggest determinants of agent quality.
