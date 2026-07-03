---
title: "Context Pruning for LLMs"
description: "How removing low-value context can improve AI accuracy, speed, and cost at the same time."
---

More context is not always better. Large language models often perform best when they receive the **right** information, not the maximum amount of information. Context pruning is the process of removing irrelevant, stale, or redundant material before generation.

## Why Pruning Helps

- It reduces token cost
- It improves latency
- It lowers the chance of distraction from irrelevant details
- It makes instructions and evidence easier for the model to follow

## What Gets Pruned

Common candidates include outdated chat history, duplicate retrieved passages, low-relevance search results, and boilerplate instructions that do not matter for the current step.

## The Core Idea

Context should compete for space. If a piece of information is not improving the answer, it is probably making the prompt worse. Strong AI systems are selective about what they include, not just ambitious about how much they can fit.
