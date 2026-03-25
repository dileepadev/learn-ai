---
title: Introduction to Chain of Density
description: Learn about the Chain of Density (CoD) technique for creating information-dense entity-centric summaries with LLMs.
---

Chain of Density (CoD) is a prompt-engineering technique designed to create summaries that are increasingly dense with information without increasing the overall length of the summary.

## How Chain of Density Works

The goal of CoD is to iteratively generate a summary, identifying "missing entities" from the source text and incorporating them into the previous summary in each step. This process is typically repeated 5 times.

1. **Step 1:** Generate an initial sparse summary.
2. **Steps 2-5:** Identify 1-3 new, salient entities from the source that are not in the current summary, and rewrite the summary to include them while keeping it concise.

## Benefits of CoD

- **Informative:** Packs more "meaning" into every sentence.
- **Readable:** Maintains a fixed length, avoiding the bloat of long summaries.
- **Controlled:** Allows for specific targeting of entities or facts.

## When to Use CoD

Use CoD when you need high-quality, dense summaries for news articles, research papers, or any lengthy document where entity extraction is critical.
