---
title: Discourse Parsing and Coherence Analysis
description: Learn how discourse parsing models the rhetorical structure connecting sentences in a document, beyond what sentence-level NLP captures.
---

Discourse parsing analyzes how sentences and clauses relate to each other across a whole document — contrast, elaboration, causation, sequencing — capturing structure that sentence-level tasks like part-of-speech tagging or dependency parsing don't address at all.

## Rhetorical Structure Theory

Rhetorical Structure Theory (RST), the most influential discourse framework, represents a document as a tree where leaves are elementary discourse units (roughly clause-sized spans of text) and internal nodes label the rhetorical relation connecting adjacent spans, such as `Elaboration`, `Contrast`, `Cause`, or `Background`.

```text
"The server crashed. As a result, all pending requests were lost."
[The server crashed.]  --Cause-->  [all pending requests were lost.]
```

Many relations are also asymmetric, distinguishing a "nucleus" (the more central, load-bearing span) from a "satellite" (supporting or subordinate information), which matters for tasks like extractive summarization that want to preserve nuclei while safely dropping satellites.

## Penn Discourse Treebank Style Relations

An alternative framework, following the Penn Discourse Treebank, focuses specifically on discourse connectives (words like "because," "however," "meanwhile") and the relations they signal between the text spans they connect, including cases where a relation holds implicitly with no explicit connective present at all — arguably the harder and more linguistically interesting sub-problem, since it requires inferring a relationship the writer left unstated.

## Why Coherence Matters for Generation

Evaluating whether a long generated document reads as coherent, rather than just fluent sentence by sentence, is exactly the gap discourse-level analysis addresses — a document can have zero individual grammatical errors while still failing to logically connect its ideas, repeating points without building on them, or contradicting an earlier claim later on. Discourse-aware evaluation metrics and coherence-scoring models draw directly on discourse parsing concepts to assess this document-level quality that sentence-level metrics miss entirely.

## Practical Guidance

Full RST-style discourse parsing is a specialized, less mature NLP capability than most modern sentence-level tasks, and off-the-shelf tools are less robust across domains than, say, dependency parsers. For most practical coherence-evaluation needs, using an LLM prompted to assess a document's logical flow and highlight incoherent transitions is currently a more practical approach than deploying a dedicated RST parser, unless you need the fully formal, structured discourse tree for downstream symbolic processing.
