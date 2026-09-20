---
title: Semantic Role Labeling - Who Did What to Whom
description: Learn how semantic role labeling identifies predicate-argument structure in sentences to extract who did what to whom, when, and where.
---

Semantic role labeling (SRL) identifies the predicate of a sentence (usually a verb) and labels each argument with its semantic role — agent, patient, instrument, location, time — regardless of surface word order.

```text
"Ada gave the book to Charles yesterday."
gave: PREDICATE
Ada: AGENT
the book: THEME
Charles: RECIPIENT
yesterday: TIME
```

## Why This Is More Than Parsing

Syntactic parsing tells you the grammatical subject and object of a sentence; SRL tells you the semantic function those constituents play. "The window broke" and "John broke the window" assign different syntactic subjects, but SRL correctly labels "the window" as the PATIENT in both, capturing that the window is the thing affected regardless of grammatical role.

## Approaches

Classic SRL pipelines first identify predicates, then classify candidate argument spans (often from a syntactic parse) into role labels using a frame-specific inventory such as PropBank's numbered arguments (ARG0 for agent-like, ARG1 for patient-like). Neural SRL models skip explicit parsing and instead use a transformer encoder plus a span classifier trained end-to-end, conditioning role predictions on the predicate's contextual representation.

## Applications

SRL structures text for information extraction pipelines that need to answer "who did what to whom" reliably — legal document analysis, news event extraction, and question answering that requires precise argument attribution rather than approximate keyword matching. It also supports building structured knowledge graphs from unstructured narrative text.

## Practical Guidance

Full SRL pipelines are heavier than most applications need; if you only need to extract a specific relation type, targeted relation extraction or an LLM prompted with a structured output schema is often simpler and just as accurate. Reach for SRL when you need general-purpose, frame-consistent predicate-argument structure across many verb types and don't want to define extraction rules per relation.
