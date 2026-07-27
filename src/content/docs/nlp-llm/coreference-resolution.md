---
title: Coreference Resolution - Tracking Who "It" and "They" Refer To
description: See how NLP systems link pronouns and repeated mentions back to the entities they describe.
---

Coreference resolution identifies all expressions in a text that refer to the same real-world entity. Given the sentence:

```text
Maria picked up her laptop because she needed it for the meeting.
```

A coreference system should link "her", "she", and "it" to "Maria" and "laptop" respectively, grouping mentions into clusters that each represent one entity.

## Mention Detection and Clustering

The task splits into two stages. **Mention detection** finds candidate spans that could refer to an entity—names, pronouns, and noun phrases. **Clustering** then decides which mentions belong together. Classic pairwise models score every pair of mentions for whether they corefer and cluster transitively; modern end-to-end neural models instead score entire spans and their antecedents jointly, avoiding hard-coded mention boundaries.

## Types of Coreference

- **Pronominal**: pronouns like "he", "it", "they" referring back to a noun phrase.
- **Nominal**: a different noun phrase referring to the same entity ("the company" ... "the startup").
- **Zero anaphora**: languages like Japanese or Chinese often omit the subject entirely, requiring the model to infer an implicit referent.

## Why It's Hard

Winograd Schema-style sentences illustrate the difficulty:

```text
The trophy doesn't fit in the suitcase because it is too big.
The trophy doesn't fit in the suitcase because it is too small.
```

Swapping one word changes which entity "it" refers to, and resolving it correctly requires world knowledge about the physical relationship between trophies and suitcases—not just syntax. Long documents with many entities, nested clauses, and cataphora (a pronoun appearing before its referent) add further difficulty.

## Why It Matters

Coreference resolution underlies many downstream tasks: summarization needs to know which sentences describe the same entity, question answering needs to trace pronouns back to their source, and information extraction needs consistent entity identity across a document. In large language models, coreference is handled implicitly through attention rather than an explicit clustering step, but it remains a useful diagnostic: probing whether a model resolves a pronoun correctly reveals how well it tracks entities across long context.
