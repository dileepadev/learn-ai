---
title: Relation Extraction - Turning Text into Structured Facts
description: Learn how relation extraction identifies links between entities, how schemas guide models, and how to validate extracted facts.
---

Relation extraction identifies semantic links between entities in text and turns prose into structured records.

```text
"Marie Curie worked at the University of Paris."

(Marie Curie, employed_by, University of Paris)
```

It enables searchable knowledge graphs, research databases, contract analysis, and question-answering systems.

## Relations Need a Schema

Before modeling, define allowed entity types and relations:

```text
PERSON -- employed_by --> ORGANIZATION
DRUG   -- treats      --> CONDITION
COMPANY -- acquired   --> COMPANY
```

A narrow, well-defined schema usually produces more useful results than a vague “extract all facts” request. Define direction, cardinality, time qualifiers, and evidence requirements.

## Modeling Approaches

Pipeline systems run NER first and classify each candidate entity pair. Joint models learn entities and relations together, reducing error propagation. LLM-based extraction can produce structured JSON for varied schemas, but it needs validation and grounding because a plausible relation may not be stated in the source.

## Evaluation

An extracted relation is correct only if the subject, predicate, object, and required qualifiers match the annotated fact. Report precision, recall, and F1 by relation type. Also inspect whether the model confuses negation, speculation, historical facts, and relations that are true in the world but absent from the text.

## Reliable Pipelines

Store the source span for every extracted fact, validate output against a schema, normalize entity names, and preserve confidence. For high-stakes domains, present extracted relations as reviewable evidence rather than silently updating a database. The goal is not to create the most triples; it is to create traceable, correct facts.

