---
title: Named Entity Recognition - Extracting People, Places, and Organizations
description: Learn how NER identifies spans of text, labels entity types, and is evaluated in real information-extraction workflows.
---

Named entity recognition (NER) finds spans in text that refer to entities and assigns types such as person, organization, location, product, date, or medical concept.

```text
Ada Lovelace joined the Royal Society in London.
[Ada Lovelace: PERSON] [Royal Society: ORGANIZATION] [London: LOCATION]
```

## Sequence Labeling

Many NER systems use BIO tags:

```text
Ada       B-PER
Lovelace  I-PER
joined    O
```

`B` starts an entity, `I` continues it, and `O` means outside any entity. A transformer encoder produces contextual token representations, then a classifier or conditional random field predicts a valid tag sequence.

## What Makes NER Hard

Entity boundaries and types depend on context. “Apple” can be a company or fruit; “Washington” can be a person, city, or government. Domain language introduces new types and terminology, so a news-trained model will not automatically work well on clinical notes or contracts.

## Evaluation

Measure entity-level precision, recall, and F1. A predicted entity counts as correct only when its span and type match the annotation rule. Inspect per-type scores and boundary errors rather than relying on a single aggregate F1.

## Practical Guidance

Define an annotation guide before collecting labels: clarify whether job titles, nested entities, pronouns, and abbreviations count. Start with a pretrained language model, fine-tune on representative data, and include difficult examples from production. If entities contain personal data, use access controls, redact where possible, and evaluate errors that could expose or misclassify sensitive information.

