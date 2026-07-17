---
title: Machine Translation - Translating Meaning Across Languages
description: Explore neural machine translation, evaluation beyond BLEU, and practical steps for building reliable multilingual systems.
---

Machine translation converts text from a source language to a target language while preserving meaning, tone, and relevant context. Modern systems usually use encoder-decoder transformers trained on parallel text.

## Encoder-Decoder Translation

The encoder turns source tokens into contextual representations. The decoder generates one target token at a time while attending to those representations:

```text
source sentence -> encoder -> contextual states
target prefix + states -> decoder -> next target token
```

Subword tokenization helps models handle rare words and scripts with large vocabularies. Multilingual models share parameters across languages, allowing knowledge transfer to lower-resource language pairs.

## Quality Is More Than Word Overlap

BLEU measures n-gram overlap with reference translations and remains useful for quick comparisons, but it misses many valid phrasings. COMET and human evaluation better capture adequacy and fluency. Evaluate terminology, formality, named entities, numbers, and content omissions separately.

## Common Failure Modes

- translating a word correctly but choosing the wrong sense
- dropping negation, dates, or measurements
- using an inappropriate level of formality
- amplifying bias present in the training corpus
- performing poorly for dialects or low-resource languages

## Production Practices

Build a test set from actual content and all supported locales. Maintain a glossary for product names and domain terms, preserve placeholders before translation, and display source text when users need to verify high-stakes output. Translation systems are helpful for communication, but legal, medical, and safety-critical material needs qualified human review.

