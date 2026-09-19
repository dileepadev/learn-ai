---
title: Constituency Parsing - Building Phrase-Structure Trees
description: Learn how constituency parsing groups words into nested phrases, how it differs from dependency parsing, and when it is still useful.
---

Constituency parsing analyzes a sentence's phrase structure, grouping words into nested constituents such as noun phrases and verb phrases, and represents the result as a tree.

```text
(S
  (NP (DET The) (ADJ quick) (NOUN fox))
  (VP (VERB jumps)
      (PP (ADP over) (NP (DET the) (ADJ lazy) (NOUN dog)))))
```

Each internal node is a phrase category (NP, VP, PP); each leaf is a word with its part of speech. The tree captures which words group together as a syntactic unit, independent of any direct word-to-word relationship.

## Constituency vs. Dependency Parsing

Dependency parsing (see [[named-entity-recognition]]'s sibling topic, dependency parsing) directly connects words with labeled relations like "subject of" or "modifier of," producing a flatter, word-to-word graph. Constituency parsing instead builds a hierarchical phrase structure, which is closer to traditional grammar-school sentence diagramming and is the representation required by formal grammar theories like context-free grammars.

Most modern NLP pipelines favor dependency parsing because it maps more directly onto semantic relations useful for extraction tasks, and dependency treebanks are more available across languages with flexible word order.

## Approaches

Classic constituency parsers used probabilistic context-free grammars (PCFGs) with dynamic programming (the CKY algorithm) to find the most likely tree. Modern neural constituency parsers use a transformer encoder to score candidate spans directly, then run a CKY-style decoding step to assemble the highest-scoring tree consistent with those span scores.

## Practical Guidance

Reach for constituency parsing when you need explicit phrase boundaries — for example, extracting all noun phrases as candidate entities, or building syntax-aware paraphrase and simplification systems. For most modern extraction and relation-based tasks, dependency parsing or a fine-tuned span-extraction model is simpler to integrate and sufficiently expressive.
