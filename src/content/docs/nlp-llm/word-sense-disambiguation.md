---
title: Word Sense Disambiguation - Picking the Right Meaning in Context
description: Learn how word sense disambiguation resolves ambiguous words to their correct meaning and why contextual embeddings largely subsumed it.
---

Word sense disambiguation (WSD) determines which meaning of an ambiguous word applies in a given context. "Bank" can mean a financial institution or the side of a river; "bass" can mean a fish or a low musical pitch.

```text
"He deposited the check at the bank."       -> bank (financial institution)
"They walked along the bank of the river."  -> bank (edge of a waterway)
```

## Classic Approaches

Knowledge-based methods use a lexical resource like WordNet, which enumerates word senses with glosses (short definitions) and relations between senses, then pick the sense whose gloss overlaps most with the surrounding context (the Lesk algorithm). Supervised methods train a classifier on sense-tagged corpora to predict the correct WordNet sense ID given a word and its context, but sense-tagged data is expensive and limited in coverage.

## Why Contextual Embeddings Changed the Picture

Static word embeddings like word2vec assign one vector per word form regardless of sense, forcing a separate WSD step if sense mattered downstream. Contextual embeddings from transformer encoders naturally produce a different vector for "bank" depending on its sentence, implicitly resolving sense ambiguity as a side effect of contextualization rather than as an explicit classification task. This is a major reason explicit WSD is now a niche task rather than a standard pipeline stage.

## Where Explicit WSD Still Applies

Applications that need to output an interpretable sense label — machine translation systems disambiguating a word before selecting the correct target-language word, or lexicography and linguistic annotation tools — still rely on WSD models tied to a fixed sense inventory like WordNet or BabelNet.

## Practical Guidance

For most modern NLP applications, rely on a contextual embedding model or an LLM to implicitly handle sense ambiguity rather than adding an explicit WSD stage. Add explicit WSD only when you need a labeled sense from a specific inventory for downstream symbolic processing, such as linking text to a structured knowledge base.
