---
title: Stemming and Lemmatization - Reducing Words to Their Base Form
description: Compare stemming and lemmatization for normalizing word forms, and see where each still helps in modern NLP pipelines.
---

Stemming and lemmatization both reduce inflected words to a common base form so that "running," "runs," and "ran" can be treated as related to "run." They differ in method and precision.

## Stemming

Stemming applies crude, rule-based suffix stripping without understanding grammar. The Porter stemmer, the most widely used algorithm, applies a sequence of suffix-removal rules:

```text
"running"    -> "run"
"argument"   -> "argu"
"university" -> "univers"
```

Stemming is fast and requires no dictionary, but it produces non-words ("argu," "univers") and sometimes conflates unrelated words that happen to share a suffix pattern.

## Lemmatization

Lemmatization uses a vocabulary and morphological analysis (often combined with POS tagging) to return the dictionary base form, or lemma:

```text
"running" (verb) -> "run"
"better"  (adj)  -> "good"
"was"     (verb) -> "be"
```

Lemmatization is more accurate because it accounts for part of speech and irregular forms, but it is slower and requires a language-specific lexicon or model.

## Do You Still Need Either?

Modern subword tokenizers (BPE, WordPiece, SentencePiece) already split "running" into pieces that share a subword with "run," so transformer-based models learn morphological relationships implicitly without explicit stemming or lemmatization. These techniques remain relevant for classic bag-of-words or TF-IDF pipelines, keyword-based search indexes, and rule-based text matching where reducing vocabulary size and merging word variants directly improves recall.

## Practical Guidance

Use lemmatization over stemming whenever accuracy matters more than speed, since stemming errors compound in downstream tasks like search relevance. Skip both entirely for transformer fine-tuning or embedding-based retrieval — normalizing text this aggressively can remove signal the model would otherwise use.
