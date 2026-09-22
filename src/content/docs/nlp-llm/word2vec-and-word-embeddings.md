---
title: Word2Vec and Classic Word Embeddings
description: Learn how word2vec, GloVe, and related methods learn dense vector representations of words from co-occurrence statistics.
---

Before contextual embeddings from transformers, NLP systems represented words as fixed, context-independent vectors learned from large corpora. Word2Vec and GloVe are the two most influential methods.

## Word2Vec

Word2Vec learns embeddings by predicting context from a target word (skip-gram) or a target word from its context (continuous bag-of-words, CBOW).

```text
Skip-gram: "the cat sat on the mat"
target = "sat" -> predict {"cat", "on"} within a context window
```

Training uses negative sampling: for each true (word, context) pair, sample several random (word, noise) pairs and train the model to score true pairs higher. This avoids computing a full softmax over the vocabulary at every step.

## GloVe

GloVe (Global Vectors) instead factorizes a word-word co-occurrence matrix built from the whole corpus, so word similarity directly reflects how often two words appear near each other across the entire dataset rather than in local windows sampled during training.

## What the Vectors Capture

Both methods produce vectors where semantic and even some syntactic relationships appear as consistent directions: `vector("king") - vector("man") + vector("woman")` lands close to `vector("queen")`. This arithmetic works because co-occurrence patterns for gender-related word pairs are systematically similar across many word pairs.

## Limitations and Why Contextual Embeddings Replaced Them

Word2Vec and GloVe assign one vector per word regardless of sense, so "bank" gets a single vector blending "riverbank" and "financial institution." Contextual embeddings from transformer encoders (BERT-style) produce a different vector for each occurrence of a word based on its sentence, resolving this ambiguity and generally outperforming static embeddings on downstream tasks.

## Practical Guidance

Static embeddings are still useful when you need a lightweight, CPU-friendly representation for large-scale similarity search, or as an interpretable baseline before adopting a heavier transformer-based pipeline. Pretrained GloVe and word2vec vectors are widely available and require no fine-tuning to get reasonable results on simple similarity or clustering tasks.
