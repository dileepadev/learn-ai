---
title: TF-IDF and Bag-of-Words - Classic Text Representations
description: Understand bag-of-words and TF-IDF weighting, still the fastest and most interpretable baseline for text search and classification.
---

Before embeddings, text was represented as sparse vectors counting word occurrences. Bag-of-words and TF-IDF remain strong, cheap baselines and still power parts of production search systems.

## Bag-of-Words

A document becomes a vector over the vocabulary, where each entry is the count of that word in the document. Word order is discarded entirely — "dog bites man" and "man bites dog" produce the same vector.

```text
Vocabulary: [dog, bites, man, cat]
"dog bites man" -> [1, 1, 1, 0]
```

## TF-IDF Weighting

Raw counts overweight common words like "the" and "is." Term Frequency-Inverse Document Frequency (TF-IDF) rescales each count by how rare the word is across the whole corpus:

```text
tfidf(t, d) = tf(t, d) * log(N / df(t))
```

`tf(t, d)` is the term's frequency in document `d`, `N` is the total number of documents, and `df(t)` is the number of documents containing the term. Words that appear in nearly every document get pushed toward zero weight; words that appear in few documents but repeatedly within one document get high weight.

## Why It Still Matters

TF-IDF vectors are fast to compute, require no training, and produce interpretable weights — you can point to exactly which words drove a similarity score. BM25, the ranking function behind most classic search engines and still a strong retrieval baseline in RAG systems, is a refinement of TF-IDF that adds document-length normalization and saturates term frequency.

## Practical Guidance

Use TF-IDF or BM25 as a fast lexical-retrieval baseline before adding a dense embedding retriever, and consider hybrid retrieval (combining BM25 scores with embedding similarity) since lexical matching catches exact keyword and rare-term matches that embeddings sometimes miss. For small labeled datasets, a TF-IDF vector plus logistic regression is often a competitive, cheap-to-run text classifier compared to fine-tuning a transformer.
