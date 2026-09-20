---
title: Keyphrase Extraction - Identifying the Terms That Matter Most
description: Learn how keyphrase extraction identifies the most representative terms in a document, from graph-based ranking to modern embedding methods.
---

Keyphrase extraction identifies the words and short phrases that best represent a document's core content — used for indexing, tagging, search engine optimization, and summarizing what a document is "about" in a handful of terms.

## Statistical Baselines

TF-IDF, covered in [[tf-idf-and-bag-of-words]], provides a simple keyphrase baseline: rank candidate n-grams by their TF-IDF score within the document relative to a background corpus, and take the top-scoring terms as keyphrases. This is fast and requires no training but ignores how candidate phrases relate to each other structurally within the document.

## Graph-Based Ranking

TextRank, adapted from the PageRank algorithm, builds a graph where candidate words or phrases are nodes and edges connect terms that co-occur within a fixed window in the document, then ranks nodes by a centrality score similar to how PageRank ranks web pages by how many other important pages link to them:

```text
build co-occurrence graph over candidate terms in the document
run PageRank-style centrality algorithm on this graph
top-ranked nodes -> keyphrases
```

Terms that co-occur frequently with many other important terms score highly, capturing a notion of "structural importance" within the document that pure frequency-based methods miss.

## Embedding-Based Extraction

Modern approaches like KeyBERT embed candidate n-grams and the whole document with a sentence embedding model, then rank candidates by cosine similarity to the document's overall embedding — phrases whose embedding is closest to the document's embedding are judged most representative of its content, capturing semantic relevance rather than only frequency or co-occurrence structure.

## Supervised and LLM-Based Extraction

Supervised keyphrase extraction trains a sequence-labeling model on documents with human-annotated keyphrases, which can learn domain-specific notions of importance (technical terms in scientific papers, product features in reviews) that unsupervised methods miss. Prompting an LLM directly to extract keyphrases is now a common, flexible alternative, and can be steered with instructions about desired phrase length, specificity, or category (extract only technical terms, only named entities, only product features) without retraining anything.

## Practical Guidance

Use TextRank or KeyBERT for fast, unsupervised keyphrase extraction at scale (tagging a large document corpus) where training data isn't available. Use an LLM-based approach when you need controllable, instruction-following extraction behavior, or when documents are short enough that per-document LLM calls are cost-effective.
