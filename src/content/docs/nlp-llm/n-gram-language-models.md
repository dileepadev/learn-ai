---
title: N-Gram Language Models - The Predecessor to Neural LMs
description: Learn how n-gram language models estimate next-word probabilities from counts, and why understanding them clarifies what neural LMs improved on.
---

Before neural language models, text generation and scoring relied on n-gram models: statistical models that estimate the probability of a word given the previous `n-1` words, using counts from a training corpus.

## The Core Idea

A bigram model (n=2) estimates the probability of a word given only the immediately preceding word:

```text
P(word_i | word_{i-1}) = count(word_{i-1}, word_i) / count(word_{i-1})
```

A trigram model conditions on the previous two words, and so on for higher `n`. The full sentence probability is the product of these conditional probabilities across all positions, invoking the Markov assumption that only the last `n-1` words matter for predicting the next one.

## The Sparsity Problem

Most n-grams, especially for higher `n`, never appear in the training corpus, giving them zero probability and breaking the product for any sentence containing them. Smoothing techniques — Laplace smoothing, Kneser-Ney smoothing — redistribute probability mass from seen to unseen n-grams so the model never assigns exactly zero probability to a valid sentence.

## Why Neural Models Won

N-gram models cannot capture dependencies beyond their fixed window, so "The trophy didn't fit in the suitcase because it was too big" — where resolving "it" requires reasoning far beyond a 2- or 3-word window — is out of reach. Neural language models, especially transformers with full-sequence attention, condition on arbitrarily long context and generalize across similar words via learned embeddings rather than requiring exact n-gram matches.

## Where N-Grams Still Matter

N-gram models remain useful as extremely fast, low-resource baselines for language identification, spelling correction candidate generation, and as a component in some speech recognition decoders where a lightweight local model is combined with acoustic scores. They also underlie n-gram-based text similarity metrics like BLEU, which score generated text by n-gram overlap with reference text.

## Practical Guidance

Understand n-gram models as the historical baseline that motivates key ideas in modern LMs — the Markov assumption, smoothing, and perplexity as an evaluation metric all originate here. Reach for an n-gram model only when you need something that trains in seconds on CPU with no GPU or embedding infrastructure.
