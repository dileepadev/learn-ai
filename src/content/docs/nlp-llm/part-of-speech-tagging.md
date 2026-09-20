---
title: Part-of-Speech Tagging - Labeling Words by Grammatical Role
description: Learn how part-of-speech tagging assigns grammatical categories to words and why it still matters as a building block for downstream NLP.
---

Part-of-speech (POS) tagging assigns each word in a sentence a grammatical category, such as noun, verb, adjective, or preposition. It is one of the oldest tasks in NLP and still underpins parsing, information extraction, and grammar-aware text generation.

```text
The   quick brown fox jumps over the lazy dog
DET   ADJ   ADJ   NOUN VERB  ADP  DET NOUN VERB
```

## Why Tags Are Ambiguous

Many words take different tags depending on context. "Book" is a noun in "read a book" but a verb in "book a flight." Rule-based taggers fail on these cases because they ignore surrounding words; statistical and neural taggers succeed because they condition on context.

## Approaches

Hidden Markov Models (HMMs) treat tagging as finding the most likely tag sequence given a word sequence, using transition probabilities between tags and emission probabilities from tags to words. The Viterbi algorithm finds the optimal sequence efficiently.

Modern taggers use a transformer encoder to produce contextual embeddings for each token, followed by a linear classifier per token. This removes the need for hand-built emission tables and generalizes better to unseen words.

## Tag Sets

Universal Dependencies defines a cross-lingual tag set (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, and others) so taggers trained on one language's grammar concepts transfer more easily to another. The older Penn Treebank tag set is finer-grained (NN, NNS, VBD, VBG, JJ) and still common in English-only tools.

## Practical Guidance

Use spaCy or a Universal Dependencies-trained transformer for production tagging rather than training from scratch. Evaluate on your actual domain: tagging accuracy on news text does not guarantee accuracy on social media, code comments, or clinical notes, where slang, abbreviations, and unconventional punctuation break tokenization assumptions. POS tags remain useful as features for rule-based extraction, grammar checking, and as an interpretable intermediate signal even in an LLM-first pipeline.
