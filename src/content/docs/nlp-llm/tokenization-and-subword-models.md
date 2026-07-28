---
title: Tokenization and Subword Models - How Text Becomes Numbers
description: Explore why modern NLP models tokenize text into subwords, and how BPE, WordPiece, and Unigram differ.
---

Before any neural network can process text, it must be converted into a sequence of discrete IDs. The way this split happens—**tokenization**—directly shapes vocabulary size, sequence length, and how well a model handles rare or unseen words.

## Why Not Just Words or Characters?

```text
word-level:      vocabulary explodes, out-of-vocabulary words break the model
character-level:  vocabulary is tiny, but sequences become very long
subword-level:   balances both — common words stay whole, rare words split
```

Word-level tokenizers cannot handle a word never seen during training. Character-level tokenizers avoid that problem but force the model to learn structure over much longer sequences, which is computationally expensive and harder to optimize. Subword tokenization is the practical middle ground used by nearly all modern language models.

## Byte-Pair Encoding (BPE)

BPE starts with individual characters (or bytes) and iteratively merges the most frequent adjacent pair into a new token, repeating until a target vocabulary size is reached. Common words end up as single tokens, while rare words are split into recognizable pieces, for example:

```text
"unbelievable" -> "un", "believ", "able"
```

## WordPiece and Unigram

WordPiece (used by BERT) is similar to BPE but chooses merges that maximize the likelihood of the training corpus rather than raw frequency. The Unigram model takes the opposite direction: it starts from a large candidate vocabulary and iteratively removes tokens that contribute least to corpus likelihood, keeping the mixture that best balances coverage and compactness. SentencePiece is a common implementation wrapper that supports both BPE and Unigram and treats input as a raw stream, including spaces, so tokenization works consistently across languages without language-specific pre-splitting.

## Practical Consequences

- **Sequence length**: a poorly matched tokenizer can double the number of tokens for a given text, increasing cost and context usage.
- **Fairness across languages**: tokenizers trained mostly on English text often over-fragment other languages, making them more expensive to process per unit of meaning.
- **Numbers and code**: naive tokenizers split digits inconsistently, which can hurt arithmetic and code generation unless digit-level or fixed-width tokenization is used.

Tokenizer choice is fixed at training time and cannot be changed later without retraining embeddings, making it one of the earliest and most consequential decisions in building a language model.
