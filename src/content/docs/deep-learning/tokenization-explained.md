---
title: "Tokenization Explained: How LLMs See Text"
description: "Understand how tokenizers convert text into tokens, why tokenization choices affect model performance, and the differences between BPE, WordPiece, and SentencePiece algorithms."
---

Before an LLM can process text, that text must be converted into a sequence of integers — tokens. Tokenization is one of the most foundational and least-discussed parts of how language models work. It has surprising implications for model performance, cost, and failure modes.

## What Is a Token?

A token is the basic unit of text that a model processes. Tokens are not words — they're subword units that can be whole words, word fragments, punctuation, or whitespace.

For example, the sentence "Tokenization is fascinating!" might become:

```
["Token", "ization", " is", " fasci", "nating", "!"]
→ [15467, 2065, 318, 3382, 1989, 0]
```

The vocabulary size (number of unique tokens) for modern LLMs is typically 32,000–128,000.

## Byte Pair Encoding (BPE)

BPE is the most widely used tokenization algorithm (used by GPT-2, GPT-4, LLaMA, Mistral).

**Training algorithm**:
1. Start with a vocabulary of individual characters (or bytes).
2. Count all adjacent pairs of symbols in the training corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the vocabulary reaches the target size.

The result: common words become single tokens, rare words are split into subword pieces, and unknown words are handled gracefully by falling back to character-level pieces.

**Byte-level BPE** (used by GPT-2 and its descendants) operates on raw bytes rather than Unicode characters, guaranteeing that any input can be tokenized without unknown tokens.

## WordPiece

Used by BERT and its derivatives. Similar to BPE but uses a different merge criterion: instead of merging the most frequent pair, it merges the pair that maximizes the likelihood of the training data under a language model.

WordPiece marks subword continuations with `##`:
```
"unbelievable" → ["un", "##believe", "##able"]
```

## SentencePiece

Used by T5, LLaMA (via its tokenizer), and many multilingual models. Key differences:
- Treats the input as a raw character stream, including whitespace.
- Whitespace is represented as a special character (▁), making tokenization language-agnostic.
- Supports both BPE and unigram language model algorithms.
- Fully reversible: you can always reconstruct the original text exactly.

## Why Tokenization Matters for Performance

### Arithmetic and Spelling
Numbers are tokenized inconsistently. "100" might be one token, "101" might be two. This is why LLMs struggle with digit-level arithmetic — the model never sees individual digits as atomic units.

Similarly, spelling tasks are hard because the model doesn't see individual letters — it sees subword chunks.

### Non-English Languages
Most tokenizers are trained predominantly on English text. Non-English languages, especially those with non-Latin scripts, are tokenized less efficiently — requiring more tokens to represent the same amount of information. This means:
- Higher cost per query.
- Shorter effective context in non-English languages.
- Potentially worse performance due to less training signal per token.

### Code
Code tokenizers need to handle indentation, special characters, and language-specific syntax. Models trained with code-aware tokenization (like CodeLlama) perform better on code tasks.

## Token Count and Cost

API pricing is per token. Understanding tokenization helps you estimate and control costs:

- English prose: ~1 token per 4 characters, or ~0.75 tokens per word.
- Code: varies widely; dense code can be 1 token per 2–3 characters.
- JSON: often inefficient due to repeated structural characters.

```python
import tiktoken  # OpenAI's tokenizer library

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(len(tokens))  # 6
```

## Tokenizer-Free Models

An emerging research direction: byte-level or character-level models that skip tokenization entirely. **MegaByte** and **ByT5** operate at the byte level. The tradeoff is longer sequences (more positions to attend over) in exchange for better handling of rare words, multilingual text, and robustness to typos.

Understanding your tokenizer is a prerequisite for understanding why your model behaves the way it does on edge cases.
