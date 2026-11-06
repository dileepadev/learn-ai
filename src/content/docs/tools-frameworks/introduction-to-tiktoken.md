---
title: Introduction to tiktoken - Counting and Encoding Tokens
description: Learn how to use the tiktoken library to encode text into byte-pair tokens, estimate prompt costs, and enforce context window limits.
---

tiktoken is a fast byte-pair-encoding (BPE) tokenizer library originally released for OpenAI models. Counting tokens accurately matters because pricing, context limits, and truncation are all measured in tokens, not characters or words.

## Encoding and Decoding

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode("Tokenization splits text into subword units.")
print(tokens)            # list of integer token ids
print(len(tokens))       # token count
print(enc.decode(tokens))
```

Different models use different encodings, so choose the one matching your target model:

```python
enc = tiktoken.encoding_for_model("gpt-4o")
```

## Why Token Counts Differ from Word Counts

Common English words are often a single token, but rare words, numbers, code, and non-Latin scripts split into more tokens. As a rough rule, English averages about 0.75 words per token, though this varies widely by language and content.

## Practical Uses

- **Cost estimation**: multiply input and output token counts by per-token prices
- **Context budgeting**: reserve space for the system prompt, retrieved context, and the response
- **Chunking for RAG**: split documents by token length rather than characters
- **Truncation**: cut history safely without exceeding the context window

## A Truncation Helper

```python
def truncate_to_tokens(text, max_tokens, enc):
    ids = enc.encode(text)
    return enc.decode(ids[:max_tokens])
```

## Caveats

- Chat formats add per-message overhead tokens, so counts from raw text slightly underestimate real usage.
- tiktoken matches OpenAI-family tokenizers; for other providers use their own tokenizer or token-counting endpoint.
- Tokenization of a concatenated string can differ from the sum of its parts.
