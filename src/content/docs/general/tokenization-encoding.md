---
title: "Tokenization and Encoding: How Text Becomes Numbers"
description: "Understanding how LLMs convert text into tokens and why token counting matters for cost and performance."
---

When you send "Hello, world!" to an LLM, it doesn't process text directly. First, it breaks the text into tokens, then converts each token to a numerical embedding. This process—tokenization—is fundamental to how LLMs work.

## What Is a Token?

A token is typically a word, subword, or character. Most tokenizers use subword tokenization for efficiency.

**Example with GPT's tokenizer:**
- "Hello" → 1 token
- "world" → 1 token
- "!" → 1 token
- Total: 3 tokens

**But sometimes:**
- "Tokenization" → 1 token
- "running" → 2 tokens (run + ing)
- "💀" → 1 token
- "2024" → 1 token
- "..." → 1 token

## Why Tokenization Matters

1. **Cost:** You pay per token. A bad tokenizer wastes money
2. **Context Windows:** Everything counts toward your context limit
3. **Performance:** Rare characters/languages tokenize poorly (more tokens)
4. **Latency:** More tokens = slower generation

## Common Tokenizers

**Byte Pair Encoding (BPE)** - Used by GPT models
- Learns the most common byte pairs in training data
- Efficient for English
- Struggles with rare languages and characters

**WordPiece** - Used by BERT
- Similar to BPE
- Designed for English NLP tasks

**SentencePiece** - Used by T5, LLaMA
- Language-agnostic
- Works well across languages
- Better for non-Latin scripts

**Tiktoken** - OpenAI's tokenizer
- Optimized for GPT models
- Efficient for code (handles special characters well)

## Token Efficiency Tips

```
"Write me a poem" = ~4 tokens (efficient)

"Compose an original poem consisting of no fewer than 12 lines 
utilizing sophisticated poetic devices including metaphor, alliteration,
and assonance, adhering to a specific rhyme scheme..." = ~50 tokens

Both ask for a poem, but the second wastes tokens.
```

**Techniques to reduce tokens:**
- Be concise: "Summarize" vs. "Create a succinct summary"
- Use examples: 1 example might replace 10 tokens of explanation
- Compress instructions: Remove redundancy

## Token Counting for Different Languages

| Language | Example | Tokens | Notes |
|----------|---------|--------|-------|
| English | "Hello world" | 2 | Efficient |
| Chinese | "你好世界" (4 chars) | 4-6 | Less efficient |
| Arabic | "مرحبا بالعالم" | 5-7 | Variable efficiency |
| Code | `for i in range(10):` | 6 | Handled well |

## Practical Token Math

```
GPT-4 Pricing Example:
- Input: $0.03 per 1K tokens
- Output: $0.06 per 1K tokens

If you send:
- Prompt: 500 tokens → $0.015
- LLM generates: 200 tokens → $0.012
- Total: $0.027 per request

Scale to 10,000 requests/day: ~$270/day = $8,100/month
```

## Advanced Concepts

**Tokenizer Vocabulary Size:**
- Larger vocabulary: Better compression, but slower decoding
- GPT-4 uses ~100k tokens
- LLaMA uses ~32k tokens

**Special Tokens:**
- `<|begin_of_text|>`, `<|end_of_text|>` - Mark boundaries
- `<|system|>`, `<|user|>`, `<|assistant|>` - Role markers
- These don't appear in user-facing responses but consume context

**Encoding Implications:**
Different models tokenize the same text differently, so pricing and context limits vary by model.