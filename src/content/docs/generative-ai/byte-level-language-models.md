---
title: Byte-Level and Token-Free Language Models
description: How language models can operate directly on bytes, characters, or pixels instead of subword tokens — covering the limitations of tokenization, byte-level BPE, character-level models, MegaByte, CANINE, and the implications of removing the tokenization bottleneck.
---

**Byte-level and token-free models** are a family of language models that bypass the standard tokenization pipeline — operating directly on raw bytes, Unicode characters, or pixels rather than on a fixed vocabulary of subword tokens produced by algorithms like BPE (Byte Pair Encoding) or SentencePiece. They challenge the assumption, dominant since GPT-2, that a learned vocabulary is a necessary precondition for high-quality language modeling.

## The Tokenization Bottleneck

Nearly all modern LLMs use **subword tokenization**: text is pre-processed into a vocabulary of 32K–128K tokens before the model ever sees it. This design choice has significant consequences.

### Advantages of Subword Tokenization

- **Efficiency**: Representing "transformer" as one token rather than 11 bytes reduces sequence length, lowering quadratic attention costs.
- **Vocabulary coverage**: BPE and SentencePiece handle any language or domain by falling back to character-level representations for unknown words.
- **Semantic units**: Common words and morphemes often align with token boundaries, giving the model useful inductive structure.

### The Hidden Costs of Tokenization

**Language bias**: Tokenizers trained predominantly on English are inefficient for other languages. The word "hello" in English is 1–2 tokens; equivalent words in many low-resource languages can be 5–15 tokens, making inference 5–10× more expensive and degrading quality.

**Arithmetic fragility**: Numbers like "12345" may be tokenized as "123" + "45" or "1" + "2" + "3" + "45" depending on the tokenizer — making numerical reasoning inconsistent.

**Sensitivity to whitespace and capitalization**: Adding a space before a word often changes its token ID. "Bank" and " Bank" may be different tokens. The model must learn to handle these artifacts.

**No access to subword structure**: The model cannot see the individual characters inside a token. Spelling correction, morphological analysis, and cross-lingual transfer are harder when character-level access is absent.

**Adversarial vulnerability**: Carefully constructed tokenizer edge cases (unusual Unicode, zero-width characters, long token sequences) can cause unpredictable model behavior.

## Byte-Level BPE

**Byte-level BPE** (used by GPT-2, RoBERTa, and the GPT family) applies the BPE algorithm at the byte level rather than the character level:

- The base vocabulary is the 256 possible byte values.
- BPE merges are learned over byte sequences, producing a vocabulary of byte n-grams.
- This guarantees any Unicode string can be tokenized without unknown tokens — every byte has a representation.

**Advantages over character-level BPE**: Language-agnostic; handles emojis, math symbols, and arbitrary Unicode without special cases.

**Still subject to inefficiencies**: Rare languages and scripts produce long byte sequences before merging kicks in.

## Character-Level Models

**Character-level models** treat each character as an atomic unit, completely avoiding the vocabulary question. Early neural language models (Karpathy's char-rnn) operated this way.

**Challenges:**

- **Long sequences**: Character sequences are 3–5× longer than equivalent token sequences, making attention quadratically more expensive.
- **Learning higher-level abstractions**: The model must learn word and phrase boundaries entirely from data, without any structural prior.
- **Slower convergence**: More training steps are needed to learn the same semantic content.

Modern character-level models address sequence length with hierarchical architectures rather than processing characters flatly through a transformer.

## CANINE (Google, 2021)

**CANINE** (Character Architecture with No InnEring) is a token-free transformer encoder designed for multilingual NLP tasks. It operates directly on Unicode codepoints and addresses the sequence length problem with a **downsampling strategy**:

1. **Local attention over characters**: Shallow transformer layers process characters with local attention windows.
2. **Stride convolution for downsampling**: Character representations are downsampled to a shorter sequence of "sentence-piece-like" representations via strided convolutions.
3. **Deep transformer over downsampled sequence**: Standard self-attention over the reduced sequence.
4. **Upsampling for character-level output**: For tasks requiring character-level output, representations are upsampled back.

CANINE matches or exceeds subword-based multilingual BERT on cross-lingual benchmarks, demonstrating that tokenizer-free architectures can compete with tokenized baselines.

## MegaByte (Meta, 2023)

**MegaByte** is a byte-level autoregressive language model that addresses the efficiency problem through a **two-level hierarchical architecture**:

### Architecture

```text
Input bytes: b₁ b₂ b₃ ... bN

Step 1: Chunk bytes into patches of K bytes each
  Patch₁ = [b₁...bK],  Patch₂ = [bK+1...b2K], ...

Step 2: Global Model (large transformer)
  Operates over patches, producing patch-level context vectors
  → O((N/K)²) attention complexity

Step 3: Local Model (small transformer)
  Within each patch, autoregressively generates K bytes
  Conditioned on the global context vector for that patch
  → O(K²) per patch, O(N·K) total

Step 4: Output = byte sequence
```

### Why MegaByte Works

The global model handles **long-range dependencies** over compressed patch representations. The local model handles **fine-grained byte generation** within each patch. The two-level design reduces the overall complexity from $O(N^2)$ (naive byte-level) to approximately $O((N/K)^2 + NK)$.

**Results**: MegaByte matches GPT-3.5-class models on language tasks while operating on raw bytes, without any tokenization. It also naturally handles arbitrary binary data — images, audio, code, and multilingual text through a single unified model.

## Pixel-Level Language Models: Bytes for Vision

Taking the token-free idea further, **ByT5** (Google, 2021) applies the T5 encoder-decoder architecture directly to byte sequences for text tasks. **Pixel** and similar models apply the byte-level principle to text rendered as images — treating text as visual pixels and building language models over pixel sequences.

These approaches are primarily research probes, but they demonstrate the generality of token-free thinking across modalities.

## ByT5

**ByT5** is a T5 model that processes raw UTF-8 bytes directly, using a vocabulary of just 256 byte values plus a few special tokens. Key findings:

- Competitive with SentencePiece-tokenized T5 on most downstream tasks.
- Significantly better on **character-level tasks**: spelling correction, morphological analysis, text normalization.
- Better generalization to **low-resource and code-switching languages** that are poorly served by standard tokenizers.
- Slower in terms of training and inference due to longer sequences — partially offset by smaller parameter counts (fewer embedding parameters).

## Trade-offs Summary

| Aspect | Subword (BPE/SentencePiece) | Byte-Level | Token-Free (CANINE, MegaByte) |
| --- | --- | --- | --- |
| Sequence length | Shortest | Longer | Medium (with hierarchical compression) |
| Language fairness | Biased to tokenizer training language | Equal across bytes | Equal across bytes |
| Arithmetic | Inconsistent | Consistent | Consistent |
| Spelling/morphology | Limited | Full access | Full access |
| Training efficiency | High | Low | Medium |
| Inference speed | Fastest | Slowest (naive) | Competitive (hierarchical) |

## Practical Implications

Modern tokenizer design is increasingly aware of the problems byte-level research has surfaced:

- **Tiktoken** (OpenAI's tokenizer) uses byte-level BPE and carefully balanced multilingual data to reduce the language bias of earlier tokenizers.
- **LLaMA 3's tokenizer** expanded vocabulary to 128K to better represent diverse languages and reduce byte-level fragmentation.
- **Gemma** introduced sentencepiece tokenizers with improved multilingual coverage.

Rather than full tokenizer removal, the practical near-term outcome may be **better tokenizers** informed by byte-level research — combined with niche use of fully token-free models for specific tasks like spelling correction, multilingual OCR, and universal binary data processing.

The long-term trajectory points toward hybrid hierarchical architectures that compress efficiently while retaining byte-level fidelity where it matters — following MegaByte's blueprint.
