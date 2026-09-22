---
title: Text Preprocessing and Normalization for NLP Pipelines
description: A practical guide to cleaning, normalizing, and preparing raw text before tokenization, embedding, or model training.
---

Raw text is messy: inconsistent casing, stray whitespace, encoding artifacts, HTML remnants, and Unicode variants of the same character all degrade downstream model quality if left unhandled.

## Common Normalization Steps

Unicode normalization (NFC or NFKC) collapses visually identical but differently encoded characters — an accented letter typed as one composed codepoint versus a base letter plus a combining accent — into a single canonical form. Case folding lowercases text for tasks where casing carries no signal, but should be skipped for tasks like named entity recognition where capitalization is a strong feature. Whitespace and punctuation normalization strips redundant spaces, normalizes smart quotes and dashes, and removes control characters left over from PDF or HTML extraction.

```text
Raw:   "The café’s Wi‑Fi   is  down again."
Clean: "The cafe's Wi-Fi is down again."
```

## What Modern Pipelines Skip

Classic NLP pipelines often included aggressive steps — removing all punctuation, stripping stop words, stemming every token — that hurt modern subword tokenizers and transformer models, which rely on punctuation and word boundaries as informative signal. For LLM and embedding pipelines, minimal normalization (fix encoding, strip boilerplate, normalize whitespace) usually outperforms aggressive classic preprocessing.

## Domain-Specific Cleaning

Web-scraped text needs boilerplate removal (navigation menus, cookie banners, footers) before it is useful as training or retrieval data. OCR output needs correction for common character-substitution errors. Chat and social text needs handling for emoji, hashtags, and non-standard spelling that carries meaning and should often be preserved rather than stripped.

## Practical Guidance

Log a sample of before/after transformed text at each preprocessing step in a new pipeline — silent normalization bugs (like double-encoding UTF-8 or lowercasing text before entity extraction) are hard to spot from aggregate metrics alone. Match your preprocessing to your downstream model: a modern subword tokenizer needs far less cleanup than a bag-of-words or TF-IDF pipeline.
