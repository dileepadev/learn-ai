---
title: Attention Sinks and Streaming LLMs
description: Learn why LLMs assign large attention to the first tokens, and how keeping those sink tokens in the cache enables stable, effectively unbounded streaming generation.
---

Attention sinks are an empirical phenomenon discovered while studying how to run LLMs on streams longer than their training window. The finding made stable streaming inference possible with a fixed-size cache.

## The Observation

In trained autoregressive transformers, the first few tokens in a sequence receive a disproportionate share of attention across many layers and heads, even when they are semantically unimportant (for example, a start token).

## Why Sinks Emerge

Softmax forces attention weights to sum to 1. When a query has no strongly relevant key, the model still has to distribute its attention somewhere. Initial tokens are visible to every later position during training, so they become a convenient place to dump surplus attention. They function as a learned no-op target.

## What Breaks with Window Attention

A naive sliding window keeps only the most recent W tokens in the KV cache. Once the first tokens slide out, the sink disappears, the softmax distribution changes sharply, and perplexity spikes.

## The StreamingLLM Fix

Keep two parts in the cache:

```text
cache = [ first 4 "sink" tokens ] + [ most recent W tokens ]
```

Positions are assigned relative to cache contents rather than original text positions. With this, models stay stable over millions of tokens.

## What It Does and Does Not Do

- It gives constant memory and stable perplexity during streaming
- It does **not** extend the model's real context: evicted middle tokens are forgotten, so it cannot answer questions about distant content

## Training with a Sink

Models can be pretrained with a dedicated learnable sink token, or with softmax variants that allow attention to sum to less than 1, so that real tokens are not forced to serve as sinks. Some recent open models include learned sink parameters in attention.

## Practical Relevance

Attention sinks also explain why quantization and pruning methods often treat early-token activations specially, and why KV-cache eviction policies usually protect the initial tokens.
