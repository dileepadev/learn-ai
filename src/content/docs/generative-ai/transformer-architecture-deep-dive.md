---
title: Introduction to Transformers
description: Understand the architecture that revolutionized Natural Language Processing.
---

The Transformer architecture, introduced in the seminal paper "Attention is All You Need" (2017), has become the foundation for modern Large Language Models (LLMs) like GPT-4 and Claude.

## Why Transformers?

Before Transformers, models like Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks processed text sequentially (word by word). This made them:

1. **Slow to train:** Sequential processing cannot be easily parallelized.
2. **Poor at long-range dependencies:** They often forgot the beginning of a long sentence by the time they reached the end.

## The Core Components

Transformers solve these problems using a mechanism called **Self-Attention**.

### 1. Self-Attention Mechanism

Self-attention allows the model to look at other words in the input sequence to help get a better encoding for a specific word. For example, in the sentence "The animal didn't cross the street because it was too tired," the word "it" refers to "animal." Self-attention enables the model to make this connection.

The attention score is calculated as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where:

- $Q$ is the Query matrix.
- $K$ is the Key matrix.
- $V$ is the Value matrix.
- $d_k$ is the dimension of the keys.

### 2. Multi-Head Attention

Instead of performing self-attention once, Transformers do it multiple times in parallel ("Multi-Head"). This allows the model to focus on different types of relationships between words simultaneously.

### 3. Positional Encoding

Since Transformers don't use recurrence, they need a way to understand the order of words. Positional encoding adds information about the position of each word in the sequence.

## The Encoder-Decoder Structure

- **Encoder:** Processes the input sequence and creates a representation of it.
- **Decoder:** Uses the encoder's representation to generate an output sequence, one token at a time.

Modern architectures like GPT are "decoder-only," while models like BERT are "encoder-only."
