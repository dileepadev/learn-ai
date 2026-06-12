---
title: Recurrent Neural Networks
description: An introduction to recurrent neural networks (RNNs), the architecture designed to model sequential data and temporal dependencies.
---

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data by maintaining a hidden state that carries information from previous time steps. They were the dominant architecture for NLP, speech recognition, and time-series tasks before Transformers became prevalent.

## The Core Idea

Unlike feedforward networks that process each input independently, an RNN processes inputs one element at a time and updates a **hidden state** at each step:

```
hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)
```

This hidden state acts as a "memory" that accumulates information over the sequence. The same weights (Wₕ, Wₓ) are shared across all time steps — this is called **weight sharing**.

## Limitations of Vanilla RNNs

- **Vanishing gradients:** During backpropagation through time (BPTT), gradients shrink exponentially over long sequences, making it hard to learn long-range dependencies.
- **Exploding gradients:** The opposite problem — gradients grow uncontrollably (mitigated with gradient clipping).
- **Sequential computation:** Each step depends on the previous, preventing parallelization during training.

## Long Short-Term Memory (LSTM)

LSTMs were introduced to overcome the vanishing gradient problem by adding a **cell state** and gating mechanisms:

- **Forget gate:** Decides what information to discard from the cell state.
- **Input gate:** Decides what new information to store.
- **Output gate:** Controls what to output from the cell state.

This architecture allows LSTMs to selectively retain information over hundreds of time steps, making them far more effective than vanilla RNNs for long sequences.

## Gated Recurrent Unit (GRU)

The GRU is a simplified version of the LSTM with only two gates (reset and update). It often achieves similar performance with fewer parameters, making it faster to train.

## Bidirectional RNNs

Standard RNNs only look at past context. **Bidirectional RNNs** process the sequence in both directions and concatenate the hidden states, giving each position access to both past and future context. Widely used for tasks like named entity recognition.

## Sequence-to-Sequence Models

RNNs can be stacked in an **encoder-decoder** architecture:
- The **encoder** reads the entire input sequence and compresses it into a context vector.
- The **decoder** generates the output sequence step by step.

This approach was highly successful for machine translation before Transformers, and attention mechanisms were added to address the fixed context vector bottleneck.

## Common Applications

- **Text classification and sentiment analysis**
- **Machine translation (pre-Transformer era)**
- **Speech recognition**
- **Time-series forecasting**
- **Music and text generation**

## RNNs vs. Transformers

Today, Transformers have largely replaced RNNs for most NLP tasks due to better parallelization and superior handling of long-range dependencies. However, RNNs remain relevant for:
- **Streaming / online inference** where inputs arrive one at a time.
- **Embedded / resource-constrained devices** where Transformer overhead is prohibitive.
- **State-space models** (like Mamba) draw inspiration from RNN-like architectures and are a current research frontier.
