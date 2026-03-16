---
title: Introduction to Recurrent Neural Networks (RNNs)
description: Explore how RNNs process sequential data like text and time series.
---

Recurrent Neural Networks (RNNs) are a class of neural networks designed to work with sequential or time-series data. Unlike standard feedforward neural networks, RNNs have a "memory" that allows them to pass information from one step of a sequence to the next.

## How RNNs Work

In a traditional neural network, inputs and outputs are independent of each other. However, for many tasks (like predicting the next word in a sentence), you need to know the previous words. RNNs solve this by using loops:

1. **Hidden State**: At each time step, the RNN calculates a "hidden state" based on the current input and the previous hidden state.
2. **Persistence**: This hidden state acts as a memory, carrying information about what the network has seen so far.

## Common Use Cases

- **Natural Language Processing (NLP)**: Machine translation, sentiment analysis, and text generation.
- **Time Series Prediction**: Stock market analysis and weather forecasting.
- **Speech Recognition**: Converting spoken language into text.

## Challenges with Basic RNNs

Basic RNNs often struggle with "long-term dependencies" due to:

- **Vanishing Gradient Problem**: As the sequence gets longer, the gradients used to update weights become very small, making it hard to learn.
- **Exploding Gradient Problem**: Conversely, gradients can become excessively large, causing instability.

To address these, more advanced architectures like **LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)** were developed.
