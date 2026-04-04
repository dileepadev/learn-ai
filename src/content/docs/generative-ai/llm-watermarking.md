---
title: "Understanding LLM Watermarking"
description: "An overview of techniques used to identify AI-generated content through statistical signals and hidden markers."
---

As AI-generated content becomes more prevalent, the ability to distinguish it from human-written text is crucial for academic integrity, misinformation prevention, and copyright. **LLM Watermarking** is a technique designed to embed invisible signals into the generated text.

## How It Works: The "Green List" Method

Most watermarking techniques work during the **token selection** phase of text generation:

1. When generating a token, the model's vocabulary is split into a "green list" and a "red list" based on a hash of the previous token.

2. The model is biased to select tokens from the "green list."

3. A human writing naturally would select a random distribution of green and red tokens, but a watermarked AI will have a statistically improbable number of green tokens.

## Detection

Detecting a watermark doesn't require access to the model, just the hash function and the seed used for the green list. By analyzing the frequency of green-list tokens, a detector can calculate a "z-score" indicating the likelihood that the text was AI-generated.

## Limitations

- **Editing**: Significant editing or paraphrasing by a human can "wash away" the watermark.
- **Translation**: Translating the text to another language and back often removes the statistical bias.
- **Quality Trade-offs**: Aggressive watermarking can sometimes lead to less diverse or creative text generation.
