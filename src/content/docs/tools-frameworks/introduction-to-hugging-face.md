---
title: Introduction to Hugging Face
description: Getting started with the Hugging Face ecosystem for NLP and machine learning.
---

Hugging Face is the central hub for AI models, datasets, and demo applications. Their `transformers` library has become the industry standard for working with state-of-the-art models.

## Core Components

- **Hub:** A platform for hosting and sharing models, datasets, and spaces.
- **Transformers Library:** Simple API to download and use pre-trained models.
- **Datasets Library:** Easy access to thousands of datasets from various domains.
- **Tokenizers Library:** Highly optimized tool for converting text to numbers.

## Getting Started

You can load a model with just a few lines of code:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love learning about AI!"))
```
