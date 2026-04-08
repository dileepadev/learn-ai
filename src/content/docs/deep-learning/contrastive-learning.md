---
title: "Contrastive Learning: Understanding by Comparison"
description: "How models learn to identify concepts by looking at how similar and different they are."
---

**Contrastive Learning** is a powerful self-supervised technique used to train vision and language models without needing human-labeled data.

## The Principle

The model is taught to:

1. **Pull Together**: Representations of the same image (e.g., two different crops or color-jittered versions of a dog).
2. **Push Apart**: Representations of different images (e.g., the dog and a car).

## CLIP: A Classic Example

OpenAI's CLIP model uses contrastive learning by training on millions of (image, caption) pairs. It learns to associate the visual representation of an image with the text representation of its caption, enabling powerful zero-shot capabilities.
