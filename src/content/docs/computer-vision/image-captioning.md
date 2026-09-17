---
title: Image Captioning - Generating Natural Language Descriptions of Images
description: Learn how image captioning models bridge vision and language, from encoder-decoder architectures to modern vision-language models.
---

Image captioning generates a natural-language description of an image's content, requiring a model to both recognize what's in a scene and express it fluently.

## Encoder-Decoder Architecture

The classic approach pairs a CNN image encoder with an RNN or transformer text decoder. The encoder produces a fixed or spatial feature representation of the image; the decoder generates the caption one word at a time, conditioning each word on the image features and the words generated so far.

```text
image -> CNN encoder -> feature map
feature map + "<start>" -> decoder -> "a"
feature map + "a" -> decoder -> "dog"
feature map + "a dog" -> decoder -> "running"
... -> "<end>"
```

Attention over spatial regions of the feature map lets the decoder focus on different parts of the image as it generates each word — attending to the region containing the dog when generating "dog," and to the region showing motion or open space when generating "running."

## Evaluation Metrics

BLEU, METEOR, and CIDEr all measure n-gram overlap between generated and reference captions, but they correlate imperfectly with human judgment of caption quality, since a fluent, accurate caption can use entirely different wording than the reference. CIDEr, which weights n-grams by how distinctive they are across the whole dataset (similar to TF-IDF weighting), correlates better with human judgment than plain BLEU for this task.

## Modern Vision-Language Models

Current captioning systems are usually a general-purpose vision-language model prompted to describe an image, rather than a dedicated captioning architecture trained end-to-end on caption pairs alone. This allows controllable captioning — asking for a one-sentence summary, a detailed accessibility description, or a caption focused on a specific object — through prompting rather than retraining.

## Practical Guidance

Accessibility use cases (alt-text generation) need factual accuracy above all — hallucinated details in an image description are actively harmful, so validate caption quality with human review before deploying at scale for accessibility purposes. For creative or marketing captioning, prioritize fluency and tone over strict factual density.
