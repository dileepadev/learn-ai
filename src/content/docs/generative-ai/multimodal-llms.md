---
title: "Multimodal LLMs: Vision, Audio, and Beyond"
description: "Understand how modern LLMs process images, audio, and video alongside text — covering vision encoders, cross-modal attention, and the architecture of models like GPT-4o and Gemini."
---

The most capable AI systems today aren't just language models — they're **multimodal models** that can see, hear, and reason across different types of information simultaneously. Understanding how they work helps you use them more effectively and build better applications.

## What Makes a Model Multimodal?

A multimodal model can process and generate content across multiple modalities: text, images, audio, video, and structured data. The key challenge is representing these fundamentally different data types in a shared space where the model can reason about their relationships.

## Vision-Language Models

### Architecture: Vision Encoder + LLM

The dominant architecture for vision-language models:

1. **Vision Encoder**: A pretrained vision model (typically a ViT — Vision Transformer) converts an image into a sequence of patch embeddings.
2. **Projection Layer**: A learned linear layer or small MLP maps vision embeddings into the LLM's token embedding space.
3. **LLM**: Processes the combined sequence of image tokens and text tokens.

The image is effectively "tokenized" into visual tokens that the LLM treats like text tokens.

### LLaVA and Open-Source Vision Models

**LLaVA (Large Language and Vision Assistant)** popularized this architecture for open-source models. It uses CLIP as the vision encoder and connects it to an LLM via a simple projection layer, trained on image-text instruction pairs.

### Native Multimodality: GPT-4o and Gemini

More recent models like GPT-4o and Gemini are trained natively on multiple modalities from the start, rather than bolting a vision encoder onto a text model. This enables:
- Better cross-modal reasoning.
- Real-time audio processing.
- Video understanding.
- Interleaved image and text generation.

## Audio Processing

Audio is typically processed by:
1. Converting the waveform to a mel spectrogram (a 2D frequency-time representation).
2. Encoding the spectrogram with a convolutional or transformer encoder (like Whisper's encoder).
3. Projecting audio embeddings into the LLM's token space.

**GPT-4o** processes audio natively, enabling real-time voice conversation with natural prosody and emotion detection.

## Video Understanding

Video adds the temporal dimension. Approaches include:
- **Frame sampling**: Sample frames at regular intervals and process as a sequence of images.
- **Video encoders**: Specialized encoders (like VideoMAE) that capture temporal relationships.
- **Temporal attention**: Attention mechanisms that operate across frames.

The main challenge is the enormous token count — a 1-minute video at 1 fps with 256 tokens per frame is 15,360 tokens.

## Practical Capabilities

Modern multimodal LLMs can:
- **Document understanding**: Parse PDFs, forms, invoices with mixed text and images.
- **Chart and graph analysis**: Extract data and trends from visualizations.
- **Code screenshot debugging**: Identify errors from screenshots of code or error messages.
- **Medical imaging**: Assist with radiology report generation (with appropriate caveats).
- **UI automation**: Understand and interact with graphical interfaces.

## Limitations

- **Spatial reasoning**: Models still struggle with precise spatial relationships ("is the red circle to the left of the blue square?").
- **Counting**: Accurately counting objects in images remains unreliable.
- **Fine-grained visual details**: Small text, subtle differences between similar objects.
- **Temporal reasoning in video**: Understanding causality and long-range temporal dependencies.

Multimodal capabilities are improving rapidly, but these limitations are important to account for in production applications.
