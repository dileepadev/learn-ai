---
title: Multimodal AI Models
description: Understanding multimodal models that process and generate across multiple modalities — text, images, audio, and video.
---

Multimodal AI models can understand and generate content across more than one modality — most commonly text and images, but increasingly audio, video, and structured data as well. Models like GPT-4o, Gemini, and Claude are multimodal, accepting mixed inputs and producing outputs that may span modalities.

## What Is a Modality?

A modality is a type or format of data:
- **Text:** Language, code, structured data.
- **Images:** Photos, diagrams, screenshots.
- **Audio:** Speech, music, environmental sounds.
- **Video:** Sequences of frames with audio.
- **3D / sensor data:** Point clouds, depth maps, tabular sensor readings.

## How Multimodal Models Work

Most multimodal LLMs use a **modality encoder + language model** design:

1. A specialized encoder (e.g., a Vision Transformer for images, Whisper for audio) converts non-text inputs into embeddings.
2. These embeddings are projected into the language model's token space via a learned adapter layer.
3. The language model processes the combined token sequence and generates a text response.

For image generation, a separate decoder (diffusion model or autoregressive image model) is often attached.

## Notable Models

- **GPT-4o:** Natively processes text, images, and audio in a unified model.
- **Gemini 1.5 Pro:** Handles text, images, audio, video, and code in a 1M token context.
- **Claude 3.5:** Text and image input, text output.
- **LLaVA:** Open-source multimodal model connecting a CLIP vision encoder to Llama.
- **Whisper:** OpenAI's speech-to-text model, widely used as an audio encoder in multimodal pipelines.

## Key Capabilities

- **Visual question answering:** Answer questions about images.
- **Document understanding:** Parse PDFs, forms, receipts, screenshots.
- **Image captioning and description.**
- **Audio transcription and spoken dialogue.**
- **Video understanding:** Summarize or answer questions about video content.
- **Chart and diagram analysis.**

## Challenges

- **Alignment across modalities:** Ensuring the model correctly grounds language in visual or audio content.
- **Hallucination in vision:** Models can misread text in images or hallucinate visual details.
- **Training data complexity:** Requires large, paired multimodal datasets.
- **Evaluation:** Measuring multimodal understanding is harder than text-only benchmarks.
