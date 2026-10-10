---
title: Audio Language Models - Understanding Sound with LLMs
description: Learn how audio language models connect audio encoders to LLMs for speech, music, and sound understanding, and how they differ from cascaded ASR pipelines.
---

Audio language models (audio LLMs) extend text language models to accept audio as input, and some also generate audio output. They enable tasks beyond transcription: answering questions about a recording, describing sounds, and following spoken instructions.

## Cascade versus End-to-End

**Cascaded pipeline**: speech recognition converts audio to text, an LLM processes the text, and text-to-speech produces audio. It is modular but loses information in the transcription step: tone, emotion, speaker identity, background sounds, and music.

**Audio-native model**: audio is encoded directly into representations the LLM can attend to, preserving paralinguistic information and reducing latency.

## Typical Architecture

```text
audio waveform
   -> audio encoder (e.g. Whisper-style encoder)       # spectrogram -> frame embeddings
   -> adapter / projector (linear layer or Q-Former)   # map to LLM embedding space
   -> LLM (decoder-only)                               # text tokens + audio embeddings
   -> text output (optionally -> audio codec tokens)
```

The audio encoder is usually pretrained on large-scale speech or general audio data, and the LLM is often kept mostly frozen while the adapter and a LoRA are trained.

## Training Stages

1. **Alignment**: train the adapter on audio-caption and ASR data
2. **Instruction tuning**: mixed tasks such as question answering, captioning, translation, and emotion recognition
3. **Preference tuning**: improve helpfulness and safety

## Generating Audio

Output audio can be produced by predicting discrete codec tokens (from neural audio codecs) that a decoder turns into a waveform, which enables real-time voice conversation in a single model.

## Capabilities and Tasks

- Speech recognition and translation
- Spoken question answering
- Audio captioning and sound event detection
- Music understanding
- Emotion and speaker attribute recognition
- Voice assistants with low-latency turn-taking

## Challenges

- Audio produces many frames per second, so sequences are long; downsampling and compression are needed
- Evaluation is harder than text, and benchmarks differ by task
- Safety concerns include voice cloning and sensitive attribute inference
