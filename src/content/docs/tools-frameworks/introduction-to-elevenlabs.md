---
title: Introduction to ElevenLabs - High-Fidelity Voice AI
description: Learn how ElevenLabs' text-to-speech and voice cloning API powers natural-sounding synthetic voices, and its safeguards against misuse.
---

ElevenLabs provides a text-to-speech and voice cloning API known for unusually natural prosody and emotional expressiveness compared to earlier-generation TTS systems, widely used for narration, dubbing, and conversational voice agents.

## Basic Text-to-Speech

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key="ELEVENLABS_API_KEY")

audio = client.text_to_speech.convert(
    voice_id="Rachel",
    text="Welcome back. Here's your daily briefing.",
    model_id="eleven_multilingual_v2"
)
```

The API exposes fine-grained controls over voice stability and style exaggeration, letting a developer trade off consistency (a more monotone, predictable voice) against expressiveness (a more dynamic, emotionally varied voice) depending on the use case — narration typically favors stability, while character voice acting favors expressiveness.

## Voice Cloning and the Voice Library

ElevenLabs supports both instant voice cloning from a short reference sample and professional voice cloning trained on a larger, curated dataset for higher fidelity, alongside a marketplace of licensed voices contributed by voice actors who are compensated for usage — an explicit alternative to unlicensed cloning of arbitrary reference audio.

## Conversational AI and Low-Latency Streaming

Beyond one-shot text-to-speech, ElevenLabs offers a conversational AI product combining low-latency streaming speech synthesis with speech-to-text and LLM orchestration, aimed at building responsive voice agents where end-to-end latency between a user finishing speaking and hearing a response is critical to feeling natural.

## Safeguards Against Misuse

Given the platform's cloning fidelity, ElevenLabs implements safeguards including moderation of cloning requests for public figures, audio watermarking research (embedding detectable signals in generated audio), and a no-go voices list restricting cloning of certain protected or frequently impersonated voices without verified consent.

## Practical Guidance

For narration and content production, tune stability and style settings per use case rather than using defaults, since the right expressiveness level varies significantly between formal narration and character dialogue. If offering a cloning feature to end users, layer additional consent verification on top of the platform's built-in safeguards, since responsibility for verifying a user's right to clone a given voice ultimately sits with the application collecting that reference audio.
