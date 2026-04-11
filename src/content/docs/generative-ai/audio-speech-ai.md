---
title: Audio and Speech Generative AI
description: Discover how AI systems generate, transcribe, and transform audio and speech — from automatic speech recognition and text-to-speech synthesis to voice cloning, music generation, and audio language models.
---

Audio and speech AI has undergone a revolution comparable to what large language models brought to text. Systems that once required hours of recorded speech and weeks of training can now transcribe any language in real time, clone a voice from seconds of audio, and generate studio-quality music from a text prompt.

## The Audio Signal

All audio AI begins with the raw signal: a **waveform** — a one-dimensional time series of air pressure values sampled at a fixed rate (typically 16kHz–44.1kHz). Working directly with raw waveforms is computationally expensive due to high temporal resolution.

Most systems convert audio into **spectrograms** — 2D time-frequency representations that compress temporal information while preserving perceptually meaningful features:

$$S[t, f] = |STFT(x)[t, f]|^2$$

**Mel spectrograms** apply a non-linear frequency scale that aligns with human auditory perception, reducing the feature dimension while preserving perceptually important information.

## Automatic Speech Recognition (ASR)

ASR converts spoken audio into text. Modern ASR follows an **encoder-decoder** or **CTC** architecture.

### Whisper (OpenAI)

Whisper is a large-scale ASR model trained on 680,000 hours of diverse, multilingual web audio using weak supervision (subtitles as pseudo-labels). Key properties:

- Covers 99 languages
- Handles diverse recording conditions, accents, and noise
- Zero-shot language identification and translation built-in
- Sequence-to-sequence: audio mel spectrogram → transformer encoder → text decoder

### CTC-based Models

Connectionist Temporal Classification (CTC) provides an alternative loss function for ASR that doesn't require alignment labels. Models like **wav2vec 2.0** (Facebook AI) learn representations from unlabeled audio via self-supervised learning, then fine-tune for ASR on small labeled datasets — achieving strong performance even with just a few hundred hours of labeled speech.

## Text-to-Speech (TTS) Synthesis

TTS converts text into natural-sounding speech. Modern neural TTS operates through a two-stage pipeline:

```
Text → [Acoustic Model] → Mel Spectrogram → [Vocoder] → Waveform
```

### Acoustic Models

- **Tacotron 2 (Google):** Sequence-to-sequence model with attention; directly predicts mel spectrograms from text
- **FastSpeech 2:** Non-autoregressive spectrogram prediction with explicit duration, pitch, and energy predictors — faster and more controllable than Tacotron
- **VITS (Variational Inference TTS):** End-to-end model that predicts waveforms directly; eliminates the two-stage pipeline

### Vocoders

Convert mel spectrograms to audio waveforms:

- **WaveNet (Google):** Dilated causal convolutions generating samples autoregressively; high quality but slow (real-time factor ~1000x)
- **WaveGlow / HiFi-GAN:** GAN-based vocoders that generate waveforms in parallel; near-WaveNet quality at orders of magnitude higher speed

## Voice Cloning

Voice cloning synthesizes speech in a target speaker's voice using only a short reference audio sample (from seconds to a few minutes).

**Zero-shot voice cloning** systems (like **XTTS v2**, **Vall-E**, **CosyVoice**) encode a reference audio clip into a speaker embedding and condition the TTS decoder on it. The model generalizes style, accent, and vocal timbre to new text.

**Vall-E (Microsoft):** Frames TTS as a **language modeling problem over audio tokens**. Audio is discretized into codec tokens using EnCodec, then Vall-E predicts those tokens conditioned on text + 3 seconds of speaker reference audio:

$$P(\text{audio tokens} | \text{text}, \text{3s reference})$$

This allows zero-shot voice personalization at unprecedented naturalness.

## Audio Codec and Tokenization

Converting continuous audio into discrete tokens is foundational to treating audio like language:

- **EnCodec (Meta):** A neural audio codec that compresses waveforms into low-bitrate discrete codebooks (8 codebooks × 1024 codes) while achieving high reconstruction quality
- **SoundStream (Google):** Similar residual vector quantization approach
- **DAC (Descript Audio Codec):** Improved codec with higher quality at lower bitrate

Once audio is tokenized, standard **autoregressive Transformer language modeling** can be applied to audio directly.

## Audio Language Models

**AudioLM (Google, 2022)** was an early demonstration that an LM trained purely on audio tokens generates coherent, natural-sounding speech continuations — preserving speaker identity, prosody, and background acoustics without any text input.

**MusicLM (Google, 2023)** extends this to music generation conditioned on text descriptions:

- A hierarchical model generates semantic tokens → acoustic tokens → waveform
- Can generate music matching style descriptions: *"an upbeat Celtic folk song with fiddle, tin whistle, and bodhran"*

**AudioCraft (Meta, 2023)** packages **MusicGen** (music generation) and **AudioGen** (sound effects generation) into open-source models:

```python
from audiocraft.models import MusicGen
model = MusicGen.get_pretrained("facebook/musicgen-large")
model.set_generation_params(duration=30)
audio = model.generate(["Epic orchestral trailer music with choir and brass"])
```

## Speech-to-Speech Models and Voice AI

Modern systems can perform **real-time speech-to-speech translation** and **voice AI assistants** without text as an intermediate representation:

- **Seamless (Meta):** Multimodal model for real-time speech translation across 100+ languages
- **GPT-4o Audio mode:** End-to-end audio input/output with emotional expressiveness, including tone matching and natural interruptions
- **Moshi (Mistral):** Full-duplex spoken dialogue model enabling simultaneous speaking and listening

## Challenges and Ethics

### Deepfake Audio

Voice cloning enables the creation of convincing fake audio of any person. Harms include:

- Fraud and social engineering attacks
- Political disinformation
- Non-consensual voice reproduction

**Detection:** Audio deepfake detectors (e.g., trained on the ASVspoof benchmark) analyze spectral artifacts, but the arms race between generation and detection is ongoing.

### Bias in ASR

ASR systems have documented higher word error rates for:

- Non-native accents
- African American Vernacular English (AAVE)
- Older speakers and children
- Low-resource languages

These disparities can create barriers in medical, legal, and accessibility contexts.

### Consent and Artist Rights

AI music generation raises complex intellectual property questions: models trained on copyrighted music may implicitly reproduce stylistic elements. Legal frameworks are still evolving.

## Applications

- **Accessibility:** Real-time captions for deaf and hard-of-hearing users; voice interfaces for mobility-impaired users
- **Call centers:** ASR + TTS for conversational IVR systems
- **Podcasting:** AI voice cloning for content localization
- **Game audio:** Procedurally generated voice acting and ambient sound
- **Medical:** Clinical note dictation, diagnostic analysis of speech patterns (Parkinson's, depression)

## Further Reading

- Radford et al. (2022), *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*
- Wang et al. (2023), *Vall-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*
- Agostinelli et al. (2023), *MusicLM: Generating Music From Text*
- Défossez et al. (2022), *High Fidelity Neural Audio Compression (EnCodec)*
