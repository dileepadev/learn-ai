---
title: Diffusion Models for Audio - Generating Sound One Denoising Step at a Time
description: See how diffusion models extend from images to raw waveforms and spectrograms to generate music, speech, and sound effects.
---

Diffusion models generate data by learning to reverse a gradual noising process. For audio, the same core idea applies, but the choice of representation—raw waveform versus spectrogram—changes what the model actually has to learn.

```text
clean audio -> add noise step by step -> pure noise   (forward process, fixed)
pure noise -> remove noise step by step -> generated audio  (reverse process, learned)
```

## Waveform vs. Spectrogram Diffusion

Raw waveforms are extremely long, high-frequency sequences (44,100 samples per second for CD-quality audio), which makes direct waveform diffusion computationally expensive and prone to missing long-range structure. Many systems instead diffuse over a **spectrogram**—a 2D time-frequency representation—treating audio generation like image generation, then convert the result back to a waveform with a separate vocoder network. Others diffuse in a compressed latent space learned by an audio autoencoder, similar to latent diffusion in image generation, trading some fine detail for much lower computational cost.

## Conditioning Signals

Audio diffusion models are typically conditioned on additional inputs to control what they generate:

- **Text prompts** describing mood, genre, or instrumentation for music generation.
- **Melody or rhythm conditioning**, where a reference melody constrains the harmonic structure while other attributes vary.
- **Text transcripts and speaker embeddings** for text-to-speech systems, which combine diffusion with a target speaker's voice characteristics.

## Why Diffusion Fits Audio Well

Unlike autoregressive models that generate audio sample-by-sample or frame-by-frame in a fixed order, diffusion models generate the whole clip iteratively and can revise earlier parts of the signal at each denoising step. This tends to produce more globally coherent long-range structure, such as maintaining a consistent tempo or harmonic key across a full musical passage, at the cost of typically needing more sampling steps than a single forward pass would require.

## Practical Challenges

- Long-form generation (multi-minute tracks) still struggles with maintaining coherent structure across the whole piece, not just locally.
- Evaluating generated audio quality is difficult, since automatic quality metrics correlate only loosely with human perceptual judgments of music and speech naturalness.
- Training data provenance is a significant concern, since large audio datasets often include copyrighted music without clear licensing for AI training use.
