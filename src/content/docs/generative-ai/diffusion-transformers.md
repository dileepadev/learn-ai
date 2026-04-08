---
title: "Diffusion Transformer (DiT): Scaling Image Generation"
description: "Understanding the architecture that combines Diffusion Models with the scalability of Transformers."
---

The **Diffusion Transformer (DiT)** is a breakthrough architecture (used in models like Sora) that replaces the traditional U-Net backbone in diffusion models with a Transformer.

## Why DiT?

- **Scalability**: Transformers scale much more predictably with compute and data than U-Nets.
- **Flexibility**: DiT can handle different resolutions and aspect ratios more easily by treating image patches as a sequence.
- **Latent Space Processing**: By operating in a compressed latent space, DiT can generate high-quality images and video with less memory.

## Impact on Video Generation

DiT has become the standard for high-fidelity video generation, as its ability to model long-range dependencies is critical for maintaining temporal consistency (ensuring objects don't morph or disappear over time).
