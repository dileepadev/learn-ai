---
title: Diffusion Models
description: Understanding diffusion models — the generative AI architecture behind Stable Diffusion, DALL-E 2, and Imagen.
---

Diffusion models are a class of generative models that learn to create data (images, audio, video) by reversing a process of gradually adding noise. They have become the dominant approach for high-quality image synthesis, enabling systems like Stable Diffusion, DALL-E 2, and Midjourney.

## The Core Idea

Diffusion models work in two phases:

1. **Forward process (diffusion):** Gradually add Gaussian noise to a data sample over many steps until it becomes pure random noise. This process is fixed and requires no learning.

2. **Reverse process (denoising):** Train a neural network to predict and remove the noise added at each step, learning to reconstruct the original data from noise.

At inference time, the model starts with random noise and iteratively denoises it to produce a new, realistic sample.

## Forward Process

Given a data point x₀ (e.g., an image), the forward process adds noise over T steps:

```
xₜ = √(αₜ) · xₜ₋₁ + √(1 - αₜ) · ε    where ε ~ N(0, I)
```

After enough steps (typically T = 1000), xₜ is approximately pure Gaussian noise.

## Reverse Process

A U-Net (or Transformer) is trained to predict the noise ε added at each step, given the noisy image and the timestep. The training objective is simple:

```
L = ||ε - εθ(xₜ, t)||²
```

During generation, the model iteratively applies the denoiser to go from noise back to a clean sample.

## Conditioning: Text-to-Image

To steer generation toward a specific prompt, text embeddings are injected into the denoiser using **cross-attention**. A text encoder (like CLIP) converts the prompt into embeddings, which guide what the denoiser produces at each step.

**Classifier-Free Guidance (CFG)** amplifies the text conditioning: the model is trained with and without the text condition, then at inference the conditioned and unconditioned outputs are combined with a guidance scale to increase adherence to the prompt.

## Latent Diffusion Models (LDMs)

Running diffusion in pixel space is expensive. **Latent diffusion models** (the architecture behind Stable Diffusion) first compress images into a lower-dimensional latent space using a VAE, then run the diffusion process there. This dramatically reduces computation while preserving quality.

## Key Variants

- **DDPM (Denoising Diffusion Probabilistic Models):** The foundational paper (Ho et al., 2020).
- **DDIM:** Deterministic sampling — fewer steps needed, faster generation.
- **Score-based models:** Related framework using score matching.
- **Flow Matching:** A newer, more efficient alternative with straight trajectories in latent space (used in Flux, Stable Diffusion 3).

## Applications

- **Image generation:** Text-to-image (Stable Diffusion, DALL-E 3).
- **Image editing:** Inpainting, outpainting, style transfer.
- **Video generation:** Sora, Runway, Kling.
- **Audio synthesis:** Music and voice generation.
- **3D generation:** NeRF and 3D asset synthesis.
- **Drug discovery:** Generating molecular structures.

## Strengths and Limitations

**Strengths:**
- High-quality, diverse outputs.
- Strong text alignment with guidance.
- More stable training than GANs.

**Limitations:**
- Slow inference (many denoising steps).
- Computationally expensive to train.
- Can still produce artifacts or violate physical constraints.
