---
title: Latent Consistency Models
description: Understand Latent Consistency Models (LCMs) — a distillation technique that compresses diffusion sampling from 50+ steps to 1–4 steps while maintaining image quality, enabling real-time image generation and interactive creative tools.
---

Latent Consistency Models (LCMs) are a class of generative models derived from diffusion models through a process called **consistency distillation**. Where a standard diffusion model requires 20–50 neural network evaluations (denoising steps) to produce a high-quality sample, an LCM can generate comparable images in **1 to 4 steps** — a 10–50× speedup that transforms diffusion-based generation from a batch process into an interactive, real-time tool.

## The Diffusion Bottleneck

Stable Diffusion and similar latent diffusion models generate images by iteratively denoising a noise tensor $z_T \sim \mathcal{N}(0, I)$ toward a data sample $z_0$ along a probability flow ODE trajectory:

$$\frac{dz}{dt} = f(z, t)$$

This ODE is solved by discretizing into $T$ steps and calling the score network $\epsilon_\theta(z_t, t)$ at each step. Each step is a full forward pass through the UNet or DiT backbone — expensive and irreducible with standard solvers like DDPM, DDIM, or DPM-Solver.

While DPM-Solver and DDIM reduce steps to 20–50 for acceptable quality, real-time generation (< 100ms) at interactive resolutions remained out of reach.

## Consistency Models: The Core Idea

**Consistency models** (Song et al., 2023) introduce a new class of generative model trained to directly map **any point on a diffusion trajectory back to its clean data origin**:

$$f_\theta(z_t, t) \approx z_0 \quad \forall t \in [0, T]$$

The fundamental property is **self-consistency**: for any two points $z_t$ and $z_{t'}$ on the same trajectory (i.e., both corresponding to the same clean $z_0$):

$$f_\theta(z_t, t) \approx f_\theta(z_{t'}, t') \approx z_0$$

This is enforced by the **consistency training objective**:

$$\mathcal{L}_{\text{CT}}(\theta, \theta^-) = \mathbb{E}\!\left[d\!\left(f_\theta(z_{t_{n+1}}, t_{n+1}),\, f_{\theta^-}(z_{t_n}, t_n)\right)\right]$$

where $\theta^-$ is an exponential moving average (EMA) of $\theta$ (a slow-moving target network), and $d(\cdot, \cdot)$ is a perceptual distance metric (LPIPS or $\ell_2$). The consistency loss minimizes the discrepancy between the model's predictions at adjacent trajectory points, progressively building up the self-consistency property across the whole trajectory.

At inference, a **single** call $f_\theta(z_T, T)$ maps pure noise directly to the clean image. For higher quality, a **multi-step** procedure alternates consistency evaluations with small amounts of re-injection noise, improving sample quality with each additional step.

## Latent Consistency Models

**LCM** (Luo et al., 2023) applies consistency distillation to **pre-trained latent diffusion models** like Stable Diffusion, combining consistency training with two key modifications:

### 1. Operating in Latent Space

LCMs work in the compressed latent space of a pretrained VAE, not in pixel space. This dramatically reduces the dimensionality of the trajectory (e.g., $64 \times 64 \times 4$ latents vs. $512 \times 512 \times 3$ pixels), making consistency training computationally tractable.

### 2. Classifier-Free Guidance Consistency

Text-conditional generation requires **classifier-free guidance (CFG)** — blending conditional and unconditional score predictions:

$$\hat{\epsilon} = \epsilon_\theta(z_t, t, \varnothing) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing))$$

LCM integrates CFG directly into the consistency function during distillation by using **augmented score functions** that fold guidance into the trajectory itself. This allows the distilled model to generate guided images in a single step without needing to evaluate two conditional/unconditional branches at inference.

### 3. Distillation from a Teacher

Rather than training from scratch, LCM **distills** from a pretrained diffusion model (the "teacher") using the consistency training loss. The teacher provides the target $z_{t_n}$ by running one denoising step from $z_{t_{n+1}}$. This transfer from an already-capable teacher dramatically reduces the amount of training needed — LCM can be distilled with ~32 GPU-hours from Stable Diffusion, vs. thousands of GPU-hours required to train SD from scratch.

## LCM-LoRA: Plug-in Acceleration

**LCM-LoRA** (Luo et al., 2023) distills LCM into a **low-rank adapter (LoRA)** rather than fine-tuning the full model weights. The result is a universal acceleration adapter:

- Attach LCM-LoRA to **any** fine-tuned Stable Diffusion checkpoint (DreamBooth models, stylized models, etc.).
- The base model's learned knowledge and style are preserved; only the sampling speed changes.
- Requires only 4-bit compatible LoRA weights (~3MB) versus a full model checkpoint (~4GB).

This composability made LCM-LoRA one of the most widely adopted acceleration techniques in the open-source diffusion model ecosystem.

## Inference: Step Count vs. Quality Trade-off

| Steps | Latency (A100) | Quality |
| --- | --- | --- |
| 1 | ~50ms | Draft quality — good for previews |
| 2 | ~100ms | Near-DDIM-20 quality |
| 4 | ~200ms | Matches DDIM-50 on most metrics |
| 8 | ~400ms | Slight improvement in fine details |
| 50 (DDIM) | ~2500ms | Baseline quality |

The 4-step LCM profile is the practical sweet spot: roughly equivalent to DDIM with 50 steps at ~12× lower latency.

## Consistency Fine-Tuning

Beyond distillation from an existing teacher, **consistency fine-tuning (CTF)** trains a consistency model jointly during the initial pretraining phase — not as a post-hoc distillation. CTF models (e.g., iCT, consistency trajectory models) achieve higher quality at 1–2 steps than distillation-based LCMs, at the cost of requiring consistency training from the start rather than leveraging an existing pretrained model.

## Applications Enabled by Real-Time Diffusion

LCMs' sub-100ms latency unlocks application categories that were previously impossible:

- **Interactive canvas tools:** Real-time inpainting and image editing where the canvas updates as the user paints.
- **Video stream stylization:** Applying diffusion-style generation to live video at multiple frames per second.
- **Game asset generation:** Generating textures and sprites in-engine without pre-baking.
- **On-device generation:** Running image generation on mobile CPUs and NPUs within user interaction latency budgets.
- **Iterative creative prompting:** Testing dozens of prompt variations per minute rather than waiting minutes per generation.

## Relationship to Other Fast Diffusion Techniques

| Technique | Approach | Typical Steps |
| --- | --- | --- |
| DDIM | Deterministic ODE solver | 20–50 |
| DPM-Solver++ | High-order ODE solver | 10–20 |
| Progressive Distillation | Step doubling distillation | 4–8 |
| Consistency Models | Self-consistency distillation | 1–4 |
| Flow Matching | Straighter ODE trajectories | 5–20 |
| ADD (Adversarial Diffusion) | GAN-augmented distillation | 1–4 |
| SDXL-Turbo | Score distillation + adversarial | 1–4 |

LCMs occupy the same speed tier as adversarial distillation methods (ADD, SDXL-Turbo) but without the training complexity of a GAN discriminator.

## Limitations

- **Mode truncation:** Very few steps limit the diversity of generated samples — the model collapses toward the most probable modes in the data distribution.
- **Fine detail degradation:** Extremely fine-grained textures and high-frequency details are rendered less sharply than full-step DDIM.
- **Distillation cost:** Distilling from a new base model requires rerunning the distillation process; LCM-LoRA mitigates this but does not eliminate it.
- **Guidance strength limits:** Very high CFG guidance values that produce sharp results in DDIM may not transfer cleanly to the few-step regime.

## Summary

Latent Consistency Models close the latency gap between diffusion models and GAN-speed generation by learning to map any noisy trajectory point directly to its clean origin. Distilled from pre-trained latent diffusion models and optionally packaged as LoRA adapters, LCMs achieve 1–4 step generation with quality that rivals 20–50 step DDIM — enabling real-time interactive image generation, on-device deployment, and high-throughput creative workflows. They represent the most practical solution currently available for closing the speed–quality gap in diffusion-based synthesis.
