---
title: Flow Matching for Generative Models
description: How flow matching provides a simpler and more efficient alternative to diffusion models for generative AI — covering continuous normalizing flows, conditional flow matching (CFM), rectified flows, and their applications in image, audio, and video generation.
---

**Flow matching** is a generative modeling framework that trains a neural network to define a **continuous transformation** (a flow) between a simple prior distribution (e.g., Gaussian noise) and the target data distribution. It has emerged as one of the most promising alternatives and complements to diffusion models, offering simpler training objectives, straighter sampling trajectories, and faster inference.

Models like **Stable Diffusion 3**, **FLUX**, **Voicebox** (Meta), and **Matcha-TTS** are built on flow matching, marking its transition from a research idea to production-grade generative AI.

## Background: Continuous Normalizing Flows

The theoretical foundation of flow matching is **Continuous Normalizing Flows (CNFs)**. A CNF defines a time-dependent vector field $v_t(x)$ that generates a flow — a family of diffeomorphisms $\phi_t: \mathbb{R}^d \to \mathbb{R}^d$ satisfying the ODE:

$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x)), \quad \phi_0(x) = x$$

Starting from a sample $x_0 \sim p_0$ (e.g., Gaussian noise), integrating this ODE forward in time from $t=0$ to $t=1$ produces a sample $x_1 \sim p_1$ (the data distribution).

**Training CNFs classically** required simulating the ODE and computing the log-likelihood via the instantaneous change of variables formula — computationally expensive and unstable at scale.

## Flow Matching: The Key Insight

**Flow matching** (Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2022) bypasses the expensive CNF training objective. Instead of maximizing likelihood, it directly trains the neural network to match a **target vector field** constructed from simple interpolants between noise and data.

### The Flow Matching Objective

Given a data sample $x_1 \sim p_{\text{data}}$ and a noise sample $x_0 \sim \mathcal{N}(0, I)$, define a simple interpolation path:

$$x_t = (1-t) x_0 + t x_1, \quad t \in [0, 1]$$

The target vector field at each point on this path is simply:

$$u_t(x_t | x_1) = x_1 - x_0$$

This is the constant direction from noise to data — a **straight line** in data space. The flow matching loss trains a network $v_\theta(x_t, t)$ to match this:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

This is extremely simple to implement and compute — no likelihood estimation, no score matching, no ELBO.

## Conditional Flow Matching (CFM)

The marginal vector field (averaged over all data points) is complex and hard to estimate. **Conditional Flow Matching** conditions on individual data points, making the target tractable:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, p(x_1), p(x_0|x_1)} \left[ \| v_\theta(x_t, t) - u_t(x_t | x_1) \|^2 \right]$$

The conditional target vector field $u_t(x_t | x_1)$ is simple and analytical (just the interpolation direction). Despite conditioning on individual samples, it has been proven that minimizing $\mathcal{L}_{\text{CFM}}$ is equivalent to minimizing $\mathcal{L}_{\text{FM}}$ — so the marginal flow is learned correctly.

This makes CFM:

- Mathematically principled.
- Extremely easy to train (no Monte Carlo estimation of intractable integrals).
- Flexible in the choice of interpolant and noise distribution.

## Rectified Flows

**Rectified flows** (Liu et al., 2022) is an equivalent framework emphasizing the straight-line transport property. The key observation: if the learned flow is perfectly straight (the vector field points directly from $x_0$ to $x_1$), then a **single Euler step** suffices for sampling — integration is trivial.

In practice, flows are not perfectly straight after training. The **reflow** procedure iteratively straightens them:

1. Train a flow model.
2. Generate $(x_0, x_1)$ pairs by sampling noise and running the model.
3. Retrain a new model on these straighter coupled pairs.
4. Repeat.

Each reflow iteration produces straighter trajectories and improves one-step or few-step sampling quality. **InstaFlow** used reflow to achieve competitive one-step image generation.

## Flow Matching vs. Diffusion Models

| Aspect | Diffusion Models | Flow Matching |
| --- | --- | --- |
| Training objective | Score matching (DSM) | Direct vector field regression |
| Forward process | Stochastic (adds noise) | Deterministic (interpolation) |
| Sampling trajectory | Curved (curved SDE paths) | Straight (or near-straight) |
| Number of steps needed | 20–1000 (DDPM), 10–50 (DDIM) | 2–30 (fewer due to straighter paths) |
| Theoretical basis | SDEs, Langevin dynamics | ODEs, optimal transport |
| Implementation complexity | Moderate | Lower |

**Key advantage of flow matching**: Straighter sampling trajectories mean fewer neural function evaluations (NFE) are needed for high-quality samples, making inference faster.

## Optimal Transport CFM

The interpolant $x_t = (1-t)x_0 + t x_1$ can be made even straighter by choosing the pairing between $x_0$ and $x_1$ optimally — the **optimal transport** assignment that minimizes total transport cost. **OT-CFM** uses mini-batch optimal transport to approximately achieve this, resulting in flows that are nearly geodesic and require very few integration steps.

## Applications

### Stable Diffusion 3 and FLUX

**Stable Diffusion 3** (Stability AI, 2024) replaces the DDPM-based diffusion process with **rectified flows**, achieving better sample quality and faster inference than SD 2.x/SDXL. The multi-modal DiT architecture processes text and image tokens jointly with a flow matching training objective.

**FLUX** (Black Forest Labs, 2024 — the team behind Stable Diffusion) uses flow matching with a hybrid transformer architecture. FLUX.1 became the state-of-the-art for text-to-image generation, particularly for text rendering, photorealism, and prompt adherence.

### Voicebox (Meta, 2023)

**Voicebox** is a flow matching-based speech synthesis model capable of:

- Text-to-speech synthesis.
- Zero-shot voice style transfer.
- Speech infilling (filling masked portions of audio).
- Cross-lingual TTS.

It uses CFM in the mel-spectrogram domain, conditioned on text via a separate phoneme duration predictor. Voicebox demonstrated that flow matching matches or exceeds diffusion-based TTS quality with significantly fewer sampling steps.

### Matcha-TTS

**Matcha-TTS** is a lightweight, fast TTS system built on OT-CFM. It achieves high naturalness with 2–5 inference steps — a significant speedup over diffusion-based TTS systems.

### Video Generation

Flow matching is increasingly applied to video generation. **Movie Gen** (Meta, 2024) uses flow matching at the foundation of a large-scale text-to-video and image-to-video generation system, leveraging the speed advantages of straighter sampling trajectories.

## Stochastic Interpolants

**Stochastic interpolants** (Albergo et al., 2023) generalize the flow matching framework to include stochastic paths — not just deterministic straight-line interpolations. This provides a unified theoretical framework that encompasses both diffusion models (as a special case with Brownian motion noise) and flow matching (as a special case with straight interpolants), allowing principled comparison and combination.

## Sampling with ODE Solvers

Flow matching models are integrated using ODE solvers:

- **Euler method**: $x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$. Fastest but least accurate.
- **Runge-Kutta (RK4)**: Higher-order accuracy, fewer steps needed.
- **Adaptive step-size solvers** (dopri5, heun): Automatically adjust step size for accuracy-speed trade-offs.
- **Consistency models** / **flow consistency models**: Distillation approaches that further reduce NFE to 1–2 steps.

## Implementation Sketch

A minimal flow matching training loop in PyTorch:

```python
for x1 in dataloader:           # real data samples
    x0 = torch.randn_like(x1)   # noise samples
    t  = torch.rand(B, 1, 1, 1) # random timestep

    # Linear interpolation (rectified flow path)
    xt = (1 - t) * x0 + t * x1

    # Target: direction from x0 to x1
    target = x1 - x0

    # Network predicts vector field
    pred = model(xt, t)

    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
```

Sampling (Euler integration):

```python
x = torch.randn(B, C, H, W)   # start from noise
steps = 30
for i in range(steps):
    t = torch.tensor(i / steps)
    x = x + (1 / steps) * model(x, t)
# x is now a generated sample
```

## Why Flow Matching Matters

Flow matching has rapidly become the preferred training framework for new generative models because it is:

- **Simpler to implement** than DDPM — no noise schedule tuning, no VLB decomposition.
- **Faster at inference** — straighter paths require fewer steps.
- **More flexible** — works for any data modality (images, audio, video, molecular structures, proteins).
- **Theoretically principled** — connects directly to optimal transport theory.

As diffusion models continue to dominate deployed systems, flow matching is steadily becoming the training paradigm of choice for the next generation of generative AI.
