---
title: Rectified Flow
description: Learn about Rectified Flow — a generative modeling framework that trains straight-line ODE trajectories between noise and data, enabling fast, high-quality generation with fewer function evaluations.
---

Rectified Flow is a framework for learning generative models that transport noise to data via **straight-line trajectories** in the feature space. Proposed by Liu et al. (2022), it unifies flow matching and score-based diffusion under a clean formulation and enables high-quality generation with very few neural function evaluations (NFEs) — sometimes just one.

## Core Idea

Most generative models implicitly or explicitly define a transport map from a source distribution $\pi_0$ (typically Gaussian noise) to a target distribution $\pi_1$ (data). Diffusion models use a stochastic differential equation; normalizing flows use invertible networks; flow matching uses ODEs.

Rectified Flow takes the simplest possible path: it trains a velocity field $v_\theta(x, t)$ such that the ODE

$$\frac{dx}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$

transports $x_0 \sim \pi_0$ to $x_1 \sim \pi_1$, and does so along **straight lines** wherever possible.

## Training Objective

### Linear Interpolation

The key construction is to pair noise samples $Z \sim \pi_0$ with data samples $X \sim \pi_1$ and define straight-line interpolants:

$$X_t = (1 - t) Z + t X, \quad t \in [0, 1]$$

The target velocity at each point on the path is simply:

$$\dot{X}_t = X - Z$$

### Rectified Flow Loss

The model is trained to minimize the squared error between the predicted velocity and the straight-line velocity:

$$\mathcal{L}_{\text{RF}}(\theta) = \int_0^1 \mathbb{E}\left[\|v_\theta(X_t, t) - (X - Z)\|^2\right] dt$$

This is the flow matching objective specialized to linear interpolants. Compared to denoising score matching used in diffusion models, it directly supervises the transport direction rather than score functions.

The critical difference from general flow matching is that the target is always $X - Z$, a **constant vector field** along each trajectory — which biases the learned field toward straightness.

## The Reflow Procedure

Even with the linear training objective, learned trajectories may curve because pairs $(Z, X)$ are drawn independently (not coupled optimally). The **Reflow** procedure iteratively straightens the trajectories:

### Reflow Steps

1. Train a rectified flow model $v_1$ on independent pairs $(Z, X)$
2. Generate paired data: sample $Z \sim \pi_0$, compute $X' = \Phi_{v_1}(Z)$ by simulating the ODE (a "fake" data point that is paired with $Z$)
3. Train a new model $v_2$ on the correlated pairs $(Z, X')$

At each reflow step, the coupling between noise and data becomes closer to the **monotone transport map** (the OT-optimal coupling), which corresponds to perfectly straight trajectories. The objective remains the same; only the pairing changes.

### Why Straighter is Better

Straighter trajectories require fewer Euler integration steps for the same accuracy. The discretization error of the Euler method scales with the curvature of the trajectory:

$$\|X_1^{\text{approx}} - X_1^{\text{true}}\| \approx O\left(\frac{\text{curvature}}{N}\right)$$

where $N$ is the number of integration steps. Perfectly straight trajectories have zero curvature and allow one-step generation.

## One-Step Distillation

After reflowing to obtain straight trajectories, the model can be distilled into a one-step generator via consistency distillation or by solving the flow ODE once and regressing:

1. Simulate pairs $(Z, X')$ using a multi-step rectified flow
2. Train a model $v_{\text{1step}}$ to directly predict $X'$ from $Z$ in one step

This combines the training stability of flow matching with the inference speed of GANs.

## Connection to Other Frameworks

### Flow Matching

Rectified flow is a special case of **conditional flow matching (CFM)** where the interpolant is linear and coupling is independent. CFM is more general (supports optimal transport couplings and other interpolant families), but rectified flow's simplicity and iterative refinement via reflow make it practically powerful.

### Score-Based Diffusion

Diffusion models use the forward process $X_t = \sqrt{\bar\alpha_t} X_0 + \sqrt{1-\bar\alpha_t} \epsilon$, a cosine or linear schedule with curved paths. Rectified flow's linear interpolant is simpler and requires no noise schedule design.

### DDIM

The deterministic DDIM sampler of diffusion models is analogous to Euler integration of a probability flow ODE. Rectified flow's straight trajectories mean DDIM-style sampling needs far fewer steps.

## Comparison of Sampling Efficiency

| Method | Typical NFEs | FID (ImageNet 256×256) |
| --- | --- | --- |
| DDPM | 1000 | ~3.5 |
| DDIM | 50–100 | ~4.5 |
| Consistency Model | 1–2 | ~3.6 |
| Rectified Flow (1-reflow, 1-step) | 1 | ~5.0 |
| Rectified Flow (2-reflow, 1-step) | 1 | ~3.8 |
| Rectified Flow (multi-step) | 10–25 | ~2.5 |

These are approximate values that vary by implementation and training budget.

## InstaFlow and Large-Scale Applications

**InstaFlow** (Liu et al., 2023) applied rectified flow at scale to Stable Diffusion:

- Used a pre-trained Stable Diffusion model as a teacher
- Applied one reflow step on generated pairs
- Distilled to a one-step model with 0.09-second inference at 512×512

This showed that rectified flow can make billion-parameter text-to-image models generate in a single forward pass.

## SD3 and Stable Diffusion 3

Stability AI's **Stable Diffusion 3** (Esser et al., 2024) adopted rectified flow as its training objective, combined with a multimodal diffusion transformer (DiT) architecture. Key design choices included:

- **Logit-normal time sampling**: sampling $t$ from a logit-normal distribution rather than uniform to weight the loss toward harder timesteps
- **Velocity parameterization**: predicting $v_\theta = X - Z$ directly rather than the noise $\epsilon$
- **Improved scalability**: flow matching training is more stable than score matching at large batch sizes and learning rates

## Theoretical Properties

### Existence of Straight Couplings

The rectified flow operator $\text{Rectify}(\pi_0, \pi_1)$ produces a deterministic transport $T: \mathbb{R}^d \to \mathbb{R}^d$ that is the limit of the reflow procedure. Under mild conditions, this limit is the **monotone transport map** from $\pi_0$ to $\pi_1$, which is the map that minimizes $\mathbb{E}[\|X - Z\|^2]$ (the quadratic OT map).

### Convexity

The quadratic OT map $T^*$ has a convex potential $\psi$: $T^*(z) = \nabla \psi(z)$. This Brenier representation provides a rich geometric structure for analysis.

### Convergence of Reflow

Each reflow step reduces the expected trajectory curvature. Theoretically, after $k$ reflow steps, curvature decreases geometrically in $k$ under regularity assumptions.

## Practical Implementation

A minimal implementation of the training loop is:

```python
import torch

def rectified_flow_loss(model, x1, t):
    # x1: data samples [B, C, H, W]
    # t: timesteps uniformly sampled from [0, 1]
    z = torch.randn_like(x1)                  # noise
    xt = (1 - t[:, None, None, None]) * z + t[:, None, None, None] * x1
    v_target = x1 - z                          # straight-line velocity
    v_pred = model(xt, t)
    return ((v_pred - v_target) ** 2).mean()

def sample(model, z, steps=10):
    dt = 1.0 / steps
    x = z.clone()
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        v = model(x, t)
        x = x + dt * v
    return x
```

## Summary

Rectified Flow provides a clean and powerful framework for generative modeling:

- **Simple objective**: minimize squared error to straight-line velocity targets
- **Reflow**: iterative straightening reduces curvature for fast inference
- **Distillation**: one-step models achievable from straight trajectories
- **Widely adopted**: the foundation of Stable Diffusion 3 and other state-of-the-art text-to-image systems

Its combination of theoretical elegance and practical efficiency has made it one of the dominant paradigms in modern generative model training.
