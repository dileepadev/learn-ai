---
title: "Diffusion Forcing: Next-Token Generation Meets Continuous Diffusion"
description: Explore Diffusion Forcing, a hybrid training and sampling paradigm combining sequence-level autoregressive modeling with frame-level continuous diffusion.
---

Generative modeling of sequential continuous data (such as high-framerate video, audio, and robotic trajectories) traditionally faces a sharp tradeoff between two competing paradigms:

1. **Autoregressive (AR) Models:** Generate sequences token-by-token. Excellent for causal decision making and flexible length rollouts, but suffer from compounding exposure bias and quality degradation over long time horizons.
2. **Full-Sequence Diffusion Models:** Generate entire sequences simultaneously by denoising all frames jointly. Produce exceptionally crisp visual quality, but lack real-time streaming capabilities and cannot roll out indefinitely.

**Diffusion Forcing**, proposed by Chen et al. (2024) at MIT CSAIL, unifies these two approaches into a single formulation.

## Core Mechanics of Diffusion Forcing

Diffusion Forcing trains a neural network to denoise **per-frame noise levels independently** along a sequence.

In standard sequence diffusion, every frame in a video clip has the same noise level $k$ during a training step:
$$[x_1^k, x_2^k, x_3^k, \dots, x_T^k]$$

In **Diffusion Forcing**, each frame $t$ is assigned an independent noise level $k_t$:
$$[x_1^{k_1}, x_2^{k_2}, x_3^{k_3}, \dots, x_T^{k_T}]$$

```
Frame Index:       t=1         t=2         t=3         t=4
Noise Level k_t:   k_1 = 0     k_2 = 0     k_3 = 50    k_4 = 100
State:            (Clean)     (Clean)     (Partial)   (Pure Noise)
```

The model predicts the noise for frame $t$ conditioned on past frames, even if those past frames are partially noisy or completely clean.

## Why Independent Noise Levels Matter

Assigning independent noise levels unlocks unprecedented flexibility during sampling:

### 1. Zero Exposure Bias
Because the model is explicitly trained to predict frame $t$ given past frames at varying noise levels, it learns to be robust against errors made in earlier steps of generation.

### 2. Flexible Sampling Strategies
- **Autoregressive Sampling:** Denoise frame $t$ completely before moving to frame $t+1$ (enabling real-time streaming video and online RL world models).
- **Block Parallel Sampling:** Denoise a sliding window of frames simultaneously for high throughput.
- **Full-Sequence Denoising:** Denoise all frames together when full future context is available.

### 3. Variable Horizon Planning & Guidance
Robotic control policies can perform goal-conditioned rollouts by setting the target frame noise to 0 (clean goal state) while intermediate trajectory frames remain noisy, steering the generation process directly toward the target.

## Algorithmic Formulation

Let $x_{1:T}$ be a sequence of continuous frames.

1. **Noise Schedule Sampling:** For each time step $t$, draw an independent diffusion timestep $k_t \sim U(1, K)$.
2. **Corrupting Frames:** Apply Gaussian noise to frame $x_t$ according to $k_t$:
   $$x_t^{k_t} = \sqrt{\bar{\alpha}_{k_t}} x_t + \sqrt{1 - \bar{\alpha}_{k_t}} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$
3. **Sequence Denoising Loss:** The network $f_\theta$ takes the noisy sequence $x_{1:T}^{k_{1:T}}$ and predicts the noise $\epsilon_t$ for each frame:
   $$\mathcal{L}(\theta) = \sum_{t=1}^T \mathbb{E}_{\epsilon_t} \left[ \| \epsilon_t - f_\theta(x_{1:T}^{k_{1:T}}, t, k_{1:T}) \|^2 \right]$$

## Key Applications

### 1. Neural World Models for RL
In reinforcement learning, agents use world models to simulate future states. Diffusion Forcing world models allow agents to simulate infinitely long trajectories without exploding visual drift or state collapse.

### 2. Infinite Video Generation
By streaming frame generation autoregressively with temporal sliding window KV-caching, Diffusion Forcing can stream high-definition video continuously.

### 3. Long-Horizon Robot Trajectory Planning
Robotic agents use Diffusion Forcing to generate smooth kinematic trajectories that satisfy physical obstacles and end-effector target poses simultaneously.

## Comparison with Existing Paradigms

| Property | Autoregressive (e.g. VideoGPT) | Full Diffusion (e.g. Sora, SVD) | Diffusion Forcing |
|---|---|---|---|
| **Data Representation** | Discrete Tokens | Continuous Frames | Continuous Frames |
| **Generation Flow** | Step-by-step causal | Full batch denoising | Arbitrary causal or joint |
| **Streaming Output** | Yes | No | Yes |
| **Exposure Bias** | High | Low | Low |
| **Goal Conditioning** | Difficult | Hard to constrain | Built-in via target masking |

## Summary

Diffusion Forcing bridges continuous diffusion and autoregressive sequence modeling. By decoupling noise levels across sequence time steps during training, it provides the temporal control and stability of diffusion alongside the flexible streaming capabilities of autoregressive models.

## Further Reading

- Chen et al. (2024), *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* (MIT CSAIL)
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models (DDPM)*
- Harvey et al. (2022), *Flexible Diffusion Modeling of Long Sequences*
