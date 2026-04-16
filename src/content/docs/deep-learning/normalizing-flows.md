---
title: Normalizing Flows
description: Understand normalizing flows — a class of generative models that use invertible transformations to learn exact probability densities, enabling both generation and likelihood estimation.
---

**Normalizing flows** are a family of generative models that learn a bijective (invertible), differentiable mapping between a simple base distribution (typically a standard Gaussian) and a complex target distribution. Unlike VAEs or GANs, normalizing flows offer **exact likelihood computation** — making them uniquely powerful for density estimation, anomaly detection, and probabilistic modeling.

## Core Idea

The key insight is the **change of variables formula**. If $z \sim p_z(z)$ is a sample from a simple base distribution and $x = f(z)$ is a bijective transformation, then the density of $x$ is:

$$p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|$$

Or equivalently:

$$\log p_x(x) = \log p_z(z) - \log \left| \det J_f(z) \right|$$

where $J_f(z)$ is the Jacobian of the transformation $f$.

The model is trained to maximize the log-likelihood of data. Because the transformation is invertible, **both generation and inference are exact** — no variational lower bounds or adversarial training required.

## The Normalizing Flow Pipeline

```
Simple distribution (Gaussian) → f₁ → f₂ → ··· → fₖ → Complex distribution (data)
```

Each $f_i$ is an invertible, differentiable **flow layer**. The composition $f = f_k \circ \cdots \circ f_1$ transforms the simple base distribution into the data distribution step by step — "normalizing" it into a tractable form.

## Types of Flows

### Planar and Radial Flows

The earliest flows (Rezende & Mohamed, 2015) applied simple parameterized transformations. Used primarily for variational inference, they have limited expressiveness.

### Autoregressive Flows

Autoregressive flows define each dimension of $x$ as a function of all preceding dimensions and a base variable $z_i$:

$$x_i = g(z_i; h_i(x_{1:i-1}))$$

**Examples:**

- **Masked Autoregressive Flow (MAF)** — Fast density estimation; slow sampling (sequential).
- **Inverse Autoregressive Flow (IAF)** — Fast sampling; slow density estimation.

These are complementary: MAF is suited for training (density estimation), IAF for deployment (fast generation).

### Coupling Layers (RealNVP / Glow)

Coupling layers split the input into two halves. One half passes through unchanged; the other is transformed using a function of the first:

$$y_{1:d} = x_{1:d}$$
$$y_{d+1:D} = x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})$$

The **Jacobian is triangular** — its determinant is just the product of diagonal entries, making it cheap to compute. This allows $s$ and $t$ to be arbitrarily complex neural networks.

- **RealNVP** (Dinh et al., 2016) — Introduced coupling layers for image generation.
- **Glow** (Kingma & Dhariwal, 2018) — Extended RealNVP with 1×1 invertible convolutions for learned channel permutations. First flow model to generate high-resolution, realistic images.

### Continuous Flows: Neural ODEs

**Continuous normalizing flows (CNFs)** parameterize the flow as an ordinary differential equation (ODE):

$$\frac{dz(t)}{dt} = f_\theta(z(t), t)$$

The transformation from $t=0$ to $t=1$ defines the bijection. The log-likelihood change is given by the **instantaneous change of variables**:

$$\frac{d \log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial z(t)}\right)$$

**FFJORD** (Grathwohl et al., 2018) made CNFs practical by estimating the trace using the Hutchinson estimator, avoiding the $O(D^2)$ full Jacobian computation.

## Expressiveness vs. Computational Cost

| Model | Training | Sampling | Exact Likelihood | Expressiveness |
|---|---|---|---|---|
| MAF | Fast | Slow | Yes | High |
| IAF | Slow | Fast | Yes | High |
| RealNVP / Glow | Moderate | Fast | Yes | High |
| FFJORD (CNF) | Slow (ODE solver) | Slow | Yes | Very High |
| VAE | Fast | Fast | No (ELBO) | High |
| GAN | Medium | Fast | No | Very High |

## Applications

### Density Estimation and Anomaly Detection

Flows compute exact log-likelihoods of test points. Points with very low likelihood under the learned distribution are anomalies. This is used in:

- Network intrusion detection.
- Manufacturing defect identification.
- Financial fraud scoring.

### Variational Inference

IAF is used as a flexible posterior in variational autoencoders, replacing the simple diagonal Gaussian posterior and improving VAE expressiveness.

### Generative Modeling

Glow-based models produce realistic high-resolution faces and images with exact likelihood evaluation — directly competing with GANs.

### Scientific Applications

- **Molecular generation** — Flows model 3D molecular geometries for drug discovery.
- **Physics simulations** — Flows learn probability distributions over physical states (e.g., particle physics, climate models).
- **Bayesian inference** — Flows as flexible approximate posteriors in probabilistic programs.

## Limitations

- **Dimensionality** — Flows are expensive for very high-dimensional data (video, large images) due to the invertibility constraint on architectures.
- **Architecture constraints** — The bijection requirement restricts network design choices (no max-pooling, must be square weight matrices for invertible linear layers).
- **Mode covering** — Flows trained by maximum likelihood tend to "cover" all modes of the data distribution, which can blur generated samples if the distribution is multimodal.

Normalizing flows remain a powerful tool in the probabilistic modeling toolkit — particularly when exact likelihood computation, principled uncertainty quantification, or sampling from learned distributions is required.
