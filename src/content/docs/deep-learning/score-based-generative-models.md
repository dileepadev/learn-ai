---
title: Score-Based Generative Models
description: Understand score-based generative models — the alternative probabilistic framework behind modern diffusion models. Learn score matching objectives, Langevin dynamics sampling, noise-conditional score networks (NCSN), stochastic differential equations (SDEs) as a unifying framework, and how denoising score matching connects to DDPM.
---

**Score-based generative models** offer a distinct mathematical perspective on generative modeling compared to GANs, VAEs, or autoregressive models. Instead of learning a probability distribution $p(x)$ directly, they learn its **score function** — the gradient of the log probability with respect to the data:

$$s_\theta(x) \approx \nabla_x \log p(x)$$

This score field points "uphill" in log-probability space — toward regions of higher data density. Given a score function, samples can be drawn by starting from noise and following a gradient ascent path (Langevin dynamics) toward high-density regions. This perspective, pioneered by Yang Song and Stefano Ermon, ultimately unified with DDPM to create the SDE framework underlying modern text-to-image models.

## Why Learn the Score?

Learning $p_\theta(x)$ directly requires a **normalizing constant** $Z = \int p_\theta(x) \, dx$ — intractable in high dimensions. The score function $\nabla_x \log p(x)$ does not involve $Z$:

$$\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z = \nabla_x \log \tilde{p}(x)$$

since $Z$ is a constant with respect to $x$. This makes score learning tractable even when the partition function is not.

## Score Matching

**Explicit score matching** (Hyvärinen, 2005) trains $s_\theta$ to match $\nabla_x \log p_\text{data}(x)$ by minimizing:

$$\mathcal{L}_\text{ESM}(\theta) = \mathbb{E}_{p(x)} \left[ \frac{1}{2} \|s_\theta(x)\|^2 + \text{tr}(\nabla_x s_\theta(x)) \right]$$

The trace term $\text{tr}(\nabla_x s_\theta(x))$ requires second-order derivatives — expensive for high-dimensional data. **Denoising score matching** (DSM, Vincent 2011) avoids this:

$$\mathcal{L}_\text{DSM}(\theta) = \mathbb{E}_{q_\sigma(\tilde{x}|x) p(x)} \left[ \|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)\|^2 \right]$$

where $q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$ is a Gaussian corruption kernel. The optimal denoising score is:

$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2}$$

This is simply the direction pointing from the noisy sample back to the clean sample — making the training signal very interpretable.

## Langevin Dynamics Sampling

Given a trained score network $s_\theta(x) \approx \nabla_x \log p(x)$, samples are generated via **Langevin MCMC**:

$$x_{t+1} = x_t + \frac{\epsilon}{2} s_\theta(x_t) + \sqrt{\epsilon} \, z_t, \quad z_t \sim \mathcal{N}(0, I)$$

As $\epsilon \rightarrow 0$ and $t \rightarrow \infty$, this Markov chain converges to samples from $p(x)$. In practice, finite steps with a carefully tuned step size are used:

```python
import torch
import torch.nn as nn
import numpy as np

class NoisyScoreNetwork(nn.Module):
    """
    Noise-Conditional Score Network (NCSN) — the model at the heart of
    score-based generative models.
    
    Conditions on the noise level σ so a single network serves all noise scales.
    Architecture: U-Net with time/noise-level conditioning via sinusoidal embeddings.
    
    Output is the score function: s_θ(x, σ) ≈ ∇_x log p_σ(x)
    where p_σ(x) = ∫ p(y) N(x; y, σ²I) dy is the data distribution
    corrupted by Gaussian noise of std σ.
    """
    
    def __init__(self, data_dim: int = 784, hidden_dim: int = 256,
                 sigma_embed_dim: int = 32):
        super().__init__()
        
        # Sinusoidal embedding for noise level conditioning
        self.sigma_embed = nn.Sequential(
            SinusoidalEmbedding(sigma_embed_dim),
            nn.Linear(sigma_embed_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, data_dim)
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x: (B, data_dim) — noisy samples
        sigma: (B,) — noise levels for each sample
        Returns: score estimates s_θ(x, σ) with shape (B, data_dim)
        """
        sigma_emb = self.sigma_embed(sigma)
        x_cond = torch.cat([x, sigma_emb], dim=-1)
        
        # Output is rescaled: the network predicts (x_clean - x_noisy) / σ²
        # which equals -σ² × score, so divide by σ² to get score
        raw_output = self.net(x_cond)
        return raw_output / (sigma.unsqueeze(-1) ** 2 + 1e-8)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for scalar conditioning values (noise level, time)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=x.device) * (np.log(10000) / (half - 1))
        )
        emb = x.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def denoising_score_matching_loss(score_net: nn.Module,
                                   x: torch.Tensor,
                                   sigma_min: float = 0.01,
                                   sigma_max: float = 50.0) -> torch.Tensor:
    """
    Denoising score matching objective for NCSN training.
    
    For each clean sample x, sample a noise level σ and perturb x.
    The target score is the direction from noisy to clean, scaled by 1/σ².
    
    Loss is weighted by σ² (Fisher divergence weighting) so that
    the network receives balanced gradients across noise levels.
    """
    batch_size = x.shape[0]
    
    # Sample noise levels log-uniformly between σ_min and σ_max
    log_sigma = torch.zeros(batch_size, device=x.device).uniform_(
        np.log(sigma_min), np.log(sigma_max)
    )
    sigma = log_sigma.exp()  # (B,)
    
    # Corrupt data with Gaussian noise
    noise = torch.randn_like(x)
    x_noisy = x + sigma.unsqueeze(-1) * noise
    
    # Target: score of q_σ(x̃|x) = -(x̃ - x) / σ²
    target_score = -noise / sigma.unsqueeze(-1)   # simplified: noise/σ (before 1/σ² rescaling)
    
    # Predicted score
    pred_score = score_net(x_noisy, sigma)
    
    # Fisher divergence, weighted by σ² for balanced training across noise levels
    weight = sigma ** 2
    loss = (weight.unsqueeze(-1) * (pred_score - target_score) ** 2).mean()
    
    return loss


@torch.no_grad()
def annealed_langevin_sampling(
    score_net: nn.Module,
    shape: tuple,
    sigmas: list[float],
    n_steps: int = 100,
    step_size_factor: float = 0.1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Annealed Langevin dynamics: run Langevin MCMC at decreasing noise levels.
    
    Key insight from NCSN: scores are inaccurate in low-density regions (far from data).
    Start at high noise level (coarse structure) and progressively refine.
    
    This is the inference algorithm for score-based models — the equivalent of
    the DDPM reverse process but derived from Langevin dynamics.
    
    Args:
        sigmas: decreasing sequence of noise levels [σ_L, ..., σ_1, σ_0]
        n_steps: number of Langevin steps per noise level
        step_size_factor: controls step size α = step_size_factor × σ²
    """
    x = torch.randn(*shape, device=device)  # start from isotropic noise
    
    for sigma in sigmas:
        sigma_tensor = torch.full((shape[0],), sigma, device=device)
        
        # Step size for this noise level
        alpha = step_size_factor * sigma ** 2
        
        for _ in range(n_steps):
            score = score_net(x, sigma_tensor)
            noise = torch.randn_like(x)
            
            # Langevin update: gradient step + noise injection
            x = x + alpha * score + (2 * alpha) ** 0.5 * noise
    
    return x
```

## The SDE Unifying Framework

Song et al. (2021) showed that both NCSN (score-based) and DDPM (denoising diffusion) are special cases of **stochastic differential equations** (SDEs). The forward process that gradually corrupts data is an Itô SDE:

$$dx = f(x, t) \, dt + g(t) \, dW$$

where $W$ is a standard Wiener process. The corresponding **reverse-time SDE** generates samples:

$$dx = \left[ f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) \, d\bar{W}$$

The score function $\nabla_x \log p_t(x)$ is the only ingredient needed to run the reverse process. Different choices of $f$ and $g$ recover different model families:

| Forward SDE | Model family | $f(x,t)$ | $g(t)$ |
| --- | --- | --- | --- |
| Variance Exploding (VE) | NCSN / SMLD | $0$ | $\sqrt{\frac{d\sigma^2(t)}{dt}}$ |
| Variance Preserving (VP) | DDPM | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)}$ |
| Sub-VP | Improved DDPM | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t \beta ds})}$ |

## Probability Flow ODE

Every SDE has a corresponding **probability flow ODE** with the same marginal distributions $p_t(x)$ but deterministic trajectories:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

The ODE enables:

- Exact likelihood computation via the instantaneous change of variables formula
- Deterministic encoding and decoding (interpolation in latent space)
- Faster sampling (fewer function evaluations than stochastic sampling)

Solving this ODE with neural ODE solvers (like DPM-Solver, DEIS) achieves high-quality generation in 5–20 steps — the basis of fast SDXL and SD3 inference.

## Connections to DDPM

When parameterized as a noise predictor $\epsilon_\theta(x_t, t)$, the DDPM training objective is equivalent to denoising score matching with:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

This equivalence shows that the Stable Diffusion U-Net is a score network — every text-to-image diffusion model is implicitly learning and evaluating the score function of the data distribution conditioned on text. The score-based framework provides the theoretical grounding for understanding why these models work and how to improve them through better SDE design, sampling algorithms, and guidance techniques.
