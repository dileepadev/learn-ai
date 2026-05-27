---
title: Variational Autoencoders
description: Understand Variational Autoencoders (VAEs) from the ground up — the ELBO objective, the reparameterization trick that enables backpropagation through stochastic nodes, disentangled representations with beta-VAE and FactorVAE, hierarchical VAEs, and connections to diffusion models and modern generative modeling.
---

Variational Autoencoders (VAEs) introduced a principled probabilistic framework for learning generative models with continuous latent spaces. Unlike regular autoencoders that map inputs to fixed point embeddings, VAEs learn a distribution over latent codes — making it possible to sample new data by drawing from that distribution. This probabilistic perspective connects deep learning with Bayesian inference and laid groundwork for modern generative models including diffusion models.

## The Generative Model

A VAE defines a joint distribution over observed data $\mathbf{x}$ and latent variables $\mathbf{z}$:

$$p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})$$

where:

- $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ — standard Gaussian prior over latent codes
- $p_\theta(\mathbf{x} \mid \mathbf{z})$ — the **decoder**, parameterized by a neural network with weights $\theta$, mapping latent codes to data distributions (typically Gaussian or Bernoulli)

The goal is to learn $\theta$ to maximize the marginal likelihood $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$. This integral is intractable — we cannot sum over all possible latent codes.

## The Evidence Lower Bound (ELBO)

To make training tractable, VAEs introduce an **encoder** (inference network) $q_\phi(\mathbf{z} \mid \mathbf{x})$ — a distribution over latent codes given an observed input. Using Jensen's inequality:

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right] - D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \;\|\; p(\mathbf{z})\right)$$

This is the **ELBO** (Evidence Lower BOund) $\mathcal{L}(\theta, \phi; \mathbf{x})$. It decomposes into:

- **Reconstruction term**: $\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$ — how well the decoder reconstructs $\mathbf{x}$ from latent codes sampled using the encoder
- **KL regularization term**: $-D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$ — penalizes the encoder for producing a posterior that differs from the prior, encouraging a compact, continuous latent space

For a Gaussian encoder $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}),\, \mathrm{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$ and Gaussian prior, the KL term has a closed form:

$$D_{\mathrm{KL}} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

## The Reparameterization Trick

The reconstruction term requires sampling $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})$, which is a non-differentiable operation. The **reparameterization trick** resolves this by expressing the sample as a deterministic function of the parameters and a separate noise variable:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now gradients flow through $\boldsymbol{\mu}_\phi$ and $\boldsymbol{\sigma}_\phi$ during backpropagation — the randomness is in $\boldsymbol{\epsilon}$, which has no parameters.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()

        # Encoder: x → (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → x̂
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # for [0,1] pixel values
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # deterministic at inference

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def elbo_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    Reconstruction loss (binary cross-entropy) + beta * KL divergence.
    beta=1: standard VAE; beta>1: beta-VAE for disentanglement.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kl) / x.size(0)


# Training loop
model = VAE(input_dim=784, latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for x_batch, _ in dataloader:
        x_flat = x_batch.view(-1, 784)
        recon, mu, logvar = model(x_flat)
        loss = elbo_loss(recon, x_flat, mu, logvar, beta=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Generative Sampling and Interpolation

The structured latent space enables two powerful operations:

```python
# Sample new images
model.eval()
with torch.no_grad():
    z = torch.randn(64, 32)  # 64 samples from prior
    samples = model.decode(z)

# Interpolate between two images
z1, _ = model.encode(x1.view(1, 784))
z2, _ = model.encode(x2.view(1, 784))
alphas = torch.linspace(0, 1, 10)
interpolations = [model.decode((1 - a) * z1 + a * z2) for a in alphas]
```

## Disentangled Representations

A disentangled representation is one where individual latent dimensions correspond to interpretable, independent factors of variation (e.g., one dimension for object size, another for rotation). Standard VAEs do not guarantee disentanglement.

### beta-VAE

Higgins et al. (2017) proposed multiplying the KL term by $\beta > 1$, increasing pressure to use the prior efficiently. Higher $\beta$ forces the encoder to compress information more aggressively, often resulting in disentangled latent dimensions at the cost of reconstruction quality:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - \beta\, D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$$

### FactorVAE and TC-VAE

FactorVAE (Kim & Mnih, 2018) and TC-VAE (Chen et al., 2018) decompose the KL term into total correlation — a measure of statistical dependence between dimensions — and penalize it directly. This gives finer control over disentanglement than the blunt $\beta$ penalty.

## Hierarchical VAEs

Standard VAEs with a single latent layer cannot capture complex hierarchical structure in data. **Hierarchical VAEs** stack multiple latent layers:

$$p(\mathbf{z}_1, \ldots, \mathbf{z}_L, \mathbf{x}) = p(\mathbf{x} \mid \mathbf{z}_1) \prod_{l=1}^{L-1} p(\mathbf{z}_l \mid \mathbf{z}_{l+1}) \cdot p(\mathbf{z}_L)$$

**NVAE** (Vahdat & Kautz, 2020) uses residual cells with batch normalization across 30+ latent groups, achieving near-diffusion-quality image generation. **VDVAE** (Child, 2021) demonstrated that very deep hierarchical VAEs (≥ 75 layers) could match GAN quality on many benchmarks.

## Connections to Modern Generative Models

VAEs are the conceptual predecessor to several modern approaches:

- **VQ-VAE** replaces the continuous Gaussian latent with a discrete codebook — enabling language-model-style generation over discrete tokens (see image tokenization)
- **Latent Diffusion Models** (Stable Diffusion) train a diffusion model in the compressed latent space of a VAE encoder — combining the structured latent space of VAEs with the generative quality of diffusion
- **Flow Matching** and normalizing flows provide another route to tractable exact likelihoods in generative models

| Property | VAE | GAN | Diffusion |
| --- | --- | --- | --- |
| Training stability | High | Low | High |
| Sample quality | Moderate | High | Very high |
| Latent space | Continuous, structured | Unstructured | None (or latent) |
| Mode coverage | Good | Mode dropping | Good |
| Inference speed | Fast | Fast | Slow (many steps) |
| Likelihood | Approximate (ELBO) | None | Exact (SDE) |

## Summary

VAEs established a principled probabilistic framework for generative modeling with continuous latent spaces:

- The **ELBO objective** makes maximum likelihood tractable by introducing an inference network and lower-bounding the log-likelihood
- The **reparameterization trick** enables gradient flow through stochastic sampling nodes, making end-to-end training possible
- The **KL regularization** structures the latent space to be continuous and approximately Gaussian — enabling smooth interpolation and meaningful sampling from the prior
- **beta-VAE** and its variants encourage disentangled latent dimensions by increasing pressure on the KL term
- **Hierarchical VAEs** stack multiple latent layers to capture complex structure, approaching GAN-quality generation
- VAEs provided the foundation for latent diffusion models — the architecture behind Stable Diffusion and most modern text-to-image systems
