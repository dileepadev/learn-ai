---
title: Variational Inference
description: Learn variational inference — the optimization-based approach to approximate Bayesian posterior computation. Covers the Evidence Lower Bound (ELBO), mean-field approximation, black-box variational inference, normalizing flows for expressive posteriors, and the connection to Variational Autoencoders.
---

**Variational inference** (VI) is a method for approximating intractable Bayesian posteriors by turning inference into an optimization problem. Rather than computing the exact posterior $p(\mathbf{z} | \mathbf{x})$ over latent variables $\mathbf{z}$ given observations $\mathbf{x}$ — which is rarely analytically tractable — VI proposes a family of approximate distributions $q_\phi(\mathbf{z})$ and finds the member of that family closest to the true posterior.

VI is central to modern probabilistic machine learning, scaling to large datasets and high-dimensional latent spaces where exact inference (MCMC, sum-product, belief propagation) becomes computationally prohibitive.

## The Posterior Inference Problem

Given a probabilistic model with observed variables $\mathbf{x}$ and latent variables $\mathbf{z}$, Bayes' theorem gives:

$$p(\mathbf{z} | \mathbf{x}) = \frac{p(\mathbf{x} | \mathbf{z})\, p(\mathbf{z})}{p(\mathbf{x})}$$

The denominator — the **marginal likelihood** (evidence) $p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$ — requires integrating over all possible values of $\mathbf{z}$. This integral is analytically intractable for most models of interest (non-conjugate priors, neural network likelihoods, etc.).

## The Evidence Lower Bound (ELBO)

VI replaces exact inference with optimization. We introduce a variational distribution $q_\phi(\mathbf{z})$ parameterized by $\phi$ and minimize its KL divergence from the true posterior:

$$\phi^* = \arg\min_\phi\, \text{KL}\!\left[q_\phi(\mathbf{z}) \,\|\, p(\mathbf{z} | \mathbf{x})\right]$$

Expanding the KL divergence using Bayes' theorem:

$$\text{KL}[q_\phi \| p] = \mathbb{E}_{q_\phi}\!\left[\log q_\phi(\mathbf{z})\right] - \mathbb{E}_{q_\phi}\!\left[\log p(\mathbf{x}, \mathbf{z})\right] + \log p(\mathbf{x})$$

Since $\log p(\mathbf{x})$ is constant with respect to $\phi$, minimizing KL is equivalent to maximizing the **Evidence Lower Bound**:

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log p(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z})\right]$$

Which decomposes as:

$$\mathcal{L}(\phi) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log p(\mathbf{x} | \mathbf{z})\right]}_{\text{reconstruction / expected log-likelihood}} - \underbrace{\text{KL}\!\left[q_\phi(\mathbf{z}) \,\|\, p(\mathbf{z})\right]}_{\text{regularization toward prior}}$$

The ELBO lower-bounds the log marginal likelihood:

$$\log p(\mathbf{x}) \geq \mathcal{L}(\phi)$$

with equality when $q_\phi = p(\mathbf{z} | \mathbf{x})$ exactly. Maximizing the ELBO simultaneously tightens the bound and improves the posterior approximation.

## Mean-Field Variational Inference

The **mean-field assumption** factorizes $q$ across latent variable groups, assuming they are independent:

$$q_\phi(\mathbf{z}) = \prod_{j=1}^{J} q_j(z_j)$$

This is a strong simplification — the true posterior may have complex dependencies between $z_j$ — but it makes the optimization tractable via **Coordinate Ascent VI (CAVI)**:

```python
import numpy as np
from scipy.stats import norm

def cavi_gaussian_mixture(X: np.ndarray, K: int = 3, max_iter: int = 100,
                           tol: float = 1e-4) -> dict:
    """
    Mean-field CAVI for a Gaussian mixture model with K components.
    
    Model:
        z_n ~ Categorical(pi)        (cluster assignment)
        x_n | z_n=k ~ N(mu_k, sigma^2)
    
    Variational family:
        q(z_n) = Categorical(phi_n)   (per-point responsibilities)
        q(mu_k) = N(m_k, s_k^2)      (per-cluster mean)
    """
    N = len(X)
    
    # Initialize variational parameters
    phi = np.random.dirichlet(np.ones(K), size=N)  # (N, K) responsibilities
    m = np.linspace(X.min(), X.max(), K)            # cluster mean estimates
    s2 = np.ones(K)                                 # cluster mean variances
    sigma2 = 1.0                                    # observation noise (fixed)
    pi = np.ones(K) / K                             # mixture weights (fixed)
    
    elbo_history = []
    
    for iteration in range(max_iter):
        # --- Update phi_n (cluster responsibilities) ---
        # E[log q(z_n = k)] ∝ log pi_k + E_q[log p(x_n | mu_k, sigma^2)]
        log_phi = np.zeros((N, K))
        for k in range(K):
            # E_q[(x_n - mu_k)^2] = (x_n - m_k)^2 + s_k^2
            expected_sq = (X - m[k])**2 + s2[k]
            log_phi[:, k] = (np.log(pi[k])
                             - 0.5 * np.log(2 * np.pi * sigma2)
                             - expected_sq / (2 * sigma2))
        # Softmax normalization
        log_phi -= log_phi.max(axis=1, keepdims=True)
        phi = np.exp(log_phi)
        phi /= phi.sum(axis=1, keepdims=True)
        
        # --- Update q(mu_k) = N(m_k, s_k^2) ---
        for k in range(K):
            N_k = phi[:, k].sum()
            # Posterior precision = prior precision + likelihood precision
            s2[k] = 1.0 / (1.0 + N_k / sigma2)
            # Posterior mean = s2_k * (prior_mean/prior_var + sum phi_nk * x_n / sigma2)
            m[k] = s2[k] * (phi[:, k] @ X) / sigma2
        
        # --- Compute ELBO ---
        elbo = 0.0
        for k in range(K):
            N_k = phi[:, k].sum()
            expected_sq = (X - m[k])**2 + s2[k]
            elbo += np.sum(phi[:, k] * (
                np.log(pi[k] + 1e-12)
                - 0.5 * expected_sq / sigma2
            ))
            # Entropy of q(mu_k)
            elbo += 0.5 * np.log(2 * np.pi * np.e * s2[k])
        # Entropy of q(z_n)
        elbo -= np.sum(phi * np.log(phi + 1e-12))
        elbo_history.append(elbo)
        
        if iteration > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            print(f"Converged at iteration {iteration}")
            break
    
    return {
        "responsibilities": phi,
        "cluster_means": m,
        "cluster_mean_variances": s2,
        "elbo_history": elbo_history
    }
```

## Black-Box Variational Inference (BBVI)

CAVI requires deriving model-specific update equations. **Black-box VI** (Ranganath et al., 2014) computes ELBO gradients using Monte Carlo, making it applicable to any differentiable model:

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{q_\phi}\!\left[\nabla_\phi \log q_\phi(\mathbf{z}) \cdot \left(\log p(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z})\right)\right]$$

The **score function estimator** (REINFORCE gradient) enables this. In practice, the **reparameterization trick** provides lower-variance gradients when $q_\phi$ allows sampling via $\mathbf{z} = g(\phi, \epsilon)$ with $\epsilon \sim p(\epsilon)$:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class VariationalPosterior(nn.Module):
    """
    Diagonal Gaussian variational posterior q(z|x) = N(mu, diag(sigma^2)).
    Supports reparameterized sampling for low-variance gradient estimates.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(latent_dim))  # log for positivity

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def rsample(self, n_samples: int = 1) -> torch.Tensor:
        """Reparameterized sample: z = mu + sigma * eps, eps ~ N(0, I)."""
        eps = torch.randn(n_samples, self.latent_dim)
        return self.mu + self.sigma * eps  # (n_samples, latent_dim)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Log probability under the variational distribution."""
        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z).sum(-1)

    def kl_to_standard_normal(self) -> torch.Tensor:
        """
        Closed-form KL divergence: KL[N(mu, sigma^2) || N(0, I)]
        = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        """
        return 0.5 * (self.mu**2 + self.sigma**2 - 2 * self.log_sigma - 1).sum()


def train_bbvi(
    log_likelihood_fn,   # callable: z (tensor) -> log p(x|z) scalar
    latent_dim: int = 10,
    n_steps: int = 1000,
    n_samples: int = 10,
    lr: float = 1e-3
) -> VariationalPosterior:
    """
    Train a diagonal Gaussian variational posterior using BBVI with
    reparameterized gradients.
    """
    q = VariationalPosterior(latent_dim)
    optimizer = optim.Adam(q.parameters(), lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        z = q.rsample(n_samples)              # (n_samples, latent_dim)
        
        # ELBO = E_q[log p(x|z)] - KL[q || p]
        log_lik = torch.stack([log_likelihood_fn(z[i]) for i in range(n_samples)])
        expected_log_lik = log_lik.mean()
        kl = q.kl_to_standard_normal()
        
        elbo = expected_log_lik - kl
        loss = -elbo   # maximize ELBO = minimize negative ELBO
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: ELBO = {elbo.item():.3f}")
    
    return q
```

## The Reparameterization Trick

For distributions where we can write $\mathbf{z} = g_\phi(\boldsymbol\epsilon)$ with $\boldsymbol\epsilon$ drawn from a fixed noise distribution:

$$\mathbb{E}_{q_\phi(\mathbf{z})}\![f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol\epsilon)}\![f(g_\phi(\boldsymbol\epsilon))]$$

Gradients now flow through $g_\phi$ directly, yielding much lower variance than the score function estimator. This is the key trick that makes **Variational Autoencoders** (VAEs) trainable end-to-end via backpropagation.

## Expressive Posteriors with Normalizing Flows

Mean-field diagonal Gaussians often fail to capture posterior correlations. **Normalizing flows** compose a sequence of invertible transformations to turn a simple base distribution into an expressive posterior:

$$\mathbf{z}_K = f_K \circ \cdots \circ f_1(\mathbf{z}_0), \quad \mathbf{z}_0 \sim q_0$$

The log density transforms as:

$$\log q_K(\mathbf{z}_K) = \log q_0(\mathbf{z}_0) - \sum_{k=1}^{K} \log\left|\det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right|$$

Popular flow layers include **Planar Flows**, **Affine Coupling Layers** (RealNVP), and **Autoregressive Flows** (IAF). Each trade off expressivity against computational cost of the Jacobian determinant.

## Connection to Variational Autoencoders

VAEs (Kingma & Welling, 2013) instantiate variational inference with neural network encoders and decoders:

- **Encoder** $q_\phi(\mathbf{z} | \mathbf{x})$: maps input to posterior parameters $(\boldsymbol\mu, \log \boldsymbol\sigma^2)$.
- **Decoder** $p_\theta(\mathbf{x} | \mathbf{z})$: reconstructs input from sampled latent code.
- **ELBO** = reconstruction loss − KL divergence from standard Gaussian prior.

The reparameterization trick enables end-to-end gradient-based training of both networks simultaneously.

| Method | Approximation quality | Scalability | Setup effort |
|---|---|---|---|
| **Exact (conjugate)** | Exact | Low | Low |
| **MCMC (HMC/NUTS)** | Asymptotically exact | Moderate | Moderate |
| **Mean-field VI (CAVI)** | Limited (no correlations) | High | High (per model) |
| **BBVI** | Moderate | High | Low |
| **Flow-based VI** | High | Moderate | Moderate |
| **Laplace approximation** | Local Gaussian | High | Low |

Variational inference is the workhorse of scalable probabilistic deep learning, forming the theoretical backbone of VAEs, diffusion models, and many Bayesian neural network approaches.
