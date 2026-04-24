---
title: Energy-Based Models
description: Learn how energy-based models (EBMs) define probability distributions through an energy function, enabling flexible density modeling via contrastive divergence, score matching, and MCMC sampling — with deep connections to modern diffusion models.
---

**Energy-based models (EBMs)** define probability distributions implicitly through a scalar **energy function** $E_\theta(x)$, where lower energy corresponds to higher probability. Unlike generative models that define an explicit likelihood (such as normalizing flows or VAEs), EBMs impose almost no architectural constraints — any function, including deep neural networks, can serve as the energy function. This flexibility makes EBMs a powerful, unified framework that underlies many modern generative and discriminative models.

The fundamental challenge of EBMs is computing the **partition function** — the normalizing constant that makes the distribution sum to one — which is generally intractable. The history of EBM research is largely the history of clever approximations that make training practical despite this intractability.

## The Energy-Based Formulation

An EBM defines a probability distribution over data $x \in \mathcal{X}$ as:

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \quad Z(\theta) = \int \exp(-E_\theta(x)) \, dx$$

where $E_\theta : \mathcal{X} \to \mathbb{R}$ is the **energy function** parameterized by $\theta$, and $Z(\theta)$ is the **partition function** (normalization constant).

Data points observed in training receive low energy; unlikely data points receive high energy. The model "learns" the shape of the data distribution by sculpting the energy landscape — creating valleys at high-density regions and hills at low-density regions.

A key feature: any function can be used as $E_\theta$ — convolutional networks, transformers, graph networks. The only requirement is that $\exp(-E_\theta(x))$ is integrable.

## The Intractability of the Partition Function

For continuous data (e.g., images), $Z(\theta)$ is a high-dimensional integral over all possible images — completely intractable to compute directly. This intractability propagates to:

**The likelihood gradient**: Computing $\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z(\theta)$ requires computing $\nabla_\theta \log Z(\theta) = -\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x)]$ — an expectation under the model distribution, itself requiring sampling from the EBM.

**Maximum likelihood training**: Direct maximum likelihood requires computing $Z(\theta)$ at every parameter update — intractable except for small discrete state spaces.

All practical EBM training algorithms are approximations to maximum likelihood that avoid computing $Z(\theta)$ exactly.

## Contrastive Divergence

**Contrastive Divergence (CD)** (Hinton, 2002) is the seminal practical training algorithm for EBMs. The maximum likelihood gradient decomposes as:

$$\nabla_\theta \log p_\theta(x_{data}) = -\nabla_\theta E_\theta(x_{data}) + \mathbb{E}_{p_\theta(x)}[\nabla_\theta E_\theta(x)]$$

This is the difference between:

- The **positive phase**: Push down energy on real data $x_{data}$ (decrease energy where data is).
- The **negative phase**: Push up energy on samples from the model $p_\theta$ (increase energy elsewhere).

The challenge is the negative phase — we need samples from $p_\theta$, which requires running a Markov chain until convergence.

**CD-k** approximates the negative phase by running only $k$ steps of **MCMC** (typically Gibbs sampling or Langevin dynamics) initialized from real data, rather than running the chain to convergence:

```python
def contrastive_divergence_k(model, x_data, k=1, step_size=0.01):
    # Positive phase: energy on real data
    pos_energy = model.energy(x_data)
    
    # Negative phase: run k steps of Langevin MCMC from data
    x_neg = x_data.clone().detach().requires_grad_(True)
    for _ in range(k):
        energy = model.energy(x_neg)
        grad = torch.autograd.grad(energy.sum(), x_neg)[0]
        noise = torch.randn_like(x_neg) * (2 * step_size) ** 0.5
        x_neg = (x_neg - step_size * grad + noise).detach().requires_grad_(True)
    
    neg_energy = model.energy(x_neg)
    
    # Loss: decrease energy on data, increase on negative samples
    loss = (pos_energy - neg_energy).mean()
    return loss, x_neg.detach()
```

CD-k is biased (it doesn't compute the true gradient) but is computationally cheap and works well in practice for models like Restricted Boltzmann Machines (RBMs).

## Persistent Contrastive Divergence (PCD)

**PCD** maintains a persistent buffer of "fantasy particles" — long-running MCMC chains that are continued across training iterations rather than restarted from data:

```python
class PersistentBuffer:
    def __init__(self, buffer_size, data_shape):
        # Initialize buffer with noise
        self.buffer = torch.randn(buffer_size, *data_shape)
    
    def sample_and_update(self, model, n_steps=20, step_size=0.01):
        x = self.buffer.clone().requires_grad_(True)
        for _ in range(n_steps):
            energy = model.energy(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]
            noise = torch.randn_like(x) * (2 * step_size) ** 0.5
            x = (x - step_size * grad + noise).detach().requires_grad_(True)
        self.buffer = x.detach()
        return self.buffer
```

PCD produces better MCMC mixing than CD-k because the chains have time to explore the model distribution. The tradeoff is that during early training, the buffer contains poor-quality samples.

## Score Matching

**Score matching** (Hyvärinen, 2005) is an elegant alternative that avoids sampling entirely. Instead of matching data distribution probabilities, it matches **score functions** — gradients of log-densities:

$$\mathcal{L}_{SM} = \mathbb{E}_{p_{data}}\left[\frac{1}{2}\|\nabla_x \log p_\theta(x)\|^2 + \text{tr}(\nabla_x^2 \log p_\theta(x))\right]$$

Since $\log p_\theta(x) = -E_\theta(x) - \log Z(\theta)$ and $Z(\theta)$ does not depend on $x$:

$$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

The score is the negative energy gradient — the partition function cancels out. Score matching optimizes the score function without ever computing or approximating $Z(\theta)$.

**Denoising Score Matching (DSM)** (Vincent, 2011) provides a practical estimator: add noise to data and train the score function to point from noisy data back to the clean data manifold. This formulation connects EBMs directly to modern diffusion models.

## Connection to Diffusion Models

Diffusion models are intimately connected to EBMs and score matching:

- **Score-based generative models** (Song & Ermon, 2019) model the score function $s_\theta(x) \approx \nabla_x \log p_{data}(x)$ directly and generate samples by running **Langevin dynamics** guided by the score.
- **DDPM** (Ho et al., 2020) implicitly learns a score function through the denoising objective — the denoising network $\epsilon_\theta(x_t, t)$ approximates the score at each noise level.
- The **NCSN** (Noise Conditional Score Network) explicitly trains a score network at multiple noise levels, enabling annealed Langevin sampling from high noise to low noise.

EBMs provide the theoretical foundation for score-based generation — diffusion models can be understood as EBMs trained via denoising score matching with a specific noise schedule and sampling procedure.

## Joint Energy-Based Models (JEMs)

**JEM** (Grathwohl et al., 2020) reinterprets classifier logits as an energy function, turning any classifier into both a generative model and a discriminative model:

$$E_\theta(x) = -\log \sum_y \exp(f_\theta(x)[y])$$

where $f_\theta(x)[y]$ is the logit for class $y$. The joint model $p_\theta(x, y) \propto \exp(f_\theta(x)[y])$ captures both the class-conditional distribution and the marginal data distribution.

JEM is trained with a combined objective:

- **Discriminative**: Standard cross-entropy classification loss.
- **Generative**: Contrastive divergence on the marginal $p_\theta(x)$.

This joint training produces classifiers with improved robustness to adversarial examples, better calibration, and the ability to generate samples — without sacrificing classification accuracy.

## Restricted Boltzmann Machines (RBMs)

**RBMs** are the classical shallow EBMs that inspired modern deep learning:

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^\top W \mathbf{h} - \mathbf{b}^\top \mathbf{v} - \mathbf{c}^\top \mathbf{h}$$

where $\mathbf{v}$ are visible units (data) and $\mathbf{h}$ are hidden units. The bipartite structure makes conditional distributions $p(\mathbf{h} \mid \mathbf{v})$ and $p(\mathbf{v} \mid \mathbf{h})$ tractable — enabling efficient Gibbs sampling.

Stacked RBMs (Deep Belief Networks, Deep Boltzmann Machines) were the primary generative models before VAEs and GANs, and CD-k training of RBMs was Hinton's key contribution that helped revive interest in deep learning.

## Modern EBM Applications

### Anomaly Detection

EBMs naturally quantify anomalies: unusual inputs have high energy (low likelihood). After training on normal data, high-energy inputs are flagged as anomalies — providing principled outlier detection without requiring anomaly examples:

```python
def anomaly_score(model, x, threshold):
    energy = model.energy(x)
    return energy > threshold  # High energy = anomaly
```

### Compositional Generation

EBMs compose naturally: the sum of two energy functions defines a distribution that satisfies both constraints:

$$E_{combined}(x) = E_{concept_1}(x) + E_{concept_2}(x)$$

This compositional property enables zero-shot generation of concept combinations — training EBMs on individual concepts and combining them at test time.

### Structured Prediction

For structured outputs (parse trees, molecule graphs, layouts), energy functions can model complex output dependencies without normalizing over all possible structures at training time.

## Training Stability Challenges

EBM training is notoriously unstable due to:

- **Mode collapse**: MCMC chains may get stuck in modes, producing biased negative samples.
- **Energy drift**: The energy landscape can collapse or diverge without regularization.
- **Gradient explosion/vanishing**: Large energy gradients during MCMC can destabilize training.

Practical stabilization techniques include gradient clipping, buffer warm-up strategies, spectral normalization on the energy network, and regularizing the energy magnitude.

Despite these challenges, EBMs remain a powerful and principled framework — providing both a unifying theoretical foundation for modern generative models and practical tools for applications requiring explicit energy or uncertainty quantification.
