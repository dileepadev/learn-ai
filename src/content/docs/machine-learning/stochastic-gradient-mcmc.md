---
title: Stochastic Gradient MCMC
description: Understand stochastic gradient Markov chain Monte Carlo — covering SGLD, SGHMC, SGNHT, cyclical SGLD, and how these methods enable scalable Bayesian inference in neural networks by combining the efficiency of mini-batch SGD with the statistical guarantees of MCMC sampling.
---

Bayesian inference requires sampling from the posterior distribution $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta)$. Classical MCMC methods like Metropolis-Hastings and Hamiltonian Monte Carlo (HMC) are statistically rigorous but require full-data gradient evaluations and expensive acceptance steps — prohibitively slow for modern neural networks with millions of parameters and massive datasets.

**Stochastic Gradient MCMC (SG-MCMC)** combines the scalability of mini-batch stochastic gradient descent with the asymptotic correctness of MCMC, enabling principled posterior sampling in large-scale settings.

## From Langevin Dynamics to SGLD

### Langevin Dynamics

Langevin dynamics is a continuous-time stochastic differential equation (SDE) whose stationary distribution is the target posterior:

$$d\theta = \frac{1}{2} \nabla_\theta \log p(\theta \mid \mathcal{D}) \, dt + dW_t$$

where $W_t$ is a Wiener process (Brownian motion). Discretizing with step size $\epsilon$:

$$\theta_{t+1} = \theta_t + \frac{\epsilon}{2} \nabla_\theta \log p(\theta \mid \mathcal{D}) + \sqrt{\epsilon} \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, I)$$

This resembles gradient descent with added Gaussian noise. With the full dataset gradient and a Metropolis-Hastings correction step, this produces exact samples from $p(\theta \mid \mathcal{D})$ as $\epsilon \to 0$.

### Stochastic Gradient Langevin Dynamics (SGLD)

**SGLD** (Welling & Teh, 2011) replaces the full-data gradient with a mini-batch estimate:

$$\nabla_\theta \log p(\theta \mid \mathcal{D}) \approx \frac{N}{n} \sum_{i \in \mathcal{B}} \nabla_\theta \log p(x_i \mid \theta) + \nabla_\theta \log p(\theta)$$

where $N$ is the dataset size and $n = |\mathcal{B}|$ is the mini-batch size. The SGLD update becomes:

$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2} \tilde{g}_t + \sqrt{\epsilon_t} \, \eta_t$$

where $\tilde{g}_t$ is the mini-batch gradient estimate. The injected noise compensates for the gradient noise from mini-batching — no Metropolis correction is needed when $\epsilon_t \to 0$.

**Key insight**: standard SGD with a decaying learning rate approaches a point estimate; SGLD with an appropriately small (but non-zero) step size maintains a distribution over parameter space.

```python
import torch
from torch.optim import Optimizer

class SGLD(Optimizer):
    def __init__(self, params, lr, noise_scale=1.0):
        defaults = {"lr": lr, "noise_scale": noise_scale}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            noise_scale = group["noise_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Gradient step
                p.add_(p.grad, alpha=lr / 2)
                # Langevin noise injection
                noise = torch.randn_like(p) * (lr * noise_scale) ** 0.5
                p.add_(noise)

# Usage: treat like Adam/SGD in training loop
optimizer = SGLD(model.parameters(), lr=1e-4)
for x_batch, y_batch in dataloader:
    loss = criterion(model(x_batch), y_batch) + prior_log_prob(model) / N
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Step Size Schedule

Unlike SGD where the learning rate is tuned for convergence, SGLD's step size governs the quality of the posterior approximation:

- **Too large**: dominated by discretization error — samples are biased
- **Too small**: slow mixing — samples are highly correlated

Common schedule: $\epsilon_t = a(b + t)^{-\gamma}$ with $\gamma \in (0.5, 1)$, ensuring $\sum_t \epsilon_t = \infty$ (explore fully) and $\sum_t \epsilon_t^2 < \infty$ (noise eventually vanishes).

## Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)

Plain SGLD's noise injection is isotropic and independent at each step. **Stochastic Gradient HMC** (Chen et al., 2014) augments the system with momentum variables, enabling coherent long-range exploration of the posterior:

$$d\theta = M^{-1} r \, dt$$
$$dr = -\nabla_\theta U(\theta) \, dt - C M^{-1} r \, dt + \sqrt{2(C - \hat{B})} \, dW_t$$

where:

- $r$: momentum variable
- $U(\theta) = -\log p(\theta \mid \mathcal{D})$: potential energy (negative log-posterior)
- $C$: friction matrix (damping term)
- $\hat{B}$: estimate of the stochastic gradient noise covariance

The friction term $C M^{-1} r$ dissipates energy injected by gradient noise, maintaining the correct stationary distribution.

```python
class SGHMC(Optimizer):
    def __init__(self, params, lr, noise_scale=1.0, alpha=0.1):
        # alpha: friction coefficient
        defaults = {"lr": lr, "noise_scale": noise_scale, "alpha": alpha}
        super().__init__(params, defaults)
        # Initialize momentum
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["momentum"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            noise_scale = group["noise_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p]["momentum"]
                # Momentum update with friction and noise
                noise = (2 * alpha * lr * noise_scale) ** 0.5 * torch.randn_like(p)
                m.mul_(1 - alpha).add_(p.grad, alpha=-lr).add_(noise)
                # Parameter update
                p.add_(m)
```

SGHMC mixes faster than SGLD because momentum carries the sampler through low-probability regions between modes.

## SGNHT: Nosé-Hoover Thermostat

Standard SGHMC requires estimating the stochastic gradient noise covariance $\hat{B}$, which is difficult in practice. **SGNHT** (Ding et al., 2014) introduces a thermostatic variable $\xi$ that automatically controls the temperature:

$$d\theta = M^{-1} r \, dt$$
$$dr = -\nabla_\theta U(\theta) \, dt - \xi r \, dt + \sqrt{2A} \, dW_t$$
$$d\xi = \left(\frac{r^T M^{-1} r}{d} - 1\right) dt$$

The thermostat $\xi$ adapts to maintain the correct temperature without explicit noise covariance estimation, making SGNHT more robust in practice.

## Cyclical SGLD (cSGLD)

A major challenge for SG-MCMC is **posterior multimodality** — modern neural network posteriors have many isolated modes, and a single chain cannot traverse them in reasonable time.

**Cyclical SGLD** (Zhang et al., 2020) uses a cyclical learning rate schedule instead of a monotonically decaying one:

$$\epsilon_t = \frac{\epsilon_0}{2}\left(1 - \cos\left(\frac{\pi (t \bmod T)}{T}\right)\right) + \epsilon_{\min}$$

Each cycle has two phases:

1. **Exploration phase** (high $\epsilon$): large step size allows the chain to escape local modes
1. **Sampling phase** (low $\epsilon$): small step size collects approximate posterior samples within the current mode

Samples collected at the end of each cycle's sampling phase are kept. The ensemble of samples from multiple cycles approximates a **multi-modal posterior**, capturing uncertainty across different modes.

```python
import math

class CyclicalSGLD(SGLD):
    def __init__(self, params, lr_max=1e-3, lr_min=1e-7, cycle_length=200, explore_fraction=0.8):
        super().__init__(params, lr=lr_max)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.cycle_length = cycle_length
        self.explore_fraction = explore_fraction
        self._step = 0

    def is_sampling_phase(self):
        cycle_position = self._step % self.cycle_length
        return cycle_position >= int(self.explore_fraction * self.cycle_length)

    @torch.no_grad()
    def step(self):
        # Update learning rate according to cyclical schedule
        cycle_position = self._step % self.cycle_length
        fraction = cycle_position / self.cycle_length
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 - math.cos(math.pi * fraction)
        )
        for group in self.param_groups:
            group["lr"] = lr
        self._step += 1
        super().step()
```

## Collecting and Using Posterior Samples

SG-MCMC produces a sequence of parameter vectors $\{\theta_1, \theta_2, \ldots, \theta_T\}$. Predictions average over collected samples:

$$p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T p(y^* \mid x^*, \theta_t)$$

```python
samples = []
optimizer = CyclicalSGLD(model.parameters())

for step, (x, y) in enumerate(train_loader):
    loss = compute_loss(model, x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Collect sample at the end of each cycle's sampling phase
    if optimizer.is_sampling_phase() and step % collect_every == 0:
        samples.append({k: v.clone() for k, v in model.state_dict().items()})

# Bayesian ensemble prediction
def bayesian_predict(x, samples, model):
    preds = []
    for sample in samples:
        model.load_state_dict(sample)
        with torch.no_grad():
            preds.append(torch.softmax(model(x), dim=-1))
    return torch.stack(preds).mean(0)
```

## Comparison: SG-MCMC vs Deep Ensembles vs Variational Inference

| Property | SGLD / SGHMC | cSGLD | Deep Ensembles | Variational BNN |
| --- | --- | --- | --- | --- |
| Theoretical basis | MCMC | MCMC | None (heuristic) | Variational Bayes |
| Multi-modal coverage | Poor (single chain) | Good (cyclical) | Good (independent inits) | Poor (mode-seeking) |
| Extra parameters | 0 | 0 | $M \times$ params | $2\times$ params |
| Training cost | $1\times$ (+ longer) | $1\times$ | $M\times$ | $1\times$ |
| Calibration quality | Good | Very good | Best | Moderate |
| Implementation | Easy | Easy | Easy | Hard |

## Practical Considerations

### Burn-in Period

Early SG-MCMC samples are biased by the initialization. Discard the first $B$ steps (burn-in) before collecting samples:

- Typical burn-in: 20–50% of total training steps
- Monitor a diagnostic (e.g., loss value stability) to determine when the chain has mixed

### Thinning

Consecutive SG-MCMC samples are highly autocorrelated. Collect every $k$-th sample to reduce storage while maintaining diversity.

### Data Subsampling Bias

SG-MCMC is asymptotically correct only as step size $\to 0$, but the stochastic gradient introduces a bias at finite step sizes. In practice, this bias is acceptable when using sufficiently small step sizes in the sampling phase.

## Summary

Stochastic gradient MCMC brings the statistical rigor of Bayesian inference to large-scale deep learning:

- **SGLD** adds calibrated Langevin noise to SGD updates, turning optimization into posterior sampling
- **SGHMC** adds momentum for faster mixing and better traversal of the posterior landscape
- **SGNHT** eliminates the need to estimate gradient noise covariance via an adaptive thermostat
- **Cyclical SGLD** uses periodic large learning rates to escape modes, collecting diverse multi-modal posterior samples

Compared to variational methods, SG-MCMC requires no architectural changes and makes no mean-field assumptions. Compared to deep ensembles, it achieves similar or better uncertainty quantification at a fraction of the training cost. For practitioners seeking principled uncertainty estimates in neural networks without the complexity of full variational inference, cyclical SGLD represents the current practical optimum.
