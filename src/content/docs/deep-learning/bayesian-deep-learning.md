---
title: Bayesian Deep Learning
description: Explore Bayesian deep learning — the principled framework for uncertainty quantification in neural networks — covering variational inference, Monte Carlo Dropout, Laplace approximation, deep ensembles, and calibration metrics for reliable predictions.
---

Standard neural networks produce point estimates: given an input, they output a single prediction with no indication of how confident that prediction should be. Bayesian deep learning treats network weights as random variables, placing priors over them and computing posterior distributions that encode uncertainty. This enables models that say "I don't know" rather than quietly producing confident-but-wrong predictions.

## Why Uncertainty Matters

In safety-critical applications — medical diagnosis, autonomous driving, financial risk — knowing when a model is uncertain is as important as knowing its prediction. Uncertainty quantification (UQ) enables:

- **Out-of-distribution detection**: flag inputs far from training data
- **Active learning**: query labels for maximally informative examples
- **Safe exploration**: avoid high-uncertainty states in reinforcement learning
- **Calibrated decision-making**: risk-adjusted choices with confidence intervals

Neural networks without UQ are frequently **overconfident** — assigning near-100% probability to incorrect predictions on OOD inputs.

## The Bayesian Formulation

Given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the Bayesian framework seeks the posterior over weights $\mathbf{w}$:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathcal{D})}$$

Prediction for a new input $x^*$ marginalizes over the posterior:

$$p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \mathbf{w}) \, p(\mathbf{w} \mid \mathcal{D}) \, d\mathbf{w}$$

The posterior $p(\mathbf{w} \mid \mathcal{D})$ is **intractable** for neural networks — the evidence $p(\mathcal{D})$ requires integrating over all possible weight configurations, which is computationally infeasible. Practical methods use approximations.

## Variational Inference

Variational inference (VI) approximates the intractable posterior $p(\mathbf{w} \mid \mathcal{D})$ with a tractable distribution $q_\phi(\mathbf{w})$ parameterized by $\phi$, minimizing the KL divergence:

$$\mathcal{L}(\phi) = -\underbrace{\mathbb{E}_{q_\phi(\mathbf{w})}[\log p(\mathcal{D} \mid \mathbf{w})]}_{\text{expected log-likelihood}} + \underbrace{\text{KL}[q_\phi(\mathbf{w}) \| p(\mathbf{w})]}_{\text{complexity penalty}}$$

This is the **Evidence Lower Bound (ELBO)** negated. Minimizing $\mathcal{L}$ maximizes the ELBO.

### Mean-Field Approximation

The simplest choice is a fully factored (mean-field) Gaussian:

$$q_\phi(\mathbf{w}) = \prod_i \mathcal{N}(w_i; \mu_i, \sigma_i^2)$$

Each weight $w_i$ has an independent mean $\mu_i$ and variance $\sigma_i^2$. The weight count doubles — every parameter has a mean and variance — increasing memory and compute by $2\times$.

### Reparameterization Trick

Backpropagating through the sampling operation $w \sim q_\phi(w)$ requires the reparameterization trick:

$$w = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This expresses the sample as a deterministic function of $\mu$, $\sigma$, and a noise variable $\epsilon$, making gradients flow through $\mu$ and $\sigma$.

```python
import torch
import torch.nn as nn

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x):
        # Softplus to ensure positive variance
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # Reparameterization
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        # Analytical KL for Gaussian prior N(0, 1)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl_w = -0.5 * (1 + 2 * torch.log(weight_sigma) - self.weight_mu**2 - weight_sigma**2).sum()
        return kl_w
```

## Monte Carlo Dropout

**MC Dropout** (Gal & Ghahramani, 2016) is the most widely used Bayesian approximation. The insight: applying dropout at inference time and averaging predictions across multiple stochastic forward passes approximates a Gaussian process posterior.

No architectural changes are required if dropout is already present in the network — just keep dropout active at test time.

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # Active at train and test time
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def mc_dropout_predict(model, x, num_samples=50):
    model.train()  # Keep dropout active
    predictions = torch.stack([model(x) for _ in range(num_samples)])
    # predictions: (num_samples, batch_size, output_dim)
    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    return mean, variance
```

The predictive mean is the average prediction; the variance captures **epistemic uncertainty** (uncertainty due to limited data). MC Dropout is cheap, model-agnostic, and requires no changes to training — making it the default first approach for UQ in deep learning.

### Limitation

MC Dropout underestimates uncertainty in practice compared to full Bayesian methods. It also requires multiple forward passes at inference — $50\times$ compute overhead for 50 samples.

## Laplace Approximation

The Laplace approximation fits a Gaussian centered at the MAP estimate:

$$p(\mathbf{w} \mid \mathcal{D}) \approx \mathcal{N}(\mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1})$$

where $\mathbf{H} = -\nabla^2 \log p(\mathbf{w}_{\text{MAP}} \mid \mathcal{D})$ is the Hessian of the negative log-posterior. The predictive distribution is:

$$p(y^* \mid x^*, \mathcal{D}) \approx \int p(y^* \mid x^*, \mathbf{w}) \, \mathcal{N}(\mathbf{w}; \mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1}) \, d\mathbf{w}$$

### Practical Implementation

Computing the full Hessian requires $O(|\mathbf{w}|^2)$ memory — infeasible for large networks. **Kronecker-factored Laplace (KFLA)** and **diagonal Laplace** use structured approximations:

```python
from laplace import Laplace

# Standard MAP training first
model = train_map(train_loader)

# Apply Laplace approximation post-hoc
la = Laplace(model, "classification",
             subset_of_weights="last_layer",  # Only last layer for efficiency
             hessian_structure="kron")        # Kronecker-factored Hessian

la.fit(train_loader)
la.optimize_prior_precision()  # Marginal likelihood tuning

# Uncertainty-aware predictions
pred_mean, pred_var = la(x_test, pred_type="glm", link_approx="probit")
```

Laplace approximation is attractive because it repurposes any pre-trained MAP model without retraining. **Last-layer Laplace** applies the approximation only to the final classification head, reducing cost while capturing most of the useful uncertainty.

## Deep Ensembles

**Deep ensembles** (Lakshminarayanan et al., 2017) train $M$ independent networks from different random initializations and aggregate their predictions:

$$p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p_{\theta_m}(y^* \mid x^*)$$

For regression, each network outputs a mean and variance, and the ensemble combines them:

$$\mu_* = \frac{1}{M} \sum_m \mu_m, \quad \sigma_*^2 = \frac{1}{M} \sum_m (\sigma_m^2 + \mu_m^2) - \mu_*^2$$

```python
import torch
import torch.nn as nn

class Ensemble:
    def __init__(self, model_class, num_models=5, **kwargs):
        self.models = [model_class(**kwargs) for _ in range(num_models)]

    def train_all(self, train_loader, epochs):
        for model in self.models:
            train_single(model, train_loader, epochs)

    def predict(self, x):
        preds = [model(x) for model in self.models]
        means = torch.stack([p[0] for p in preds])
        variances = torch.stack([p[1] for p in preds])

        ensemble_mean = means.mean(dim=0)
        # Law of total variance
        ensemble_var = (variances + means**2).mean(dim=0) - ensemble_mean**2
        return ensemble_mean, ensemble_var
```

Despite their simplicity, deep ensembles consistently outperform variational methods and MC Dropout on calibration benchmarks. Their main drawback is $M\times$ memory and compute cost.

## SWA-Gaussian (SWAG)

**Stochastic Weight Averaging-Gaussian** approximates the posterior by collecting weight snapshots during SGD's later training phase:

1. Run SGD past convergence, collecting weight iterates $\mathbf{w}_1, \ldots, \mathbf{w}_T$
1. Compute mean $\bar{\mathbf{w}}$ and covariance structure from iterates
1. At test time, sample multiple weight vectors from the fitted Gaussian

SWAG captures the flat loss basin explored by SGD's noise — a region where many weight configurations achieve similar training loss but differ in generalization. Uncertainty estimates arise from this basin width.

## Epistemic vs Aleatoric Uncertainty

| Type | Source | Reducible? | Example |
| --- | --- | --- | --- |
| Epistemic | Model uncertainty (limited data) | Yes — more data reduces it | Low training-data region |
| Aleatoric | Data noise (inherent randomness) | No — irreducible | Sensor noise, label ambiguity |

Decomposing uncertainty into these types enables targeted interventions: epistemic uncertainty drives data collection; aleatoric uncertainty informs risk assessment.

For regression networks with predicted variance $\sigma^2(x)$:

- **Aleatoric**: the predicted $\sigma^2(x)$ from a single model (captures input-dependent noise)
- **Epistemic**: the disagreement across ensemble members / MC samples in $\mu(x)$

## Calibration Metrics

A model is **calibrated** if its confidence matches empirical accuracy. A model claiming 80% confidence should be correct 80% of the time.

### Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

where bins $B_b$ group predictions by confidence level.

### Reliability Diagram

Plot accuracy vs confidence — a perfectly calibrated model lies on the diagonal:

```python
import numpy as np
import matplotlib.pyplot as plt

def reliability_diagram(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            accs.append(y_true[mask].mean())
            confs.append(y_prob[mask].mean())

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.bar(confs, accs, width=0.1, alpha=0.7)
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
```

### Temperature Scaling

Post-hoc calibration divides logits by a scalar temperature $T$ learned on a validation set:

$$p_i = \text{softmax}(z_i / T)$$

$T > 1$ softens the distribution (reduces overconfidence); $T < 1$ sharpens it. Temperature scaling is cheap and effective for calibrating overconfident MAP networks.

## Method Comparison

| Method | Cost | Calibration | OOD Detection | Implementation |
| --- | --- | --- | --- | --- |
| MC Dropout | $\sim 50\times$ inference | Moderate | Moderate | Easy — no retraining |
| Deep Ensembles | $M\times$ train + inference | Best | Best | Easy — train $M$ models |
| Laplace (last layer) | $1\times$ inference | Good | Good | Moderate — post-hoc |
| Variational BNN | $2\times$ params | Varies | Good | Hard — full retraining |
| SWAG | $\sim 30\times$ inference | Good | Good | Moderate — modified training |
| Temperature Scaling | $1\times$ | Good (calibration only) | Poor | Very easy — 1 parameter |

## Summary

Bayesian deep learning provides principled uncertainty quantification for neural networks:

- **Variational inference** minimizes the ELBO to approximate the posterior, using reparameterization for tractable gradients
- **MC Dropout** treats dropout masks as variational parameters — a cheap, widely applicable approximation
- **Laplace approximation** fits a Gaussian at the MAP solution — post-hoc, no retraining required
- **Deep ensembles** train multiple independent models — simple, state-of-the-art calibration, high cost
- **SWAG** exploits SGD noise to characterize the loss basin posterior

Calibration metrics (ECE, reliability diagrams) and temperature scaling provide the tools to evaluate and correct overconfidence. For production systems, deep ensembles are the practical gold standard; MC Dropout and Laplace are preferred when compute budget is tight.
