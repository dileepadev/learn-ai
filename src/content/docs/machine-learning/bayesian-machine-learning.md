---
title: Bayesian Machine Learning
description: An in-depth exploration of Bayesian machine learning, covering probabilistic inference, Gaussian processes, variational methods, and probabilistic programming frameworks for building models that quantify uncertainty.
---

Bayesian machine learning treats model parameters as random variables with probability distributions, rather than fixed unknown values to be estimated. This probabilistic perspective enables models to express uncertainty in their predictions — a critical property for safety-sensitive applications such as medical diagnosis, autonomous systems, and scientific discovery.

## The Bayesian Framework

Bayesian inference begins with Bayes' theorem:

$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \, P(\theta)}{P(\mathcal{D})}$$

where:

- $P(\theta)$ — the **prior** distribution over parameters, encoding beliefs before observing data
- $P(\mathcal{D} \mid \theta)$ — the **likelihood** of the data given parameters
- $P(\mathcal{D})$ — the **marginal likelihood** (evidence), a normalising constant
- $P(\theta \mid \mathcal{D})$ — the **posterior** distribution, updated beliefs after observing data

Predictions are made by integrating over the posterior rather than plugging in a single point estimate:

$$P(y^* \mid x^*, \mathcal{D}) = \int P(y^* \mid x^*, \theta) \, P(\theta \mid \mathcal{D}) \, d\theta$$

This **posterior predictive distribution** naturally captures epistemic uncertainty (uncertainty about model parameters) and aleatoric uncertainty (irreducible noise in the data).

## Conjugate Models and Exact Inference

When the prior and likelihood are conjugate, the posterior has the same functional form as the prior and can be computed analytically.

| Prior | Likelihood | Use case |
| --- | --- | --- |
| Beta | Binomial | Proportion estimation |
| Dirichlet | Multinomial | Topic modelling |
| Gaussian | Gaussian | Linear regression |
| Gamma | Poisson | Count data |

Conjugate models are tractable but restrictive. Real-world models generally require approximate inference.

## Gaussian Processes

A Gaussian Process (GP) is a distribution over functions. Any finite collection of function evaluations follows a multivariate Gaussian distribution:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

where $m(x)$ is the mean function and $k(x, x')$ is the covariance (kernel) function encoding assumptions about smoothness and periodicity.

GPs provide exact posterior inference over function values given observations, yielding **calibrated uncertainty estimates** alongside predictions.

```python
import gpytorch, torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
```

### Kernel Functions

The choice of kernel determines the inductive bias:

| Kernel | Properties |
| --- | --- |
| RBF (Squared Exponential) | Infinitely differentiable, very smooth |
| Matérn 3/2, 5/2 | Finite differentiability, more realistic roughness |
| Periodic | Captures periodic patterns |
| Linear | Equivalent to Bayesian linear regression |
| Composed kernels | Additive / multiplicative combinations for rich structure |

GPs scale as $\mathcal{O}(n^3)$ due to matrix inversion, motivating sparse approximations such as inducing point methods (SVGP) that reduce cost to $\mathcal{O}(nm^2)$ with $m \ll n$ inducing points.

## Variational Inference

Variational inference (VI) frames posterior computation as an optimisation problem. It introduces a family of tractable distributions $q_\phi(\theta)$ and minimises the KL divergence from the posterior:

$$\phi^* = \arg\min_\phi \, \text{KL}(q_\phi(\theta) \| P(\theta \mid \mathcal{D}))$$

This is equivalent to maximising the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log P(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| P(\theta))$$

The first term rewards fitting the data; the second acts as regularisation, keeping the approximate posterior close to the prior.

### Mean-Field VI

Mean-field VI assumes full factorisation: $q_\phi(\theta) = \prod_i q_i(\theta_i)$. This is efficient but underestimates posterior correlations between parameters.

### Normalising Flow VI

Normalising flows transform a simple base distribution through a sequence of invertible mappings, producing a rich, non-Gaussian approximate posterior without the factorisation assumption.

## Markov Chain Monte Carlo

MCMC methods draw samples from the posterior without specifying a parametric form for $q$.

- **Metropolis-Hastings** — proposes moves and accepts/rejects based on the likelihood ratio
- **Hamiltonian Monte Carlo (HMC)** — uses gradient information to propose efficient moves along high-probability regions
- **No-U-Turn Sampler (NUTS)** — adaptive HMC that automatically tunes trajectory length, used in Stan and PyMC

MCMC is asymptotically exact but computationally expensive for large datasets and high-dimensional parameter spaces.

## Bayesian Neural Networks

Bayesian Neural Networks (BNNs) place distributions over network weights rather than point estimates. Full posterior inference is intractable, so practical BNNs use:

- **Variational BNNs** — optimise ELBO with weight distributions parameterised by mean and variance
- **Monte Carlo Dropout** — interpret dropout at test time as approximate Bayesian inference
- **Deep Ensembles** — train multiple independent networks and use disagreement as an uncertainty proxy
- **Laplace Approximation** — fit a Gaussian to the loss landscape curvature at the MAP estimate

| Method | Calibration | Cost | Scalability |
| --- | --- | --- | --- |
| Full VI | Good | High | Moderate |
| MC Dropout | Moderate | Low | High |
| Deep Ensembles | Excellent | High | Moderate |
| Laplace | Good | Low (post-hoc) | High |

## Probabilistic Programming

Probabilistic programming languages (PPLs) let analysts express models as code and automate inference.

```python
import pymc as pm
import numpy as np

with pm.Model() as linear_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha + beta * X
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, target_accept=0.95)
```

Popular frameworks:

| Framework | Language | Backend | Strengths |
| --- | --- | --- | --- |
| PyMC | Python | JAX / Aesara | Research, accessibility |
| Stan | Stan DSL | C++ | High performance, clinical |
| NumPyro | Python | JAX | GPU acceleration, speed |
| Pyro | Python | PyTorch | Deep generative models |
| TensorFlow Probability | Python | TF/JAX | Production integration |

## When to Apply Bayesian Methods

- **Small datasets** — Bayesian regularisation through priors replaces cross-validated regularisation tuning
- **Active learning** — uncertainty estimates guide selection of the most informative data points to label
- **Multi-task learning** — hierarchical Bayesian models share statistical strength across related tasks
- **Decision-making under uncertainty** — posterior predictive distributions enable risk-aware decisions
- **Scientific modelling** — posterior enables hypothesis testing and interpretable parameter estimation

The main trade-off relative to frequentist methods is computational cost. For large-scale settings, scalable VI or ensemble methods often provide a practical balance between uncertainty quantification and efficiency.
