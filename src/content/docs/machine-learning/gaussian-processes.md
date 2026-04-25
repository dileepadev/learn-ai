---
title: Gaussian Processes
description: Learn how Gaussian Processes provide a principled Bayesian framework for regression and classification with calibrated uncertainty estimates — covering kernels, posterior inference, hyperparameter optimization, and practical applications in scientific modeling and Bayesian optimization.
---

A **Gaussian Process (GP)** is a probability distribution over functions. Rather than learning a single function from data (as a neural network does), a GP maintains a distribution over all functions consistent with observed data — producing predictions that come with principled, calibrated uncertainty estimates. This makes GPs uniquely powerful for scientific modeling, Bayesian optimization, and any application where knowing *how uncertain* a prediction is matters as much as the prediction itself.

Formally, a collection of random variables $\{f(x) : x \in \mathcal{X}\}$ is a Gaussian Process if every finite subset has a joint Gaussian distribution. A GP is completely specified by:

- A **mean function** $m(x) = \mathbb{E}[f(x)]$
- A **covariance (kernel) function** $k(x, x') = \mathrm{Cov}(f(x), f(x'))$

and written $f \sim \mathcal{GP}(m, k)$.

## The Prior Distribution

The GP prior encodes beliefs about the function before observing any data. With a zero mean prior $m(x) = 0$ and a squared exponential kernel:

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

the prior asserts that:

- Function values at nearby inputs are strongly correlated (controlled by the **lengthscale** $\ell$).
- The overall function amplitude scales with $\sigma_f^2$ (**signal variance**).
- The function is smooth and infinitely differentiable.

Samples from this prior are smooth continuous functions — the lengthscale $\ell$ controls how rapidly the function varies; small $\ell$ produces wiggly functions, large $\ell$ produces slowly varying ones.

## Gaussian Process Regression (GPR)

Given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ with noisy observations $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, the GP posterior has a closed-form solution.

Define the kernel matrix $K = [k(x_i, x_j)]_{i,j=1}^n$ and the test kernel vector $\mathbf{k}_* = [k(x_i, x_*)]_{i=1}^n$. The posterior at a test point $x_*$ is Gaussian with:

$$\mu_* = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

$$\sigma_*^2 = k(x_*, x_*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

The posterior mean $\mu_*$ is the prediction; $\sigma_*^2$ is the **epistemic uncertainty** — high in regions far from training data, low near observed data. This calibrated uncertainty is the key feature that distinguishes GPs from point-estimate models.

### Implementation with GPyTorch

```python
import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()  # Squared exponential kernel
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Training: optimize marginal log likelihood
model.train(); likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# Prediction with uncertainty
model.eval(); likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()  # 95% credible interval
```

## Kernel Functions

The choice of kernel encodes structural assumptions about the function being modeled:

### Squared Exponential (RBF)

$$k_{SE}(x, x') = \sigma_f^2 \exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)$$

Produces infinitely differentiable functions. Suitable for smooth, slowly varying phenomena. Often too smooth for real physical systems.

### Matérn Kernels

$$k_{\nu}(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\|x-x'\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|x-x'\|}{\ell}\right)$$

The smoothness parameter $\nu$ controls differentiability: $\nu = 1/2$ gives the Ornstein-Uhlenbeck process (once differentiable), $\nu = 3/2$ gives twice-differentiable functions, $\nu = 5/2$ is common in practice. Matérn-5/2 is often preferred over RBF for physical phenomena.

### Periodic Kernel

$$k_{per}(x, x') = \exp\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)$$

Encodes periodic structure with period $p$. Can be combined with other kernels to model periodic phenomena with a long-term trend.

### Composite Kernels

Kernels compose naturally:

- **Sum**: $k_1 + k_2$ — the function is a sum of components with different characteristics (e.g., smooth long-term trend + noisy local variation).
- **Product**: $k_1 \cdot k_2$ — encodes interactions (e.g., periodic function with amplitude that varies smoothly).
- **Deep kernels**: A neural network learns a feature map $\phi(x)$; the kernel is applied in feature space $k(\phi(x), \phi(x'))$ — combining GP uncertainty with deep representation learning.

## Hyperparameter Optimization

Kernel hyperparameters ($\ell$, $\sigma_f^2$, $\sigma_n^2$) are learned by maximizing the **log marginal likelihood**:

$$\log p(\mathbf{y} | X, \theta) = -\frac{1}{2}\mathbf{y}^\top (K_\theta + \sigma_n^2 I)^{-1} \mathbf{y} - \frac{1}{2}\log|K_\theta + \sigma_n^2 I| - \frac{n}{2}\log 2\pi$$

This objective balances data fit (first term) against model complexity (second term — the log determinant penalizes over-fitting by penalizing large kernel matrices). Optimization via gradient descent with respect to $\theta$ is tractable and produces well-calibrated models without a separate validation set.

## Computational Complexity and Scalability

Exact GP inference requires solving the linear system $(K + \sigma_n^2 I)^{-1}\mathbf{y}$, which costs $O(n^3)$ time and $O(n^2)$ memory — prohibitive for large datasets ($n > 10{,}000$).

### Sparse GP Approximations

**Sparse GPs** introduce $m \ll n$ **inducing points** $Z = \{z_j\}_{j=1}^m$ that summarize the information in the full dataset:

- **FITC** (Fully Independent Training Conditional): Approximates the exact posterior using only the inducing points, reducing cost to $O(nm^2)$.
- **SVGP** (Stochastic Variational GP): Uses variational inference with minibatch training — enabling GP regression on millions of data points with $O(m^3)$ cost per iteration.

```python
# Scalable GP with inducing points via SVGP
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

# Can train on 1M+ data points with mini-batches
```

## Bayesian Optimization

**Bayesian optimization** is the canonical application of GPs for optimizing expensive black-box functions — experiments, simulations, or neural network hyperparameter tuning where each evaluation costs significant time or money:

1. Fit a GP surrogate model to all evaluated points.
2. Use an **acquisition function** to select the next evaluation point — balancing exploration (high uncertainty) and exploitation (high predicted value).
3. Evaluate the true function at the selected point.
4. Update the GP and repeat.

### Acquisition Functions

**Expected Improvement (EI)**:

$$EI(x) = \mathbb{E}[\max(f(x) - f^*, 0)] = (\mu(x) - f^*)\Phi(Z) + \sigma(x)\phi(Z)$$

where $Z = (\mu(x) - f^*)/\sigma(x)$, $f^*$ is the current best observed value, and $\Phi$, $\phi$ are the Gaussian CDF and PDF.

**Upper Confidence Bound (UCB)**:

$$UCB(x) = \mu(x) + \kappa \sigma(x)$$

where $\kappa$ controls the exploration-exploitation trade-off.

```python
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# Fit GP to observed data
gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Optimize acquisition function to find next candidate
EI = ExpectedImprovement(gp, best_f=train_Y.max())
candidate, acq_value = optimize_acqf(
    EI, bounds=bounds, q=1, num_restarts=10, raw_samples=512
)
```

## GP Classification

For classification, the Gaussian likelihood assumption is replaced with a Bernoulli likelihood — the GP prior is placed on a latent function $f(x)$, which is passed through a sigmoid to produce class probabilities $p(y=1|x) = \sigma(f(x))$. The posterior is no longer Gaussian, requiring approximate inference:

- **Laplace approximation**: Fits a Gaussian to the posterior mode.
- **Expectation propagation (EP)**: More accurate than Laplace for classification.
- **Variational inference**: Scalable to large datasets.

## GPs vs. Neural Networks

| Property | Gaussian Processes | Neural Networks |
| --- | --- | --- |
| Uncertainty quantification | Principled, calibrated | Requires special techniques (ensembles, MC Dropout) |
| Data efficiency | High (prior encodes structure) | Low (needs large datasets) |
| Scalability | $O(n^3)$ exact; approximate methods scale | $O(n)$ stochastic gradient |
| Interpretability | Kernel interpretable | Black box |
| Hyperparameter tuning | Marginal likelihood | Expensive validation |
| Function space | Characterized by kernel | Implicit in architecture |

GPs excel in low-data regimes and scientific applications where calibrated uncertainty is required. Neural networks dominate in high-data, high-dimensional regimes where scalability is paramount. Deep kernel learning and neural process methods combine elements of both.
