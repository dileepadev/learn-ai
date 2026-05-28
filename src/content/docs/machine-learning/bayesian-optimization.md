---
title: Bayesian Optimization
description: Master Bayesian optimization for expensive black-box function optimization — covering Gaussian process surrogates, acquisition functions (Expected Improvement, UCB, Probability of Improvement, Thompson sampling), multi-fidelity and constrained BO, and practical implementation for hyperparameter tuning and neural architecture search.
---

Hyperparameter tuning often requires evaluating a function — train a model, measure validation loss — that takes hours to compute and provides no gradient information. Grid search and random search are wasteful: they ignore what previous evaluations revealed about promising regions. **Bayesian Optimization** (BO) builds a probabilistic model of the objective function and uses it to decide where to evaluate next, efficiently balancing exploration of unknown regions with exploitation of known promising ones.

## The Problem Setting

We want to minimize an expensive black-box function $f: \mathcal{X} \to \mathbb{R}$:

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$$

"Expensive" means each evaluation of $f$ costs significant time or money (training a deep learning model, running a drug synthesis experiment, simulating a physical system). We have a budget of $T$ evaluations.

BO maintains a **surrogate model** $p(f \mid \mathcal{D}_t)$ — a posterior distribution over functions given observations $\mathcal{D}_t = \{(\mathbf{x}_i, y_i)\}_{i=1}^t$ — and an **acquisition function** $\alpha(\mathbf{x})$ that scores candidate points, trading off exploration (high uncertainty) and exploitation (promising mean).

## Gaussian Process Surrogates

A **Gaussian Process** (GP) is the canonical surrogate for BO. It places a distribution over functions such that any finite set of function values follows a multivariate Gaussian:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}'))$$

where $m(\mathbf{x})$ is the mean function (usually zero) and $k(\mathbf{x}, \mathbf{x}')$ is the kernel function encoding smoothness assumptions. The most common kernel is the Matérn-5/2:

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2\!\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right), \quad r = \|\mathbf{x} - \mathbf{x}'\|_2$$

Given observations $\mathcal{D}_t$, the GP posterior at a new point $\mathbf{x}^*$ is Gaussian with:

$$\mu(\mathbf{x}^*) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\sigma^2(\mathbf{x}^*) = k(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

where $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ and $\mathbf{k}_{*,i} = k(\mathbf{x}^*, \mathbf{x}_i)$.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer:
    def __init__(self, bounds: list, noise: float = 1e-6):
        """
        bounds: list of (min, max) tuples for each dimension
        """
        self.bounds = np.array(bounds)
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=noise,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self.X_obs = []
        self.y_obs = []

    def update(self, x, y):
        """Add a new observation."""
        self.X_obs.append(x)
        self.y_obs.append(y)
        self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))

    def predict(self, X):
        return self.gp.predict(X, return_std=True)
```

## Acquisition Functions

### Expected Improvement (EI)

EI is the most widely used acquisition function. It measures the expected improvement over the current best observation $y^+ = \min_{i} y_i$:

$$\alpha_{\mathrm{EI}}(\mathbf{x}) = \mathbb{E}\!\left[\max(y^+ - f(\mathbf{x}),\, 0)\right]$$

For a GP surrogate, EI has a closed form:

$$\alpha_{\mathrm{EI}}(\mathbf{x}) = (y^+ - \mu(\mathbf{x}) - \xi)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z)$$

where $Z = \frac{y^+ - \mu(\mathbf{x}) - \xi}{\sigma(\mathbf{x})}$, $\Phi$ is the standard normal CDF, $\phi$ is the standard normal PDF, and $\xi \geq 0$ is an exploration parameter.

```python
from scipy.stats import norm


def expected_improvement(X_candidates, gp, y_best, xi: float = 0.01):
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = sigma.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    Z = (y_best - mu - xi) / (sigma + 1e-9)
    ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei.flatten()
```

### Upper Confidence Bound (UCB)

UCB selects the point with the highest optimistic upper bound on the function value:

$$\alpha_{\mathrm{UCB}}(\mathbf{x}) = -\mu(\mathbf{x}) + \kappa\,\sigma(\mathbf{x})$$

The exploration parameter $\kappa$ directly controls the tradeoff: large $\kappa$ encourages exploration, small $\kappa$ favors exploitation. GP-UCB has theoretical convergence guarantees — with appropriate $\kappa$ annealing, it achieves sublinear cumulative regret.

### Thompson Sampling

Thompson Sampling draws a sample function $\tilde{f} \sim p(f \mid \mathcal{D}_t)$ from the GP posterior and selects the maximizer of the sample:

$$\mathbf{x}_{t+1} = \arg\min_{\mathbf{x}} \tilde{f}(\mathbf{x})$$

Sampling from a GP can be done efficiently using the Cholesky decomposition: $\tilde{\mathbf{f}} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$, where $\mathbf{L}$ is the Cholesky factor of the covariance matrix. Thompson Sampling naturally parallelizes — sample $B$ independent functions simultaneously and select one point from each for a batch evaluation.

## Full Bayesian Optimization Loop

```python
import numpy as np
from scipy.optimize import minimize


def bayesian_optimization(
    objective_fn,      # expensive black-box function to minimize
    bounds,            # [(min, max), ...] for each dimension
    n_init: int = 5,
    n_iter: int = 50,
    xi: float = 0.01,
):
    optimizer = BayesianOptimizer(bounds)

    # Initial random exploration
    X_init = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_init, len(bounds)),
    )
    for x in X_init:
        y = objective_fn(x)
        optimizer.update(x, y)

    history = []
    for t in range(n_iter):
        y_best = min(optimizer.y_obs)

        # Maximize EI by multi-start optimization
        def neg_ei(x):
            x = x.reshape(1, -1)
            return -expected_improvement(x, optimizer.gp, y_best, xi=xi)

        best_x, best_ei = None, float("inf")
        for _ in range(10):  # 10 random restarts
            x0 = np.random.uniform(
                [b[0] for b in bounds], [b[1] for b in bounds]
            )
            res = minimize(
                neg_ei, x0, method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 100},
            )
            if res.fun < best_ei:
                best_ei, best_x = res.fun, res.x

        y_new = objective_fn(best_x)
        optimizer.update(best_x, y_new)
        history.append({"iter": t, "x": best_x, "y": y_new, "best": min(optimizer.y_obs)})
        print(f"Iter {t+1}: y={y_new:.4f}, best={min(optimizer.y_obs):.4f}")

    best_idx = np.argmin(optimizer.y_obs)
    return optimizer.X_obs[best_idx], optimizer.y_obs[best_idx], history
```

## Practical: Hyperparameter Tuning with Optuna

Modern BO libraries integrate the GP, acquisition function, and optimization loop:

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# Optuna uses TPE (Tree-structured Parzen Estimator) — a BO variant
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Multi-Fidelity Bayesian Optimization

When evaluations vary in fidelity (e.g., training for 10 epochs vs. 100 epochs), **multi-fidelity BO** queries cheap approximations first and escalates promising configurations to full fidelity:

- **BOHB** (Falkner et al., 2018): combines Hyperband's successive halving schedule with TPE — the dominant approach for neural architecture search
- **MTBO** (multi-task BO): transfers knowledge across related tasks (similar datasets, similar architectures) by modeling correlations between task-specific GPs

## Constrained Bayesian Optimization

Many real-world objectives have constraints (e.g., maximize accuracy subject to latency ≤ 100ms). Constrained BO fits a separate GP for each constraint and uses a constrained EI:

$$\alpha_{\mathrm{cEI}}(\mathbf{x}) = \alpha_{\mathrm{EI}}(\mathbf{x}) \cdot \prod_j \Pr[g_j(\mathbf{x}) \leq 0]$$

where $g_j$ are constraint functions and the probability of feasibility $\Pr[g_j(\mathbf{x}) \leq 0]$ is computed analytically from the constraint GP posterior.

## Summary

Bayesian Optimization is the gold standard for optimizing expensive black-box functions with a limited evaluation budget:

- **Gaussian Process surrogates** provide calibrated uncertainty estimates over the objective, capturing both the mean prediction and confidence interval at unobserved points
- **Expected Improvement** is the canonical acquisition function — closed-form, principled, and effective in practice; UCB and Thompson Sampling offer useful alternatives
- The full BO loop alternates between fitting the GP, maximizing the acquisition function, evaluating the objective, and updating the model
- **Multi-fidelity BO** (BOHB) dramatically improves efficiency by using cheap low-fidelity evaluations to filter out poor regions before expensive full evaluations
- For practical hyperparameter tuning, Optuna's TPE and BoTorch's full GP-EI/TS are the dominant production choices
