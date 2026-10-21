---
title: Gaussian Mixture Models - Soft Clustering with Probabilities
description: Learn how Gaussian Mixture Models represent data as a mixture of Gaussians fit with expectation-maximization, and how they compare with k-means.
---

A Gaussian Mixture Model (GMM) assumes data is generated from a mixture of several Gaussian distributions with unknown parameters. It is used for clustering, density estimation, and as a building block in speech and anomaly detection.

## The Model

```text
p(x) = sum over k of  pi_k * N(x | mu_k, Sigma_k)
```

- `pi_k`: mixing weight of component k (weights sum to 1)
- `mu_k`: component mean
- `Sigma_k`: component covariance

## Soft Assignments

Unlike k-means, which gives each point a single cluster, a GMM computes a **responsibility**: the posterior probability that component k produced point x. This represents uncertainty for points between clusters.

## Fitting with Expectation-Maximization

EM alternates two steps until the log-likelihood converges:

1. **E-step**: compute responsibilities using the current parameters
2. **M-step**: update means, covariances, and weights as responsibility-weighted statistics

EM increases the likelihood every iteration but converges only to a local optimum, so run several initializations (k-means initialization is common).

## Usage

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=0)
gmm.fit(X)

labels = gmm.predict(X)
probs = gmm.predict_proba(X)
log_density = gmm.score_samples(X)   # low values suggest outliers
```

## Covariance Types

| Type | Meaning |
|------|---------|
| `spherical` | One variance per component (like k-means) |
| `diag` | Axis-aligned ellipses |
| `tied` | All components share a covariance |
| `full` | Arbitrary ellipses, most flexible |

## Choosing the Number of Components

Fit models for several K and compare **BIC** or **AIC**, which penalize complexity; pick the minimum.

## Caveats

- Sensitive to initialization and can collapse a component onto a single point (singular covariance); a regularization term helps
- Assumes elliptical clusters; strongly non-Gaussian shapes fit poorly
- Cost grows with dimension for full covariances
