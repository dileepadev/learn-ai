---
title: Optimal Transport in Machine Learning
description: Understand optimal transport theory and the Wasserstein distance — a geometrically meaningful measure for comparing probability distributions — with applications to generative models, domain adaptation, fairness, and distribution matching in modern machine learning.
---

**Optimal transport (OT)** is a mathematical framework for comparing probability distributions by finding the most efficient "plan" for moving mass from one distribution to another. Originating with Gaspard Monge's 1781 problem of moving soil from excavations to embankments at minimum cost, optimal transport has become a fundamental tool in modern machine learning — providing a geometrically meaningful distance between distributions that respects the underlying metric structure of the data space.

Unlike statistical divergences such as KL divergence or total variation distance, the **Wasserstein distance** computed by optimal transport is sensitive to the geometry of where distributions place their mass — making it far more informative for tasks like comparing images, training generative models, and measuring distributional shift.

## The Optimal Transport Problem

Given two probability distributions $\mu$ on $\mathcal{X}$ and $\nu$ on $\mathcal{Y}$, and a cost function $c(x, y)$ measuring the cost of transporting a unit of mass from $x$ to $y$, the **Monge problem** seeks a transport map $T : \mathcal{X} \to \mathcal{Y}$ that:

$$\min_T \int c(x, T(x)) \, d\mu(x) \quad \text{s.t.} \quad T_\# \mu = \nu$$

where $T_\#\mu$ denotes the pushforward measure (the distribution of $T(x)$ when $x \sim \mu$). The map $T$ rearranges the mass of $\mu$ into the configuration of $\nu$ at minimum total cost.

The **Kantorovich relaxation** allows mass splitting — instead of a map, it seeks a **transport plan** (joint distribution) $\gamma \in \Gamma(\mu, \nu)$:

$$\min_{\gamma \in \Gamma(\mu, \nu)} \int c(x, y) \, d\gamma(x, y)$$

where $\Gamma(\mu, \nu) = \{\gamma \geq 0 : \int d\gamma(x,\cdot) = d\mu(x),\, \int d\gamma(\cdot, y) = d\nu(y)\}$ is the set of couplings with marginals $\mu$ and $\nu$.

The Kantorovich problem is a linear program — it is always feasible and its solution is guaranteed to exist.

## The Wasserstein Distance

When the cost is $c(x,y) = d(x,y)^p$ for a metric $d$, the optimal transport cost defines the **Wasserstein-p distance**:

$$W_p(\mu, \nu) = \left(\min_{\gamma \in \Gamma(\mu, \nu)} \int d(x,y)^p \, d\gamma(x,y)\right)^{1/p}$$

### Wasserstein-1 (Earth Mover's Distance)

The **Wasserstein-1** distance (also called the **Earth Mover's Distance**, EMD) has a beautiful dual formulation via the Kantorovich-Rubinstein duality:

$$W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim \mu}[f(x)] - \mathbb{E}_{y \sim \nu}[f(y)]$$

where the supremum is over all 1-Lipschitz functions. This dual formulation is the key insight enabling **Wasserstein GANs**.

### Geometric Sensitivity

The critical advantage of the Wasserstein distance over KL divergence becomes apparent on simple examples:

Consider two 1D distributions: $\mu = \delta_0$ (point mass at 0) and $\nu_t = \delta_t$ (point mass at $t$):

- $W_1(\mu, \nu_t) = t$ — the distance grows smoothly and linearly with the gap between the distributions.
- $KL(\mu \| \nu_t) = \infty$ for any $t \neq 0$ — KL divergence is infinite whenever the supports don't overlap.

For training generative models where the model distribution and data distribution may have non-overlapping support in early training, KL divergence provides no gradient signal, while Wasserstein distance provides a useful, smooth gradient.

## The Sinkhorn Algorithm

For discrete distributions (empirical samples), the optimal transport problem is a linear program — solvable but expensive at $O(n^3)$ cost for $n$ samples. **Entropy-regularized optimal transport** adds a regularization term:

$$OT_\varepsilon(\mu, \nu) = \min_{\gamma \in \Gamma(\mu, \nu)} \langle C, \gamma \rangle - \varepsilon H(\gamma)$$

where $C$ is the pairwise cost matrix, $H(\gamma) = -\sum_{ij} \gamma_{ij} \log \gamma_{ij}$ is the entropy of the transport plan, and $\varepsilon > 0$ controls regularization strength.

The **Sinkhorn algorithm** solves this regularized problem in $O(n^2)$ time via alternating row and column normalization:

```python
import torch

def sinkhorn(cost_matrix, epsilon=0.1, num_iters=100):
    """
    Sinkhorn algorithm for entropy-regularized optimal transport.
    
    Args:
        cost_matrix: [n, m] pairwise costs between source and target samples
        epsilon: regularization strength (smaller = closer to true OT)
        num_iters: number of Sinkhorn iterations
    
    Returns:
        transport_plan: [n, m] optimal transport plan
        wasserstein_dist: scalar OT cost
    """
    n, m = cost_matrix.shape
    # Uniform marginals
    mu = torch.ones(n) / n
    nu = torch.ones(m) / m
    
    # Gibbs kernel
    K = torch.exp(-cost_matrix / epsilon)
    
    # Sinkhorn iterations: alternating normalization
    u = torch.ones(n)
    v = torch.ones(m)
    for _ in range(num_iters):
        u = mu / (K @ v)
        v = nu / (K.T @ u)
    
    # Transport plan
    transport_plan = torch.diag(u) @ K @ torch.diag(v)
    wasserstein_dist = (transport_plan * cost_matrix).sum()
    return transport_plan, wasserstein_dist

# Example: Compare two sets of image embeddings
source_embeddings = torch.randn(100, 512)  # 100 source samples
target_embeddings = torch.randn(80, 512)   # 80 target samples

# L2 pairwise cost matrix
cost = torch.cdist(source_embeddings, target_embeddings, p=2)
plan, dist = sinkhorn(cost, epsilon=0.05)
print(f"Wasserstein distance: {dist:.4f}")
```

The Sinkhorn algorithm has $O(n^2/\varepsilon^2)$ complexity — practically fast enough for mini-batch training in deep learning.

## Applications in Machine Learning

### Wasserstein GANs (WGAN)

The original GAN training objective uses Jensen-Shannon divergence, which provides no gradient when the generator and real distributions have non-overlapping support — causing mode collapse and unstable training.

**WGAN** (Arjovsky et al., 2017) replaces JS divergence with the Wasserstein-1 distance, exploiting the Kantorovich-Rubinstein dual:

$$\min_G \max_{\|D\|_L \leq 1} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

The discriminator $D$ (called the **critic**) is constrained to be 1-Lipschitz (originally via weight clipping; improved in WGAN-GP via gradient penalty), and its output is not bounded to $[0,1]$ — it estimates the Wasserstein distance rather than a class probability.

WGAN-GP achieves dramatically more stable training than vanilla GAN — eliminating mode collapse on standard benchmarks and providing a meaningful loss curve that correlates with sample quality.

### Domain Adaptation with OT

**Optimal transport domain adaptation** aligns feature distributions between source and target domains by finding a transport plan and applying it to the source features:

- Compute the OT plan $\gamma^*$ between source and target feature distributions.
- Map source features to their target-domain equivalents using the transport plan (barycentric projection).
- Train a classifier on the transported source features — which now resemble target distribution features.

**DeepJDOT** (Deep Joint Distribution Optimal Transport) jointly learns feature representations and transport plans in an end-to-end trainable model, minimizing a combined OT and classification loss.

### Distribution Matching in Fine-Tuning

Optimal transport provides principled distribution matching for fine-tuning language models and reward models — aligning model output distributions to target distributions in a geometrically meaningful way.

**Direct Preference Optimization variants** (including IPO, which uses squared differences) can be interpreted as minimizing Wasserstein-type distances between policy and reference distributions in the action space.

### Fairness and Demographic Parity

**Optimal transport fairness** addresses the challenge of equalizing model predictions across demographic groups without destroying predictive accuracy:

- Compute the OT map from predictions for group A to the distribution of predictions for group B.
- Apply a post-hoc calibration using this map to equalize the prediction distributions.

This **Wasserstein barycenter** approach finds the fairest prediction distribution simultaneously close (in Wasserstein distance) to all group-conditional distributions — achieving demographic parity while minimizing the total distortion to each group's predictions.

### Sample Efficiency in Generative Models

**Mini-batch optimal transport** improves training of flow models and diffusion models by using OT coupling to pair source noise samples with target data samples — creating better-coupled mini-batches that reduce variance in the training gradient:

- **Stochastic Interpolants** with OT coupling learn more direct paths from noise to data.
- **OT-Flow** uses OT to construct flow trajectories that minimize transport cost — producing models that learn simpler, more direct generative mappings.

## Sliced Wasserstein Distance

Computing Wasserstein distances in high dimensions is expensive. The **Sliced Wasserstein Distance (SWD)** approximates the $W_2$ distance by averaging Wasserstein distances of one-dimensional projections:

$$SW_2^2(\mu, \nu) = \int_{\mathbb{S}^{d-1}} W_2^2(\theta_\# \mu, \theta_\# \nu) \, d\theta$$

One-dimensional Wasserstein distances have a closed form: sort both distributions and compute the L2 distance between sorted values. SWD is approximated by averaging over random projection directions $\theta$:

```python
def sliced_wasserstein_distance(X, Y, num_projections=200):
    """Approximates W2 distance via random projections."""
    d = X.shape[1]
    projections = torch.randn(num_projections, d)
    projections = projections / projections.norm(dim=1, keepdim=True)
    
    X_proj = X @ projections.T  # [n, num_projections]
    Y_proj = Y @ projections.T  # [m, num_projections]
    
    X_sorted = X_proj.sort(dim=0).values
    Y_sorted = Y_proj.sort(dim=0).values
    
    # Upsample to same size if needed
    swd = ((X_sorted - Y_sorted) ** 2).mean()
    return swd.sqrt()
```

SWD scales as $O(n \log n)$ (for sorting) plus $O(ndk)$ for $k$ projections — tractable for large point clouds and high-dimensional embeddings.

## Gromov-Wasserstein Distance

The standard Wasserstein distance requires that source and target live in the same metric space. **Gromov-Wasserstein (GW) distance** extends OT to comparing distributions in different metric spaces — finding the transport plan that best preserves intra-distribution distances:

$$GW(\mu, \nu) = \min_\gamma \sum_{i,j,k,l} |d_X(x_i, x_j) - d_Y(y_k, y_l)|^2 \gamma_{ik} \gamma_{jl}$$

GW distance enables comparing graphs, protein structures, and point clouds defined in incompatible coordinate systems — with applications in graph matching, shape analysis, and cross-modal embedding alignment.

Optimal transport is increasingly woven into the fabric of modern machine learning — its geometrically principled approach to distribution comparison provides foundations that purely statistical divergences cannot match.
