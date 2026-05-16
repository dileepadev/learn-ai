---
title: Influence Functions in Machine Learning
description: Learn how influence functions trace model predictions back to individual training examples — covering the theoretical foundations from robust statistics, efficient approximations for deep learning, applications to data valuation, debugging, and connections to memorization.
---

Influence functions are a classical statistical tool adapted for deep learning by Koh & Liang (2017). They answer a fundamental question: **how much does removing (or upweighting) a single training example change a model's predictions?** This makes influence functions a powerful tool for data valuation, training set debugging, understanding memorization, and identifying mislabeled examples.

## The Core Question

Given a model trained on dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$, suppose we remove training example $(x_j, y_j)$. How do the model's parameters and predictions change?

Retraining from scratch for every candidate removal is computationally prohibitive. Influence functions provide an efficient **first-order approximation** using only the curvature of the loss at the trained model.

## Theoretical Foundation

### From Robust Statistics

Influence functions originate in robust statistics (Hampel, 1974). Given an estimator $\hat{\theta}$ as a functional of the empirical distribution $F_n$, the influence function measures the effect of infinitesimally upweighting a single point:

$$\text{IF}(z; \hat{\theta}, F) = \lim_{\epsilon \to 0} \frac{\hat{\theta}(F + \epsilon(\delta_z - F)) - \hat{\theta}(F)}{\epsilon}$$

Applied to ERM (empirical risk minimization):

$$\hat{\theta} = \arg\min_\theta \frac{1}{n} \sum_i \mathcal{L}(z_i; \theta)$$

The influence of upweighting $z_j$ by $\epsilon$ on the optimal parameters is:

$$\frac{d\hat{\theta}_\epsilon}{d\epsilon}\bigg|_{\epsilon=0} = -H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_j; \hat{\theta})$$

where $H_{\hat{\theta}} = \frac{1}{n} \sum_i \nabla_\theta^2 \mathcal{L}(z_i; \hat{\theta})$ is the **Hessian** of the training loss at the optimum.

### Removing a Training Point

Removing $z_j$ is equivalent to upweighting it by $-\frac{1}{n}$, so the parameter change approximation is:

$$\hat{\theta}_{-j} - \hat{\theta} \approx \frac{1}{n} H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_j; \hat{\theta})$$

### Effect on a Test Prediction

The influence of removing training point $z_j$ on the loss of a test point $z_{\text{test}}$ is:

$$\mathcal{I}(z_j, z_{\text{test}}) = -\nabla_\theta \mathcal{L}(z_{\text{test}}; \hat{\theta})^T H_{\hat{\theta}}^{-1} \nabla_\theta \mathcal{L}(z_j; \hat{\theta})$$

A large positive value means removing $z_j$ increases the test loss — $z_j$ was helpful. A large negative value means removing $z_j$ decreases the test loss — $z_j$ was harmful.

## The Hessian Inverse Problem

The dominant computational challenge is the **Hessian inverse** $H_{\hat{\theta}}^{-1}$. For a model with $p$ parameters, $H \in \mathbb{R}^{p \times p}$ is:

- Too large to compute and store explicitly ($p$ can be billions)
- May not be positive definite (due to non-convexity)
- Expensive to invert even when computable

### Conjugate Gradient (CG)

Rather than computing $H^{-1}$ explicitly, we solve the linear system $H v = b$ using conjugate gradient. This avoids materializing $H$ and requires only **Hessian-vector products** (HVPs), which are computable in $O(p)$ time via double backpropagation.

```python
# Conceptual HVP computation (illustrative)
import torch

def hessian_vector_product(loss, params, v):
    """Compute H @ v without materializing H."""
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grad])
    hvp = torch.autograd.grad(
        (grad_flat * v.detach()).sum(),
        params,
    )
    return torch.cat([h.flatten() for h in hvp])
```

### Stochastic Estimation: LiSSA

**LiSSA (Linear time Stochastic Second-order Algorithm)** (Agarwal et al., 2017) approximates $H^{-1} v$ using the Neumann series:

$$H^{-1} = \frac{1}{\lambda} \sum_{j=0}^\infty \left(I - \frac{H}{\lambda}\right)^j$$

truncated after $J$ terms and estimated with mini-batches. This gives an unbiased estimator of $H^{-1} v$ computable without explicit Hessian storage.

### EK-FAC Approximation

For large models, **Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EK-FAC)** decomposes the Fisher information matrix (a Hessian proxy) as layer-wise Kronecker products, enabling tractable inversion even for models with millions of parameters.

## Applications

### Identifying Influential Training Examples

Given a test prediction, sort training examples by $|\mathcal{I}(z_j, z_{\text{test}})|$. The highest-influence examples are the most responsible for the prediction — useful for:

- **Debugging misclassifications**: why did the model predict class A for this test image?
- **Explaining model behavior**: which training examples support this decision?
- **Detecting poisoned data**: adversarially injected examples have outsized influence on specific test predictions

### Detecting Mislabeled Data

Training examples that are **harmful** across many test points (consistently negative influence) are candidates for mislabeling or data corruption:

$$\text{harmfulness}(z_j) = \sum_{z_{\text{test}} \in \mathcal{D}_{\text{val}}} \mathcal{I}(z_j, z_{\text{test}})$$

Koh & Liang (2017) demonstrated that influence functions correctly identify mislabeled examples in SVM and neural network classifiers at rates far above random chance.

### Data Valuation

Influence functions provide a principled measure of each training example's **value** to model performance — a quantity of interest in:

- **Data markets**: pricing training data contributions
- **Federated learning**: compensating data owners fairly
- **Data pruning**: identifying low-value or redundant examples for removal

### Leave-One-Out Cross-Validation Approximation

Leave-one-out CV requires $n$ retraining runs. Influence functions approximate the LOO prediction at $O(1)$ extra cost per removed example (after computing $H^{-1}$):

$$\mathcal{L}(z_{\text{test}}; \hat{\theta}_{-j}) \approx \mathcal{L}(z_{\text{test}}; \hat{\theta}) + \mathcal{I}(z_j, z_{\text{test}})$$

This approximation is exact in the convex case and a useful heuristic for deep networks.

## Connections to Memorization

Memorization occurs when a model learns training-data-specific patterns not generalized from the underlying distribution. Influence functions quantify memorization:

- **Highly self-influential examples**: $\mathcal{I}(z_j, z_j)$ is large — removing $z_j$ significantly increases the model's loss on $z_j$ itself. This indicates the model has memorized $z_j$ rather than learning a generalizable pattern.
- **Counterfactual memorization** (Feldman & Zhang, 2020) extends this: an example $(x, y)$ is memorized if the model correctly predicts $y$ from $x$ only because $(x, y)$ appeared in training. This is closely related to large self-influence.

Carlini et al. (2021) showed that LLMs memorize verbatim training text for examples with high influence — providing a mechanistic connection between influence functions and extractable memorization.

## Limitations and Critiques

### Non-Convexity

Influence functions assume the model is at a **local minimum** with a positive-definite Hessian. Deep networks are non-convex; the Hessian is often indefinite. Empirically, influence function approximations are sometimes accurate but can be unreliable, particularly for overparameterized models.

**Bae et al. (2022)** showed that standard influence functions poorly approximate LOO retraining for deep networks due to non-convexity and approximate training, and proposed **if-COMP** corrections.

### Computational Cost at Scale

Even with CG and HVP tricks, computing influences for all training examples against all test examples is $O(n_{\text{train}} \times n_{\text{test}})$ HVP evaluations. For billion-parameter LLMs trained on trillions of tokens, influence computation remains expensive — motivating approximations like TRAK and datamodels.

### TRAK: Scalable Influence Attribution

**TRAK (TRAcing with the Kernel)** (Park et al., 2023) approximates influence functions for large models by:

1. Projecting gradients into a random low-dimensional subspace
1. Computing influence scores in the projected space

This reduces influence computation to a few gradient projections per training example, making it tractable for models like ViTs and GPT at scale.

### Datamodels

**Datamodels** (Ilyas et al., 2022) take a different approach: rather than computing the Hessian, they train **linear models** that predict the effect of data subsets on specific test predictions. This is empirically more reliable for deep networks but requires many model retraining runs (hundreds) to fit the datamodel — expensive but accurate.

## Practical Workflow

A typical influence-function debugging workflow:

1. **Train model** on $\mathcal{D}$
1. **Identify a test failure** — a misclassification or unexpected output
1. **Compute training gradients** $\nabla_\theta \mathcal{L}(z_j; \hat{\theta})$ for each candidate training example
1. **Solve** $H^{-1} \nabla_\theta \mathcal{L}(z_{\text{test}}; \hat{\theta})$ using CG or LiSSA
1. **Score** each training example via the dot product
1. **Inspect** top-influence examples for labeling errors, distribution shift, or data leakage
1. **Intervene** — remove, relabel, or reweight harmful examples and retrain

## Summary

Influence functions provide a theoretically grounded, first-order approximation to the effect of individual training examples on model predictions. The core formula:

$$\mathcal{I}(z_j, z_{\text{test}}) = -\nabla_\theta \mathcal{L}(z_{\text{test}})^T H^{-1} \nabla_\theta \mathcal{L}(z_j)$$

captures the inner product of the test gradient and the (inverse-Hessian-weighted) training gradient — a measure of how aligned the two examples are in the curvature-adjusted parameter space.

Despite limitations from non-convexity and computational cost, influence functions remain a valuable tool for data debugging, mislabel detection, memorization analysis, and data valuation. Scalable approximations (TRAK, EK-FAC, datamodels) are extending their practical applicability to large-scale modern models.
