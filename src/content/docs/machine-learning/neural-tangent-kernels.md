---
title: "Neural Tangent Kernels"
description: "A deep dive into Neural Tangent Kernels (NTK), a theoretical framework that describes the training dynamics of infinitely wide neural networks and bridges deep learning with classical kernel methods."
---

## Introduction

The **Neural Tangent Kernel (NTK)** is a mathematical framework introduced by Jacot, Gabriel, and Hongler (2018) that characterizes the training dynamics of neural networks in the infinite-width limit. It shows that as the width of a neural network grows to infinity, gradient descent training behaves like kernel regression with a specific kernel — the neural tangent kernel — which remains constant throughout training.

This finding was theoretically surprising: infinitely wide neural networks trained with gradient flow converge to a global minimum, and their predictions evolve as a linear function of the kernel matrix. The NTK provides a bridge between deep learning (notoriously difficult to analyze) and classical kernel methods (well-understood theoretically).

---

## Background: Neural Networks and Kernel Methods

### Classical Kernel Regression

In kernel regression, predictions for a test point $x^*$ are made as:

$$f(x^*) = K(x^*, X)(K(X, X) + \lambda I)^{-1} y$$

Where $K$ is a positive semi-definite kernel function, $X$ are training inputs, and $y$ are training labels. The kernel encodes similarity between points in feature space.

### Neural Networks as Function Approximators

A neural network $f(x; \theta)$ with parameters $\theta$ learns a function mapping inputs to outputs. During training, $\theta$ changes continuously. The question is: can we characterize the function learned by the network in terms of a fixed kernel?

---

## Deriving the Neural Tangent Kernel

### Gradient Flow

Consider a network trained with gradient flow (continuous-time gradient descent):

$$\dot{\theta}(t) = -\nabla_\theta \mathcal{L}(\theta(t))$$

For a squared loss $\mathcal{L} = \frac{1}{2}\|f(X;\theta) - y\|^2$, the evolution of predictions is:

$$\dot{f}(X;\theta) = \nabla_\theta f(X;\theta) \cdot \dot{\theta} = -\nabla_\theta f(X;\theta) \nabla_\theta f(X;\theta)^\top (f(X;\theta) - y)$$

### Definition of the NTK

The **empirical NTK** at parameters $\theta$ is the matrix:

$$\hat{K}(x, x'; \theta) = \nabla_\theta f(x;\theta)^\top \nabla_\theta f(x';\theta)$$

This is the inner product of the Jacobians of the network output with respect to its parameters, evaluated at two input points. It measures how much the network output at $x$ and $x'$ co-vary when parameters are perturbed.

### The Infinite-Width Limit

**Key theorem (Jacot et al., 2018)**: For a fully-connected network of width $n$ with weights initialized from $\mathcal{N}(0, \sigma^2/n)$, as $n \to \infty$:

1. The empirical NTK $\hat{K}(x, x'; \theta_0)$ converges in probability to a deterministic kernel $K^*$ at initialization.
2. During training, $\hat{K}(x, x'; \theta(t)) \approx K^*$ remains approximately constant for all $t$.

This means the training dynamics become linear:

$$\dot{f}(X;\theta) \approx -K^*(X, X)(f(X;\theta) - y)$$

Which has the closed-form solution:

$$f(X;\theta(t)) = (I - e^{-K^*(X,X)t}) y + e^{-K^*(X,X)t} f(X;\theta_0)$$

As $t \to \infty$, the network converges to the **kernel regression solution** with kernel $K^*$.

---

## Computing the NTK

### Recursive Formula

For a fully-connected network with $L$ layers, the NTK can be computed recursively. Let $\Sigma^{(0)}(x,x') = x^\top x'$ be the input kernel. Then for layer $\ell$:

$$\Sigma^{(\ell)}(x,x') = \mathbb{E}_{f \sim \mathcal{GP}(0, \Sigma^{(\ell-1)})} [\sigma(f(x)) \sigma(f(x'))]$$

$$\dot{\Sigma}^{(\ell)}(x,x') = \mathbb{E}_{f \sim \mathcal{GP}(0, \Sigma^{(\ell-1)})} [\sigma'(f(x)) \sigma'(f(x'))]$$

The NTK is then:

$$K^{(L)}(x,x') = \sum_{\ell=1}^{L} \dot{\Sigma}^{(\ell)}(x,x') \prod_{h=\ell+1}^{L} \dot{\Sigma}^{(h)}(x,x')$$

This recursive formula depends on the activation function $\sigma$ and the network depth. For ReLU activations, the expectations over Gaussian processes have analytic forms.

### Numerical Computation

For finite-width networks, the empirical NTK can be computed using automatic differentiation:

```python
import torch
import torch.nn as nn
from functorch import jacrev, vmap

def compute_ntk(model, x1, x2):
    """Compute the empirical NTK between two batches of inputs."""
    def net_fn(params, x):
        return torch.func.functional_call(model, params, x)

    params = dict(model.named_parameters())

    # Compute Jacobians
    jac1 = vmap(jacrev(net_fn, argnums=0), in_dims=(None, 0))(params, x1)
    jac2 = vmap(jacrev(net_fn, argnums=0), in_dims=(None, 0))(params, x2)

    # Flatten and compute inner products
    # jac shape: (batch, output_dim, *param_shape)
    ntk = 0
    for key in params:
        j1 = jac1[key].flatten(start_dim=2)  # (n1, out, params)
        j2 = jac2[key].flatten(start_dim=2)  # (n2, out, params)
        ntk += torch.einsum('iop,jop->ij', j1, j2)
    return ntk
```

---

## Properties of the NTK

### Positive Semi-Definiteness

The NTK is always positive semi-definite because it is defined as a sum of outer products of gradient vectors. This guarantees that gradient descent on the squared loss with a fixed NTK always converges to a global minimum.

### Depth Dependence

Deeper networks produce NTKs with different spectral properties. For very deep networks, the NTK can become **degenerate**: the smallest eigenvalue approaches zero, slowing convergence and causing the kernel to become rank-deficient. This connects to the "depth vs. width" trade-off in practice.

### Activation Function Sensitivity

The NTK depends strongly on the choice of activation function. For ReLU:

$$\Sigma^{(\ell)}_{\text{ReLU}}(x,x') = \frac{\|x\|\|x'\|}{2\pi}(\sin\theta + (\pi - \theta)\cos\theta)$$

Where $\theta$ is the angle between $x$ and $x'$. Smooth activations (erf, GELU) produce different and often better-conditioned NTKs.

### Translation Invariance in CNNs

For convolutional networks, the NTK inherits the translation equivariance of the architecture, making it a translation-equivariant kernel — conceptually similar to a stationary kernel but adapted to the symmetry structure of the network.

---

## NTK Regime vs. Feature Learning Regime

One of the most important insights from NTK theory is the distinction between two learning regimes:

### Kernel (Lazy) Regime

When networks are very wide and step sizes are small, the NTK stays approximately constant — this is the "lazy training" regime. The network behaves like a kernel machine and does not meaningfully update its internal representations (features). The NTK at initialization determines the solution.

### Feature Learning Regime

When networks are not infinitely wide (i.e., in practice), the NTK evolves during training. The network actively learns new features by changing its internal representations. This feature learning is often what gives practical deep networks their advantage over kernel methods on complex tasks.

The gap between finite and infinite width is the gap between feature learning and kernel regression. Understanding when and why feature learning helps over kernel regression is an active research frontier.

---

## Mean Field Theory and Criticality

Related to NTK theory, **mean field theory** studies the behavior of signals and gradients in very deep networks. A network is at the "edge of chaos" (critical point) when:

$$\chi = \sigma_w^2 \langle \phi'(x)^2 \rangle_* = 1$$

Where $\sigma_w^2$ is the weight variance, $\phi'$ is the activation derivative, and $\langle \cdot \rangle_*$ denotes the expectation over the fixed-point distribution. At criticality, gradients neither explode nor vanish, enabling effective training of very deep networks.

The NTK at criticality has the slowest eigenvalue decay, enabling the network to learn the richest class of functions.

---

## Applications of NTK Theory

### Predicting Generalization

The NTK provides a theoretical basis for understanding generalization. The generalization error of an infinitely wide network on a test point $x^*$ is approximately:

$$\text{Err}(x^*) \approx K^*(x^*, X)(K^*(X,X) + \lambda I)^{-1} \text{Err}_{\text{train}}$$

This shows that generalization depends on the smoothness of the learned function with respect to the NTK metric.

### Architecture Search via Kernel Analysis

NTK eigenvalue spectra can be used to compare architectures without training. Architectures with better-conditioned NTK matrices (larger minimum eigenvalue) are predicted to train faster and generalize better.

The **Condition Number** $\kappa = \lambda_{\max} / \lambda_{\min}$ of the NTK matrix predicts training speed — a lower condition number means more uniform learning across all data directions.

### Understanding BatchNorm and Skip Connections

NTK analysis reveals why batch normalization and residual connections help training:

- BatchNorm normalizes the NTK spectrum, reducing condition number.
- Residual connections add an identity contribution to each layer's NTK term, preventing collapse of the smallest eigenvalues.

---

## Finite-Width Corrections

The infinite-width NTK is an approximation. For finite width $n$, the corrections scale as $O(1/n)$. These finite-width corrections include:

- **Feature learning**: The kernel changes as training progresses.
- **Fluctuations**: Stochastic noise from random initialization has $O(1/\sqrt{n})$ effects on predictions.
- **Kernel corrections**: The empirical NTK deviates from the limiting kernel by $O(1/\sqrt{n})$.

The **$\mu$P (maximal update parametrization)** by Yang and Hu (2021) extends NTK theory to allow stable feature learning as width scales, providing practical initialization and learning rate schemes for large networks.

---

## Limitations of NTK Theory

### Practical Networks Are Not Infinitely Wide

Real networks (hundreds to thousands of hidden units) are far from the infinite-width limit. Empirically, the NTK evolves substantially during training, especially in early layers.

### NTK Does Not Explain Overparameterized Generalization

Why do overparameterized networks generalize well despite memorizing training data? NTK theory provides the minimum-norm interpolating solution, which has good generalization in theory — but the gap between NTK predictions and empirical generalization can be significant.

### Feature Learning Advantage

NTK/lazy training networks are often outperformed by finite networks that do genuine feature learning, especially on tasks requiring hierarchical feature composition (e.g., vision, language). Transfer learning, which relies on learned features, cannot be explained by NTK theory.

---

## Empirical Validation

Experiments comparing infinite-width NTK kernel regression with finite-width neural networks on standard benchmarks:

| Method | MNIST Accuracy | CIFAR-10 Accuracy |
|--------|---------------|-------------------|
| Finite FC (trained) | 99.2% | 56.8% |
| Infinite FC NTK | 99.0% | 52.7% |
| Finite CNN (trained) | 99.6% | 89.5% |
| Infinite CNN NTK | 99.4% | 77.3% |

The performance gap is larger for CIFAR-10, reflecting the importance of feature learning for complex vision tasks.

---

## Summary

The Neural Tangent Kernel is one of the most important theoretical developments in modern deep learning, providing a precise mathematical characterization of how infinitely wide neural networks learn. By connecting neural networks to kernel methods, NTK theory enables rigorous analysis of training dynamics, convergence, and generalization. At the same time, the gap between the NTK regime and practical finite-width networks highlights that feature learning — not captured by the frozen kernel — is central to the empirical success of deep learning. NTK theory continues to inspire practical advances in initialization strategies, architecture design, and understanding of the benefit of scale.
