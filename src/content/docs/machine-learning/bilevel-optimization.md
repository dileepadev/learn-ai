---
title: Bilevel Optimization
description: Understand bilevel optimization — a nested optimization framework where an outer problem depends on the solution of an inner problem. Covers gradient-based methods (MAML, DARTS, implicit differentiation), hyperparameter optimization, neural architecture search, meta-learning, adversarial training as bilevel problems, scalability challenges, and modern approximations including truncated backpropagation and implicit function theorem approaches.
---

**Bilevel optimization** is a hierarchical optimization framework in which an **outer (upper-level)** problem depends on the solution of an **inner (lower-level)** problem. The outer objective is optimized over a set of variables $\phi$, while for each $\phi$ the inner problem minimizes a separate objective over variables $\theta$. The coupling between levels — the outer problem's objective depends on how the inner problem responds to $\phi$ — makes bilevel optimization fundamentally more complex than standard single-level optimization.

Formally:

$$\min_\phi \; F(\phi, \theta^*(\phi)) \qquad \text{subject to} \quad \theta^*(\phi) = \arg\min_\theta \; G(\phi, \theta)$$

where $F$ is the outer objective, $G$ is the inner objective, $\phi$ are the outer (meta) variables, and $\theta$ are the inner variables.

## Applications in Machine Learning

Bilevel structure appears in numerous machine learning problems:

### Meta-Learning

In **model-agnostic meta-learning (MAML)**, the outer problem learns a good initialization $\phi$ for a model, while the inner problem adapts the model to a specific task with a few gradient steps:

- **Inner**: $\theta_\tau^* = \phi - \alpha \nabla_\theta G(\phi, \mathcal{D}^\text{train}_\tau)$ — task-specific adaptation on support set.
- **Outer**: $\min_\phi \sum_\tau F(\theta_\tau^*(\phi), \mathcal{D}^\text{val}_\tau)$ — generalization across tasks.

The outer gradient $\nabla_\phi F$ requires backpropagating through the inner update — a second-order computation central to bilevel optimization.

### Hyperparameter Optimization (HPO)

Given a model trained with hyperparameters $\phi$ (learning rate, regularization strength, architecture choices), the outer problem optimizes held-out validation loss as a function of $\phi$:

- **Inner**: $\theta^*(\phi) = \arg\min_\theta \mathcal{L}_\text{train}(\theta; \phi)$ — train the model to convergence.
- **Outer**: $\min_\phi \mathcal{L}_\text{val}(\theta^*(\phi))$ — optimize hyperparameters on validation set.

Gradient-based HPO methods (DARTS, DrMAD, gradient-based HO) differentiate through the training process to obtain $\nabla_\phi \mathcal{L}_\text{val}$.

### Neural Architecture Search (NAS)

**DARTS** (Differentiable Architecture Search, Liu et al., 2019) relaxes discrete architecture choices to continuous architecture parameters $\phi$ (mixing weights over candidate operations):

- **Inner**: optimize model weights $\theta$ on training data for fixed architecture $\phi$.
- **Outer**: optimize architecture parameters $\phi$ on validation data, with $\theta$ implicitly defined by the inner solution.

DARTS uses a first-order approximation (treating $\theta$ as approximately optimal) and a second-order approximation (using one step of inner unrolling) to estimate the outer gradient.

### Data Augmentation and Curriculum Learning

The bilevel framework naturally models learned data augmentation and curriculum learning:

- **Outer**: learn augmentation parameters (magnitude, policy weights) or example weights that maximize validation performance.
- **Inner**: train the model on augmented/reweighted training data.

**AutoAugment** and **RandAugment** frame augmentation policy search as a meta-optimization problem; **L2RW** (Learning to Reweight Examples) uses bilevel optimization to down-weight noisy or mislabeled training examples.

### Adversarial Training

Min-max adversarial training is a bilevel problem:

$$\min_\theta \; \mathbb{E}_{(x,y)} \left[ \max_{\delta: \|\delta\|_p \leq \epsilon} \mathcal{L}(\theta; x+\delta, y) \right]$$

- **Inner**: find worst-case adversarial perturbation $\delta^*(x; \theta)$ via PGD.
- **Outer**: train model parameters $\theta$ to minimize loss under worst-case perturbation.

The outer gradient $\nabla_\theta$ is taken with respect to the adversarially perturbed inputs, implicitly depending on the inner solution.

## Gradient Computation Methods

The central computational challenge is computing the **hypergradient** $\nabla_\phi F(\phi, \theta^*(\phi))$ efficiently.

### Unrolled Differentiation (Reverse-Mode)

Approximate $\theta^*(\phi)$ by running $T$ steps of gradient descent:

$$\theta_T(\phi) = \theta_0 - \alpha_1 \nabla_\theta G_1 - \alpha_2 \nabla_\theta G_2 - \ldots$$

Then backpropagate through the unrolled computation graph:

$$\nabla_\phi F \approx \frac{\partial F(\phi, \theta_T(\phi))}{\partial \phi}$$

This computes exact gradients for the $T$-step approximation but requires storing $O(T)$ intermediate activations, making it memory-intensive for large $T$. Used in **MAML** (T=5-10 steps), **DARTS** (1-2 steps), and **iMAML**.

### Implicit Function Theorem (IFT)

If $\theta^*(\phi)$ is a smooth function of $\phi$ (guaranteed by strong convexity of $G$ in $\theta$), the implicit function theorem gives:

$$\nabla_\phi F = \frac{\partial F}{\partial \phi} - \frac{\partial F}{\partial \theta} \left( \frac{\partial^2 G}{\partial \theta^2} \right)^{-1} \frac{\partial^2 G}{\partial \theta \partial \phi}$$

This requires the inverse Hessian $(\partial^2 G / \partial \theta^2)^{-1}$, which is $O(|\theta|^2)$ to store and $O(|\theta|^3)$ to invert directly. In practice, approximations are used:

- **Conjugate gradient (CG)**: solve $(\partial^2 G / \partial \theta^2) v = \partial F / \partial \theta$ iteratively — only requires Hessian-vector products, computable in $O(|\theta|)$ via autodiff.
- **Neumann series**: approximate the inverse Hessian as a geometric series truncated after $K$ terms.

IFT-based methods (**iMAML**, **T1-T2**, **HOAG**) avoid storing the full unrolled computation graph and scale better than reverse-mode unrolling.

### DARTS First-Order Approximation

DARTS uses a computationally cheap approximation:

$$\nabla_\phi F(\phi, \theta^*) \approx \nabla_\phi F(\phi, \theta^+)$$

where $\theta^+$ is $\theta$ after a single gradient step on $G$. This ignores second-order terms and is only valid when $\theta$ is near convergence. Despite its simplicity, DARTS first-order approximation is surprisingly effective in NAS experiments.

## Truncated Backpropagation Through Time (TBPTT)

For sequence models trained with bilevel objectives (e.g., RNN meta-learning, learned optimizers), unrolling for $T$ steps is essential but memory-prohibitive for large $T$. **TBPTT** limits backpropagation to a window of the last $K < T$ steps:

$$\nabla_\phi F \approx \frac{\partial F}{\partial \theta_T} \cdot \frac{\partial \theta_T}{\partial \theta_{T-K}}$$

This introduces **truncation bias** — gradients from earlier steps are ignored. However, TBPTT is essential for making learned optimizer training (e.g., **VeLO**) computationally tractable.

## Convergence and Non-Convexity

Bilevel optimization is significantly harder than single-level optimization:

- **Non-convexity**: even when $G$ is strongly convex in $\theta$ (guaranteeing existence and uniqueness of $\theta^*(\phi)$), $F(\phi, \theta^*(\phi))$ may be highly non-convex in $\phi$.
- **Non-unique inner solutions**: if $G$ is non-convex in $\theta$ (as is typical for neural networks), $\theta^*(\phi)$ is not uniquely defined, and the hypergradient is undefined or set-valued.
- **Approximate inner solutions**: when the inner problem is only approximately solved (finite training steps), the hypergradient estimate is biased.

Recent theoretical work (Franceschi et al., 2018; Ghadimi and Wang, 2018) provides convergence guarantees for special cases (strongly convex inner problem, approximate IFT-based hypergradients) and characterizes the bias-variance tradeoff in truncated unrolling.

## Scalability Challenges

| Method | Memory | Compute | Bias |
| --- | --- | --- | --- |
| Full unrolling (T steps) | $O(T \cdot \|\theta\|)$ | $O(T)$ | Low |
| TBPTT (K-step truncation) | $O(K \cdot \|\theta\|)$ | $O(K)$ | High |
| IFT with CG | $O(\|\theta\|)$ | $O(\text{CG iters})$ | Low (at convergence) |
| DARTS first-order | $O(\|\theta\|)$ | $O(1)$ | High |

In practice, DARTS-based NAS and MAML-based meta-learning use shallow unrolling (1-5 steps) as the standard tradeoff between computational tractability and gradient quality.

## Memory-Efficient Bilevel Methods

**Gradient checkpointing** applied to the unrolled computation graph reduces memory from $O(T \cdot |\theta|)$ to $O(\sqrt{T} \cdot |\theta|)$ at the cost of an extra forward pass — making longer unrolling tractable.

**DrMAD** (Fu et al., 2016) uses a distillation-based approach: the inner training trajectory is first recorded, then a reverse pass estimates the hypergradient without recomputing all intermediate activations.

**ES-MAML** and **FOMAML** use first-order approximations or evolutionary strategies to estimate the outer gradient without second-order information — sacrificing accuracy for memory efficiency.

## Summary

Bilevel optimization is a principled framework for problems where an outer objective depends on the solution of an inner optimization — ubiquitous in meta-learning (MAML), neural architecture search (DARTS), hyperparameter optimization, learned data augmentation, and adversarial training. Computing the hypergradient — the derivative of the outer objective through the inner solution — is the central challenge, addressable via unrolled differentiation (expensive but accurate), implicit function theorem with conjugate gradient (memory-efficient), or first-order approximations (fast but biased). Scalability to large neural networks requires gradient checkpointing, truncated backpropagation, or shallow unrolling. Theoretical analysis shows that bilevel convergence requires careful management of inner approximation quality, and non-convex inner problems remain an open challenge for large-scale deep learning applications.
