---
title: Sharpness-Aware Minimization
description: Understand Sharpness-Aware Minimization (SAM) — the optimizer that explicitly seeks flat minima to improve generalization. Covers the flat minima hypothesis, PAC-Bayes bounds, the SAM perturbation objective, ASAM adaptive normalization, Fisher-SAM, m-SAM for distributed training, and connections to stochastic weight averaging.
---

Modern neural networks are trained using first-order optimizers (SGD, Adam) that find parameter configurations achieving low training loss. But not all low-loss configurations generalize equally well. **Sharpness-Aware Minimization (SAM)** is an optimizer that explicitly seeks configurations where the training loss is simultaneously low and **insensitive to small perturbations** — flat minima that generalize better to unseen data.

## The Flat Minima Hypothesis

The intuition that flat minima generalize better than sharp ones dates to **Hochreiter and Schmidhuber (1997)**, who argued that the description length of a model is shorter when encoded in a flat minimum (small changes in weights cause small changes in loss, so weights can be specified less precisely).

Formally, consider a sharpness measure:

$$\text{Sharpness}(\theta) = \max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta)$$

This is the maximum loss increase achievable by perturbing the parameters $\theta$ within an $\ell_2$ ball of radius $\rho$. A **flat minimum** has low sharpness (the loss surface is relatively uniform nearby); a **sharp minimum** has high sharpness (the loss surface has steep walls nearby).

The generalization gap — train loss minus test loss — correlates positively with sharpness: models that converge to sharp minima tend to overfit more than those converging to flat minima, even at the same train loss. This observation has been confirmed empirically across architectures (ResNets, ViTs, Transformers) and tasks (image classification, NLP, speech).

## PAC-Bayes Theoretical Foundation

**PAC-Bayes bounds** provide theoretical grounding for the flat minima hypothesis. Foret et al. (2021) derive a generalization bound based on the PAC-Bayes theorem:

$$\mathcal{L}_\text{test}(\theta) \leq \max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}_\text{train}(\theta + \epsilon) + h\!\left(\frac{m}{\rho^2}\right)$$

where $m$ is the number of training samples and $h$ is a complexity term. The right-hand side is precisely the **perturbed training loss** — the worst-case training loss in the neighborhood of $\theta$. Minimizing this bound minimizes both the perturbed training loss (encouraging flat minima) and implicitly accounts for complexity.

This bound motivates the SAM objective directly: instead of minimizing $\mathcal{L}(\theta)$, minimize $\max_{\|\epsilon\|\leq\rho} \mathcal{L}(\theta + \epsilon)$.

## The SAM Optimizer

**SAM** (Foret, Kleiner, Mobahi, Neyshabur, 2021) solves the following bi-level optimization:

$$\min_\theta \max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}(\theta + \epsilon)$$

### Two-Step Procedure

Each SAM training step requires two forward-backward passes:

**Step 1 — Compute the worst-case perturbation:**

The inner maximization $\max_{\|\epsilon\|\leq\rho} \mathcal{L}(\theta + \epsilon)$ is approximated by one gradient ascent step. The worst-case perturbation is the direction of steepest loss increase, normalized to the boundary of the $\rho$-ball:

$$\hat{\epsilon}(\theta) = \rho \cdot \frac{\nabla_\theta \mathcal{L}(\theta)}{\|\nabla_\theta \mathcal{L}(\theta)\|_2}$$

This is the first-order approximation to the solution of the inner max.

**Step 2 — Update parameters at the perturbed point:**

Compute gradients at $\theta + \hat{\epsilon}(\theta)$ and use them to update $\theta$:

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta + \hat{\epsilon}(\theta))$$

The update direction points toward reducing the worst-case perturbed loss, encouraging the optimizer to avoid sharp regions where small parameter changes cause large loss increases.

### Computational Cost

SAM requires **2× the compute per step** compared to SGD or Adam: one forward-backward pass to compute $\hat{\epsilon}$, and another to compute the update gradient. In practice, the additional pass can be done in half-batch mode to reduce wall-clock overhead, but the total FLOP cost remains 2×. Despite this cost, SAM consistently achieves better test accuracy than SGD/Adam with the same number of epochs, often more than compensating for the computational overhead.

## ASAM: Adaptive Sharpness-Aware Minimization

A known weakness of SAM is that its $\ell_2$ perturbation radius $\rho$ does not account for the **scale of individual parameters**. A weight of magnitude $10^{-3}$ and a weight of magnitude $10^3$ receive equal perturbation budget, but an absolute perturbation of $\rho = 0.05$ has very different relative effects on these weights.

**ASAM** (Kwon et al., 2021) addresses this with an **adaptive normalization** of the perturbation:

$$\hat{\epsilon}^\text{ASAM}(\theta) = \rho \cdot \frac{T_\theta \nabla_\theta \mathcal{L}(\theta)}{\|T_\theta \nabla_\theta \mathcal{L}(\theta)\|_2}$$

where $T_\theta = \text{diag}(|\theta_1|, |\theta_2|, \ldots, |\theta_n|)$ is a diagonal scaling matrix with the absolute values of each parameter. The effective perturbation for parameter $\theta_i$ is $\rho \cdot |\theta_i|$, making the perturbation **scale-invariant**.

ASAM improves over SAM consistently on image classification benchmarks (CIFAR-10/100, ImageNet), closing the remaining gap to the performance of larger batch sizes and more tuned hyperparameters.

## Fisher-SAM

**Fisher-SAM** (Kim et al., 2022) replaces the $\ell_2$ ball with a **Fisher information geometry** ball:

$$\hat{\epsilon}^\text{Fisher}(\theta) = \rho \cdot \frac{F(\theta)^{-1} \nabla_\theta \mathcal{L}(\theta)}{\|F(\theta)^{-1/2}\nabla_\theta \mathcal{L}(\theta)\|_2}$$

where $F(\theta)$ is the Fisher information matrix. The Fisher metric defines sharpness in the space of distributions rather than parameter space, making the perturbation **invariant to reparameterization**.

Because computing $F(\theta)^{-1}$ exactly is infeasible for large networks, Fisher-SAM uses the empirical Fisher (outer product of gradients) with diagonal approximation. Despite this approximation, Fisher-SAM achieves better calibration and generalization than ASAM in several settings, particularly on language model fine-tuning.

## m-SAM for Distributed Training

Scaling SAM to large-batch distributed training introduces a subtlety: computing $\hat{\epsilon}$ on the full dataset is computationally equivalent to a full gradient step and defeats the purpose of mini-batching.

**m-SAM** (Andriushchenko and Flammarion, 2022) addresses this by computing the perturbation on a **random subset of $m$ examples** and the update gradient on the remaining mini-batch:

1. Split mini-batch into perturbation set (size $m$) and update set (size $B - m$).
1. Compute $\hat{\epsilon}$ from gradients on the perturbation set.
1. Compute update gradient at $\theta + \hat{\epsilon}$ from the update set.

This reduces variance in the sharpness estimate (compared to computing $\hat{\epsilon}$ from a single sample) while maintaining computational efficiency. m-SAM is the recommended variant for large-scale distributed training (e.g., ImageNet-scale training on multiple GPUs).

## Connections to Stochastic Weight Averaging

**Stochastic Weight Averaging (SWA)** (Izmailov et al., 2018) finds flat minima by averaging model weights along the SGD trajectory (using a cyclic or constant learning rate schedule). This averaging corresponds to finding a point in the weight space where many SGD iterates have low loss — geometrically, a flat region where the loss surface is wide.

SAM and SWA are **complementary**:

- **SAM** actively seeks flat minima by perturbing weights during training (computationally expensive but dynamically guided).
- **SWA** finds flat minima post-hoc by averaging (computationally cheap but requires extended training).

**SWAD** (Stochastic Weight Averaging Densely, Cha et al., 2021) combines both: it applies SAM during training while maintaining a running average of weights, achieving the best of both approaches for domain generalization.

## Practical Considerations

- **Perturbation radius $\rho$**: the most important hyperparameter. Too small: insufficient flattening effect. Too large: overcorrects and slows convergence. Typical values: $\rho = 0.05$ for CIFAR, $\rho = 0.1$ for ImageNet.
- **Learning rate**: SAM generally benefits from higher learning rates than SGD/Adam with the same architecture, as the perturbation step provides implicit exploration.
- **Batch size**: SAM's advantage over SGD is larger for small batch sizes, where sharp minima are more problematic due to gradient noise.
- **Epoch count**: SAM often benefits from more training epochs, as the flat-minimum geometry requires more iterations to fully converge.

## Results and Benchmarks

| Method | CIFAR-10 Error | CIFAR-100 Error | ImageNet Top-1 |
| --- | --- | --- | --- |
| SGD (WRN-28-10) | 3.1% | 18.9% | — |
| SAM (WRN-28-10) | 2.6% | 16.8% | — |
| ASAM (WRN-28-10) | 2.4% | 15.9% | — |
| SAM (ViT-L/16) | — | — | 85.8% |
| SGD (ViT-L/16) | — | — | 85.1% |

SAM's benefit is largest on Vision Transformers, which are prone to sharp minima due to their large parameter count and limited implicit regularization compared to CNNs with built-in locality bias.

## Summary

Sharpness-Aware Minimization operationalizes the flat minima hypothesis — that flatter loss landscapes generalize better — into a practical optimization algorithm. By seeking parameters that minimize the worst-case perturbed training loss, SAM actively avoids sharp minima at the cost of 2× compute per step. ASAM addresses parameter-scale sensitivity through adaptive perturbation normalization; Fisher-SAM uses information geometry for reparameterization invariance; m-SAM scales SAM to distributed large-batch training. SAM's connection to PAC-Bayes bounds provides theoretical justification, while its complementary relationship with SWA enables further generalization improvement through weight averaging. SAM has become the default choice for competitive image classification training and is increasingly adopted for language model fine-tuning.
