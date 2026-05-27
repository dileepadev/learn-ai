---
title: Adversarial Robustness
description: Understand adversarial attacks on neural networks — FGSM, PGD, Carlini-Wagner, and patch attacks — and the defenses that make models robust, including adversarial training, randomized smoothing for certified guarantees, and AutoAttack for reliable robustness evaluation.
---

Neural networks can be fooled by inputs that are imperceptible to humans but cause dramatic prediction failures. An image classified with 99% confidence as a cat can be perturbed by a few pixel values — changes invisible to the human eye — and be confidently misclassified as a toaster. This fragility is not a corner case: it has practical implications for autonomous vehicles, medical imaging, face recognition, and any security-sensitive deployment of deep learning.

## What Are Adversarial Examples?

An **adversarial example** is an input $\mathbf{x}' = \mathbf{x} + \boldsymbol{\delta}$ where $\boldsymbol{\delta}$ is a small perturbation constrained to an $\ell_p$ ball of radius $\varepsilon$:

$$\|\boldsymbol{\delta}\|_p \leq \varepsilon$$

The $\ell_\infty$ norm (maximum absolute pixel change) is most common in research. The adversary maximizes the model's loss:

$$\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_\infty \leq \varepsilon} \mathcal{L}(f(\mathbf{x} + \boldsymbol{\delta}), y)$$

## Attack Methods

### FGSM — Fast Gradient Sign Method

FGSM (Goodfellow et al., 2014) computes the single-step perturbation in the direction of the gradient sign:

$$\mathbf{x}' = \mathbf{x} + \varepsilon \cdot \mathrm{sign}\!\left(\nabla_{\mathbf{x}} \mathcal{L}(f(\mathbf{x}), y)\right)$$

```python
import torch
import torch.nn.functional as F


def fgsm_attack(model, x, y, epsilon: float = 8/255):
    """Fast Gradient Sign Method attack."""
    x_adv = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    with torch.no_grad():
        x_adv = x + epsilon * x_adv.grad.sign()
        x_adv = x_adv.clamp(0, 1)
    return x_adv
```

FGSM is fast but weak — a single gradient step often fails to find a strong adversarial example.

### PGD — Projected Gradient Descent

PGD (Madry et al., 2018) iterates FGSM steps with projection back onto the $\varepsilon$-ball after each step. It is the dominant attack for adversarial training:

$$\mathbf{x}^{t+1} = \Pi_{\mathbf{x}+\mathcal{S}}\!\left(\mathbf{x}^t + \alpha \cdot \mathrm{sign}\!\left(\nabla_{\mathbf{x}^t} \mathcal{L}(f(\mathbf{x}^t), y)\right)\right)$$

where $\Pi$ projects back into the feasible set (intersection of $\ell_\infty$ ball and $[0,1]^d$) and $\alpha$ is the step size.

```python
def pgd_attack(model, x, y, epsilon: float = 8/255, alpha: float = 2/255, steps: int = 40):
    """PGD attack (Madry et al., 2018)."""
    # Random initialization within epsilon-ball
    x_adv = x.clone() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)

    for _ in range(steps):
        x_adv = x_adv.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            # Project back to epsilon-ball centered at original x
            delta = (x_adv - x).clamp(-epsilon, epsilon)
            x_adv = (x + delta).clamp(0, 1)
    return x_adv
```

### Carlini-Wagner (C&W) Attack

The C&W attack (Carlini & Wagner, 2017) formulates adversarial example generation as an optimization problem that directly minimizes perturbation size subject to a misclassification constraint:

$$\min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_2 + c \cdot \mathcal{L}_{\mathrm{CW}}(f(\mathbf{x} + \boldsymbol{\delta}), y)$$

where $\mathcal{L}_{\mathrm{CW}}$ uses logit differences rather than cross-entropy, enabling more precise control over confidence. C&W is much stronger than PGD against many defenses, historically breaking defenses that appeared robust to FGSM/PGD.

### Patch Attacks

**Adversarial patches** (Brown et al., 2017) are conspicuous, printable perturbations localized to a small patch that can cause misclassification regardless of scene context. Unlike pixel-budget attacks, patches are physically realizable — they can be printed and placed in front of a camera.

## AutoAttack

Evaluating robustness reliably is difficult — many published defenses were later shown to be broken by stronger attacks. **AutoAttack** (Croce & Hein, 2020) is an ensemble of parameter-free attacks that provides a reliable robustness estimate without tuning:

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm="Linf",
    eps=8/255,
    version="standard",  # APGD-CE + APGD-T + FAB + Square
)
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
```

AutoAttack is the standard benchmark for adversarial robustness in the research community.

## Adversarial Training

**Adversarial training** (Madry et al., 2018) is the most reliable empirical defense: generate adversarial examples during training and include them in each minibatch.

```python
from torch.optim import SGD


def adversarial_train_step(model, optimizer, x, y, epsilon=8/255, alpha=2/255, steps=10):
    """One step of PGD adversarial training."""
    model.eval()  # Use eval mode for attack generation (BN fix)
    x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=alpha, steps=steps)

    model.train()
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    optimizer.step()
    return loss.item()


model = ResNet18()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

for epoch in range(200):
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        adversarial_train_step(model, optimizer, x_batch, y_batch)
```

**TRADES** (Zhang et al., 2019) improves upon PGD adversarial training by decomposing the robust loss into natural accuracy plus a regularization term measuring KL divergence between clean and adversarial predictions:

$$\mathcal{L}_{\mathrm{TRADES}} = \mathcal{L}_{\mathrm{CE}}(f(\mathbf{x}), y) + \frac{1}{\lambda} D_{\mathrm{KL}}\!\left(f(\mathbf{x}) \| f(\mathbf{x}')\right)$$

The $\lambda$ parameter controls the accuracy-robustness tradeoff — smaller $\lambda$ increases robustness at the cost of clean accuracy.

## Certified Robustness with Randomized Smoothing

Adversarial training provides empirical robustness — the model withstands known attacks but has no guarantee against unknown attacks. **Randomized smoothing** (Cohen et al., 2019) provides a **certified** $\ell_2$ radius within which no perturbation can change the classification.

Given a base classifier $f$, define a smoothed classifier:

$$g(\mathbf{x}) = \arg\max_c\, \Pr_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})}\!\left[f(\mathbf{x} + \boldsymbol{\epsilon}) = c\right]$$

If the most probable class has probability $p_A \geq 0.5$, the certified $\ell_2$ radius is:

$$r = \sigma \cdot \Phi^{-1}(p_A)$$

```python
import torch
import scipy.stats as stats
import numpy as np


def certify(model, x, sigma: float = 0.25, n_samples: int = 10000, alpha: float = 0.001):
    """
    Certify a single input with randomized smoothing.
    Returns (predicted_class, certified_radius) or abstains.
    """
    model.eval()
    x_expanded = x.unsqueeze(0).expand(n_samples, -1, -1, -1)
    noise = torch.randn_like(x_expanded) * sigma
    with torch.no_grad():
        preds = model(x_expanded + noise).argmax(dim=1)

    counts = torch.bincount(preds, minlength=model.num_classes).cpu().numpy()
    top_class = counts.argmax()

    # One-sided binomial confidence interval (Clopper-Pearson)
    p_A_low = stats.binom.ppf(alpha, n_samples, counts[top_class] / n_samples)
    p_A_low = max(p_A_low, 0.5)

    if p_A_low < 0.5:
        return None, 0.0  # Abstain

    radius = sigma * stats.norm.ppf(p_A_low)
    return top_class, radius
```

## Robustness-Accuracy Tradeoff

A fundamental tension exists between natural accuracy and adversarial robustness. Tsipras et al. (2019) showed theoretically (under certain data distributions) that improving robustness necessarily reduces natural accuracy. On CIFAR-10:

| Method | Clean Acc (%) | Robust Acc (%) at ε=8/255 |
| --- | --- | --- |
| Natural training | 95.0 | < 1 |
| PGD adversarial training | 84.7 | 56.6 |
| TRADES (β=6) | 84.9 | 57.0 |
| WideResNet + AutoAugment + AT | 88.2 | 63.3 |
| Ensemble/self-training SOTA | 92.2 | 71.1 |

## Transferability and Black-Box Attacks

Adversarial examples generated on one model often **transfer** to different models with different architectures or training data — even without access to the target model's weights. This enables black-box attacks:

1. Train a substitute model on query outputs from the target
1. Generate adversarial examples on the substitute model
1. Apply them to the target (transfer attack)

Ensemble attacks (averaging gradients from multiple surrogate models) achieve higher transfer rates. Defense strategies such as adversarial training reduce transferability but do not eliminate it.

## Summary

Adversarial robustness is a fundamental challenge for trustworthy deep learning deployment:

- **Adversarial examples** exploit the geometry of high-dimensional space: tiny perturbations in pixel space can cross decision boundaries
- **FGSM** and **PGD** are the canonical attacks; PGD adversarial training remains the strongest empirical defense
- **AutoAttack** provides a reliable, parameter-free benchmark for evaluating robustness claims
- **Randomized smoothing** offers certified $\ell_2$ robustness guarantees at the cost of inference-time overhead and a robustness-accuracy tradeoff
- The **accuracy-robustness tradeoff** appears fundamental — there is no free defense — but the gap narrows with better training strategies, architectures, and data augmentation
- Understanding adversarial fragility motivates careful threat modeling before deploying neural networks in security-sensitive applications
