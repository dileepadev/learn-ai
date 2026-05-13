---
title: PAC-Bayes Bounds
description: Understand PAC-Bayes generalization bounds — a powerful framework that provides tight, data-dependent generalization guarantees for randomized predictors by combining Bayesian posteriors with PAC learning theory.
---

PAC-Bayes theory provides generalization bounds for **randomized classifiers** — models that sample their parameters from a posterior distribution rather than using a fixed parameter setting. These bounds are notable for being tight enough to be informative in practice, naturally incorporating prior knowledge, and providing a principled connection between Bayesian inference and statistical learning theory.

## Background: PAC Learning

**Probably Approximately Correct (PAC) learning** (Valiant, 1984) formalizes when a learning algorithm is reliable. A learner is PAC-learning if, with probability at least $1 - \delta$ over training set draws, the learned hypothesis has generalization error at most $\epsilon$ above its training error:

$$\Pr_{S \sim \mathcal{D}^n}\left[R(h) \leq \hat{R}(h) + \text{complexity}(h)\right] \geq 1 - \delta$$

Classical PAC bounds use complexity measures like VC dimension, Rademacher complexity, or covering numbers. These bounds are often loose for overparameterized models like neural networks.

## The PAC-Bayes Framework

**PAC-Bayes** (McAllester, 1999; Seeger, 2002) generalizes PAC learning to distributions over hypotheses. Instead of a single classifier $h$, consider a **posterior** $Q$ over a hypothesis class $\mathcal{H}$. The **Gibbs classifier** samples $h \sim Q$ at each prediction.

### Key Quantities

- **Prior** $P$: a distribution over $\mathcal{H}$ chosen before seeing training data
- **Posterior** $Q$: a data-dependent distribution after training
- **Expected generalization risk**: $R(Q) = \mathbb{E}_{h \sim Q}[R(h)]$
- **Expected empirical risk**: $\hat{R}(Q) = \mathbb{E}_{h \sim Q}[\hat{R}(h)]$

## The Main PAC-Bayes Theorem

**McAllester's PAC-Bayes Bound** (1999): For any prior $P$ over $\mathcal{H}$, any $\delta \in (0, 1)$, with probability at least $1 - \delta$ over training sets $S$ of size $n$, for **all** posteriors $Q$:

$$R(Q) \leq \hat{R}(Q) + \sqrt{\frac{\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta)}{2n}}$$

where $\text{KL}(Q \| P) = \mathbb{E}_{h \sim Q}\left[\ln \frac{Q(h)}{P(h)}\right]$ is the KL divergence from posterior to prior.

### Interpretation

The bound says: **generalization gap ≤ training error + complexity term**, where complexity is measured by $\text{KL}(Q \| P)$ — how much the learned posterior has deviated from the prior. This has an elegant Bayesian interpretation:

- A posterior that stays close to the prior (small KL) generalizes well
- The prior encodes our inductive bias; posteriors that align with it are penalized less
- The bound holds simultaneously for **all** possible posteriors $Q$, not just the learned one

## Catoni's Tighter Bound

McAllester's bound involves a square root. Catoni (2007) derived a tighter form. For binary classification with $\{0, 1\}$ losses, the **kl-bound** (using the binary KL divergence) is tighter:

$$\text{kl}(\hat{R}(Q) \| R(Q)) \leq \frac{\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta)}{n}$$

where $\text{kl}(q \| p) = q \ln \frac{q}{p} + (1-q) \ln \frac{1-q}{1-p}$ is the binary KL divergence.

This bound is **not** in closed form but can be inverted numerically to get a tighter upper bound on $R(Q)$.

## Choosing the Prior and Posterior

### Gaussian Priors and Posteriors

For neural networks parameterized by $\theta \in \mathbb{R}^d$, a common choice is:

- **Prior**: $P = \mathcal{N}(0, \sigma_0^2 I)$ or $\mathcal{N}(\theta_0, \sigma_0^2 I)$ (centered at random init $\theta_0$)
- **Posterior**: $Q = \mathcal{N}(\mu, \sigma^2 I)$ where $\mu$ are trained means and $\sigma$ are learned perturbation scales

The KL divergence in closed form:

$$\text{KL}(Q \| P) = \frac{1}{2}\sum_{i=1}^d \left(\frac{\sigma_i^2}{\sigma_0^2} + \frac{(\mu_i - \mu_{0,i})^2}{\sigma_0^2} - 1 - \ln\frac{\sigma_i^2}{\sigma_0^2}\right)$$

### Perturbation-Based Priors

Dziugaite & Roy (2017) introduced **non-vacuous PAC-Bayes bounds** for neural networks by optimizing the PAC-Bayes objective directly:

$$\min_{Q} \hat{R}(Q) + \sqrt{\frac{\text{KL}(Q \| P)}{2n}}$$

They used SGD to minimize this bound end-to-end on MNIST, obtaining bounds below 2% — the first non-vacuous bounds for realistic neural networks.

## PAC-Bayes and Flatness

A key insight connecting PAC-Bayes to modern deep learning: **flat minima generalize better**, and PAC-Bayes explains why.

If the loss landscape is flat around $\theta^*$, a perturbation $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ does not significantly change the loss. This means:

$$\hat{R}(\mathcal{N}(\theta^*, \sigma^2 I)) \approx \hat{R}(\theta^*)$$

A flat solution can afford large $\sigma$ (wide Gaussian posterior) while maintaining low training error. The KL divergence $\text{KL}(\mathcal{N}(\theta^*, \sigma^2 I) \| P)$ is then small (broad posterior close to broad prior), giving a tight bound.

This directly motivates **Sharpness-Aware Minimization (SAM)**, which explicitly finds flat minima by minimizing worst-case perturbed loss.

## PAC-Bayes and the Gibbs Classifier

The Gibbs classifier samples $h \sim Q$ independently for each prediction. In practice, one uses the **majority vote** classifier (Bayes optimal aggregation) or the **mean predictor** $f_{\mathbb{E}[h]}$. These relate to the Gibbs bound:

$$R(\text{majority vote}) \leq 2 R(Q_{\text{Gibbs}})$$

So a PAC-Bayes bound on the Gibbs classifier implies a bound on the deterministic ensemble.

## Tighter Variants and Extensions

### Seeger's Bound

Seeger (2002) provides a tighter bound using the binary KL, applicable when the loss is in $[0,1]$:

$$\text{kl}(\hat{R}(Q) \| R(Q)) \leq \frac{1}{n}\left(\text{KL}(Q \| P) + \ln \frac{2\sqrt{n}}{\delta}\right)$$

### Data-Dependent Priors

Ambroladze et al. (2007) showed that priors can be chosen based on a **held-out** portion of data without invalidating the bound. This allows prior-centering at a model trained on a validation split, dramatically reducing KL divergence.

### PAC-Bayes with Aggregate Posteriors

**Aggregated PAC-Bayes** bounds apply to the full posterior predictive rather than individual samples, directly bounding the expected risk of the mean predictor under distributional perturbations.

### Disintegrated Bounds

Standard PAC-Bayes bounds hold uniformly over all posteriors. **Disintegrated** bounds (Blanchard & Fleuret, 2007) condition on the training set, holding only for the specific posterior learned from that set — enabling tighter bounds at the cost of losing the uniform guarantee.

## Practical Evaluation of Tightness

PAC-Bayes bounds are considered useful when they are **non-vacuous** — strictly below the trivial bound of 1 (for 0/1 loss) or below 100% error. Progress milestones:

| Year | Setting | Bound |
| --- | --- | --- |
| 2017 | MNIST, 2-layer MLP | ~1.6% |
| 2019 | MNIST, CNN | ~0.6% |
| 2021 | CIFAR-10, ResNet | ~4.5% |
| 2023 | CIFAR-10, various | ~2–3% |

These bounds are still looser than empirical test error (~5–10% on CIFAR-10 for a competitive model, though bounds are for different model configurations), but the gap is narrowing.

## Connections to Other Topics

### PAC-Bayes and Variational Inference

The PAC-Bayes objective $\hat{R}(Q) + \beta \cdot \text{KL}(Q \| P)$ is identical in form to the **ELBO** used in variational Bayes:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{h \sim Q}[\log p(D | h)] - \text{KL}(Q \| P)$$

This connection means that training a Bayesian neural network by maximizing the ELBO has a natural PAC-Bayes generalization certificate.

### PAC-Bayes and Information Theory

The KL term $\text{KL}(Q \| P)$ measures the information in the posterior about the training data. Minimizing it is equivalent to the **minimum description length (MDL)** principle: prefer hypotheses that compress the training data well.

### PAC-Bayes and Double Descent

PAC-Bayes bounds based on KL divergence can explain why interpolating models generalize: in the overparameterized regime, SGD finds solutions with small weight norms (small KL from a zero-mean prior), which the bound rewards. This provides a theoretical account of implicit regularization.

## Summary

PAC-Bayes theory provides a powerful, flexible framework for generalization bounds that:

- Apply to randomized predictors (posteriors over models)
- Incorporate prior knowledge through the KL divergence term
- Connect naturally to Bayesian inference, variational methods, and flatness-based generalization
- Have been made non-vacuous for realistic deep learning settings

As a bridge between classical statistical learning theory and modern deep learning practice, PAC-Bayes remains one of the most active areas in learning theory, with ongoing work on tightening bounds, handling non-IID data, and scaling to large language models.
