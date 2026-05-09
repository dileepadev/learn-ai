---
title: Differential Privacy in Machine Learning
description: A rigorous guide to differential privacy — the mathematical framework for training machine learning models on sensitive data while providing provable privacy guarantees — covering DP-SGD, privacy accounting, and the privacy-utility tradeoff.
---

Differential privacy (DP) is a formal mathematical definition of privacy for statistical computations. When applied to machine learning, it provides a quantifiable guarantee: the trained model reveals very little information about any individual training example, regardless of what auxiliary information an adversary possesses. Unlike informal privacy practices, differential privacy is a **proof-based guarantee** — not a heuristic — making it the gold standard for privacy-preserving machine learning.

## The Formal Definition

A randomized algorithm $\mathcal{M}$ satisfies **$(\varepsilon, \delta)$-differential privacy** if for any two datasets $D$ and $D'$ that differ by exactly one record, and for any subset of outputs $S \subseteq \text{Range}(\mathcal{M})$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

### Interpreting the Parameters

- **$\varepsilon$ (epsilon)** — the **privacy budget** or **privacy loss**. Smaller $\varepsilon$ means stronger privacy. $\varepsilon = 0$ means the output is identical for all datasets (perfect privacy, zero utility). Practical deployments use $\varepsilon \in [1, 10]$; values below 1 are considered very strong.
- **$\delta$ (delta)** — an additive failure probability. Allows the bound to be violated with probability at most $\delta$. Typically set to $\delta \ll 1/n$ where $n$ is the dataset size (e.g., $\delta = 10^{-5}$). $\delta = 0$ gives pure $\varepsilon$-DP; $\delta > 0$ gives approximate DP.

The key intuition: an adversary who observes the output $\mathcal{M}(D)$ gains almost no information about whether any individual record was in $D$ or not — their posterior belief about any individual record changes by at most a factor of $e^\varepsilon$.

## The Gaussian and Laplace Mechanisms

Differential privacy is achieved by adding calibrated noise to a computation's output.

### Sensitivity

The **$\ell_2$-sensitivity** (or $\ell_1$-sensitivity) of a function $f$ bounds how much one record can change the output:

$$\Delta_2 f = \max_{D, D'} \|f(D) - f(D')\|_2$$

where $D$ and $D'$ differ by one record.

### Gaussian Mechanism

Add noise proportional to the sensitivity divided by $\varepsilon$:

$$\mathcal{M}(D) = f(D) + \mathcal{N}\!\left(0,\, \sigma^2 I\right), \quad \sigma = \frac{\Delta_2 f \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

This achieves $(\varepsilon, \delta)$-DP and is the standard mechanism used in DP-SGD.

### Laplace Mechanism

$$\mathcal{M}(D) = f(D) + \text{Lap}\!\left(0,\, \frac{\Delta_1 f}{\varepsilon}\right)$$

Achieves pure $\varepsilon$-DP (with $\delta = 0$). Preferred when the output is a scalar or low-dimensional vector and sensitivity can be tightly bounded.

## DP-SGD: Differentially Private Stochastic Gradient Descent

The primary algorithm for training neural networks with DP guarantees is **DP-SGD** (Abadi et al., 2016). It modifies the standard SGD training loop with two operations:

### Algorithm

For each training step:

1. **Sample a mini-batch** $B$ of size $m$ using Poisson sampling (each example included independently with probability $q = m/n$).
2. **Compute per-example gradients**: for each example $i \in B$, compute the gradient $\mathbf{g}_i = \nabla_\theta \mathcal{L}(\theta; x_i)$.
3. **Clip gradients**: clip each gradient to $\ell_2$-norm $C$:

$$\tilde{\mathbf{g}}_i = \mathbf{g}_i \cdot \min\!\left(1, \frac{C}{\|\mathbf{g}_i\|_2}\right)$$

1. **Add noise**: compute the noisy aggregate:

$$\hat{\mathbf{g}} = \frac{1}{m}\left(\sum_{i \in B} \tilde{\mathbf{g}}_i + \mathcal{N}(0,\, \sigma^2 C^2 I)\right)$$

1. **Update parameters**: $\theta \leftarrow \theta - \eta \hat{\mathbf{g}}$

### Why Clipping and Noise?

- **Clipping** bounds the sensitivity of the gradient aggregation: one record can change the sum of gradients by at most $C$ in $\ell_2$-norm.
- **Noise calibrated to $C$** ensures privacy. The ratio $\sigma = C / (\text{noise std})$ is the **noise multiplier** — the key hyperparameter controlling the privacy-utility tradeoff.

## Privacy Accounting

Running many training steps each with a privacy cost of $(\varepsilon_i, \delta_i)$ accumulates privacy loss. Naive **sequential composition** would give $\sum_i \varepsilon_i$ — too loose for thousands of SGD steps.

### Moments Accountant / Rényi DP

The **moments accountant** (Abadi et al., 2016) and its generalization, **Rényi Differential Privacy (RDP)** (Mironov, 2017), provide tighter composition bounds using the moment generating function of the privacy loss random variable.

RDP parameterizes privacy by order $\alpha$:

$$D_\alpha\!\left(\mathcal{M}(D) \| \mathcal{M}(D')\right) \leq \frac{\alpha}{2\sigma^2}$$

for the Gaussian mechanism. RDP composes linearly:

$$D_\alpha(\mathcal{M}_1 \circ \mathcal{M}_2) \leq D_\alpha(\mathcal{M}_1) + D_\alpha(\mathcal{M}_2)$$

After $T$ steps with sampling rate $q$, the total RDP is converted back to $(\varepsilon, \delta)$-DP via a closed-form conversion. The **Google DP Accounting library** and **OpenDP** automate this for practitioners.

## The Privacy–Utility Tradeoff

DP noise degrades model accuracy. The tradeoff depends on:

- **Dataset size $n$**: Larger datasets absorb noise better. For a fixed $(\varepsilon, \delta)$, accuracy loss decreases as $O(1/n)$.
- **Model dimensionality**: High-dimensional gradient vectors require more noise, hurting convergence.
- **Number of training steps $T$**: More steps means more privacy budget consumed.
- **Clipping norm $C$**: Too small clips useful gradients; too large requires more noise.

### Typical Numbers

For MNIST with a small CNN, $(\varepsilon=1, \delta=10^{-5})$-DP reduces accuracy from 99% to ~97%. For large-scale language models fine-tuned with DP (e.g., GPT-2 on medical records at $\varepsilon=3$), perplexity increases by 5–10 points compared to non-private fine-tuning.

## Private Fine-Tuning of Foundation Models

Training large foundation models from scratch with DP is extremely expensive due to the noise overhead. The practical approach is **private fine-tuning**:

1. Pretrain on public data (no DP required — public data has no individual privacy risk).
2. Fine-tune on private data with DP-SGD.

Because only the fine-tuning phase requires DP, and fine-tuning updates far fewer parameters (especially with LoRA or adapter layers), the noise overhead is dramatically reduced. **DP-LoRA** fine-tunes only low-rank adapters, concentrating the privacy budget on a small parameter subspace and achieving near-non-private accuracy at $\varepsilon \approx 3$–$8$ on NLP benchmarks.

## Local vs. Central Differential Privacy

| Setting | Who Adds Noise | Trust Model | Utility |
| --- | --- | --- | --- |
| **Central DP** | A trusted curator after collecting raw data | Trust the data collector | Higher — noise added once |
| **Local DP** | Each user before sending data | No trust required | Lower — noise added per user |
| **Shuffle DP** | A shuffler anonymizes before aggregation | Trust the shuffler | Intermediate |

**Local DP** is used in practice by Apple (keyboard suggestions), Google (Chrome telemetry), and Microsoft (Windows diagnostics). Each user applies a randomized response mechanism before sending data, providing privacy even if the server is compromised.

## Auditing DP Guarantees

Implementing DP correctly is difficult — software bugs can silently violate the mathematical guarantee. **DP auditing** empirically verifies that a claimed $\varepsilon$ is accurate:

1. Train many models with a **canary** (a specific record planted in some runs).
2. Run membership inference attacks to estimate how often the canary is detected.
3. Compare empirical success rates to the theoretical $\varepsilon$ bound.

Tools like **DP-Auditorium** (Google) and **ML Privacy Meter** automate this process and have identified subtle implementation errors in published DP codebases.

## Key Libraries

| Library | Focus |
| --- | --- |
| Opacus (Meta) | DP-SGD for PyTorch with per-example gradient support |
| TF Privacy (Google) | DP-SGD for TensorFlow + privacy accounting |
| OpenDP | Modular DP primitives and compositors |
| PySyft | Federated learning + DP for distributed settings |
| Google DP Accounting | Tight RDP and GDP accountants |

## Connections to Federated Learning

DP and federated learning are natural partners. In federated learning, model updates (gradients) are aggregated across devices without sharing raw data. Adding DP noise to the aggregated gradients (**central DP in FL**) or to local updates before sending (**local DP in FL**) provides formal guarantees even against a compromised aggregation server. **Google's federated DP** framework, used in Gboard, combines Secure Aggregation (cryptographic protection of individual updates) with central DP noise at the population level.

## Summary

Differential privacy transforms the informal goal of "not exposing user data" into a mathematical theorem: any adversary, regardless of auxiliary information, cannot reliably determine whether a given individual's data was used to train the model. DP-SGD, combined with tight privacy accounting via RDP and practical implementations like Opacus, brings this guarantee to neural network training with manageable accuracy cost — especially when combined with public pretraining and parameter-efficient fine-tuning. As privacy regulations tighten globally, differential privacy is increasingly the standard of evidence for privacy compliance in production machine learning systems.
