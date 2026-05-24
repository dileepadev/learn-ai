---
title: Contextual Bandits
description: Master contextual bandits — the principled framework for sequential decision-making under uncertainty — covering the explore-exploit tradeoff, LinUCB, Thompson sampling, neural bandits, offline evaluation, and real-world applications in recommendation, healthcare, and online advertising.
---

The recommendation problem has a fundamental asymmetry: you can only observe reward for the action you took, never for the actions you didn't. A music app that plays track A can observe whether the user skipped or listened — but cannot observe what would have happened with track B. This is the **contextual bandit problem**: sequentially choosing actions based on observed context, with the goal of maximizing cumulative reward, while facing uncertainty about actions not yet tried.

Contextual bandits sit between supervised learning (no sequential decisions, full feedback) and full reinforcement learning (long-horizon sequential decisions, delayed rewards). They are the right framework for personalization, content recommendation, clinical treatment selection, and online advertising.

## Problem Formulation

At each round $t = 1, 2, \ldots, T$:

1. The environment reveals a **context** $\mathbf{x}_t \in \mathbb{R}^d$
1. The agent selects an **action** $a_t \in \mathcal{A} = \{1, \ldots, K\}$
1. The environment reveals a **reward** $r_t = r(a_t, \mathbf{x}_t) + \varepsilon_t$
1. The agent updates its policy using $(\mathbf{x}_t, a_t, r_t)$

Only the reward for the chosen action is observed — never the counterfactual rewards for unchosen actions. This is the **partial feedback** or **bandit feedback** setting.

The objective is to minimize **cumulative regret**:

$$R_T = \sum_{t=1}^{T} \left[ r(a^*_t, \mathbf{x}_t) - r(a_t, \mathbf{x}_t) \right]$$

where $a^*_t = \arg\max_a r(a, \mathbf{x}_t)$ is the optimal action in hindsight for context $\mathbf{x}_t$.

## The Explore-Exploit Tradeoff

A policy that always exploits current best knowledge misses potentially superior actions — it can get trapped in local optima. A policy that explores too much wastes reward on suboptimal actions. This tension is the central challenge of bandit algorithms.

The three main strategies for balancing this tradeoff are:

- **Epsilon-greedy**: exploit with probability $1 - \varepsilon$, explore randomly with probability $\varepsilon$
- **Upper Confidence Bound (UCB)**: add an optimism bonus to reward estimates — try uncertain actions as if they might be optimal
- **Thompson Sampling**: sample reward parameters from a posterior distribution and act optimally under the sample

## LinUCB

**LinUCB** (Li et al., 2010) assumes a linear reward model: $r(a, \mathbf{x}) = \boldsymbol{\theta}_a^\top \mathbf{x} + \varepsilon$, where each arm $a$ has its own parameter vector $\boldsymbol{\theta}_a$.

For each arm $a$, LinUCB maintains:

- $\mathbf{A}_a = \mathbf{I}_d + \sum_{s \leq t, a_s=a} \mathbf{x}_s \mathbf{x}_s^\top$ — design matrix (ridge regression covariance)
- $\mathbf{b}_a = \sum_{s \leq t, a_s=a} r_s \mathbf{x}_s$ — reward-weighted features

The ridge regression estimate is $\hat{\boldsymbol{\theta}}_a = \mathbf{A}_a^{-1} \mathbf{b}_a$, and the UCB score is:

$$\text{UCB}_a(\mathbf{x}_t) = \hat{\boldsymbol{\theta}}_a^\top \mathbf{x}_t + \alpha \sqrt{\mathbf{x}_t^\top \mathbf{A}_a^{-1} \mathbf{x}_t}$$

The second term is the **uncertainty bonus**: high when $\mathbf{x}_t$ is far from previously observed contexts for arm $a$. The parameter $\alpha > 0$ controls exploration intensity.

```python
import numpy as np


class LinUCB:
    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.alpha = alpha
        self.n_arms = n_arms
        self.d = context_dim

        # Per-arm covariance and reward matrices
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context):
        ucb_scores = []
        for a in range(self.n_arms):
            theta = np.linalg.solve(self.A[a], self.b[a])
            uncertainty = np.sqrt(context @ np.linalg.solve(self.A[a], context))
            ucb_scores.append(theta @ context + self.alpha * uncertainty)
        return int(np.argmax(ucb_scores))

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

LinUCB achieves regret $R_T = \mathcal{O}(d\sqrt{T \log T})$ — the regret grows sublinearly, meaning average regret per round goes to zero.

## Thompson Sampling for Bandits

Thompson Sampling (TS) takes a Bayesian approach: maintain a posterior over reward parameters and sample from it. For linear bandits with Gaussian noise:

- Prior: $\boldsymbol{\theta}_a \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Likelihood: $r \mid \boldsymbol{\theta}_a, \mathbf{x} \sim \mathcal{N}(\boldsymbol{\theta}_a^\top \mathbf{x}, \sigma^2)$
- Posterior (after $n_a$ observations): $\boldsymbol{\theta}_a \mid \text{data} \sim \mathcal{N}(\hat{\boldsymbol{\theta}}_a, \mathbf{A}_a^{-1})$

```python
class LinearThompsonSampling:
    def __init__(self, n_arms, context_dim, sigma=1.0, lambda_reg=1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.sigma2 = sigma ** 2
        self.lam = lambda_reg

        self.A = [self.lam * np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context):
        samples = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            mu = A_inv @ self.b[a]
            # Sample from posterior
            theta_sample = np.random.multivariate_normal(mu, self.sigma2 * A_inv)
            samples.append(theta_sample @ context)
        return int(np.argmax(samples))

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context) / self.sigma2
        self.b[arm] += reward * context / self.sigma2
```

Thompson Sampling often outperforms UCB methods empirically and achieves the same asymptotic regret bounds.

## Neural Bandits

When the reward function is non-linear, linear models are insufficient. **Neural Bandits** use neural networks to model rewards and adapt the UCB or TS framework to the learned representations.

### NeuralUCB

NeuralUCB (Zhou et al., 2020) uses the neural tangent kernel (NTK) approximation: at any time $t$, the gradient vector $\mathbf{g}_a(\mathbf{x}) = \nabla_\theta f_\theta(\mathbf{x}, a)$ (the gradient of the network output with respect to parameters $\theta$) can serve as a linear feature for uncertainty estimation:

$$\text{UCB}_a(\mathbf{x}_t) = f_\theta(\mathbf{x}_t, a) + \nu \sqrt{\mathbf{g}_a(\mathbf{x}_t)^\top \mathbf{Z}^{-1} \mathbf{g}_a(\mathbf{x}_t)}$$

where $\mathbf{Z}$ is a regularized covariance matrix over gradient features. In practice, this is expensive to compute exactly, so implementations use approximate methods (diagonal covariance, random projections).

### Neural Thompson Sampling

A practical neural bandit approach uses an ensemble of networks — each network in the ensemble provides a different reward estimate, and the variance across ensemble members serves as the uncertainty signal:

```python
import torch
import torch.nn as nn


class NeuralBanditEnsemble:
    def __init__(self, n_arms, context_dim, hidden_dim=64, n_heads=5, lr=1e-3):
        self.n_arms = n_arms
        self.n_heads = n_heads

        # Ensemble of reward predictors per arm
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_heads)
        ])
        self.optimizer = torch.optim.Adam(self.networks.parameters(), lr=lr)
        self.replay = []

    def select_arm(self, context):
        ctx = torch.tensor(context, dtype=torch.float32)
        arm_scores = []
        for a in range(self.n_arms):
            # Predict reward for arm a (append one-hot arm encoding)
            arm_ctx = torch.cat([ctx, torch.zeros(self.n_arms).scatter_(0, torch.tensor(a), 1.0)])
            preds = torch.stack([net(arm_ctx).item() for net in self.networks])
            # Thompson: sample from ensemble (each head is a posterior sample)
            arm_scores.append(preds[torch.randint(self.n_heads, (1,))].item())
        return int(np.argmax(arm_scores))

    def update(self, arm, context, reward):
        self.replay.append((context, arm, reward))
        if len(self.replay) >= 32:
            self._train_step()

    def _train_step(self):
        batch = [self.replay[i] for i in np.random.choice(len(self.replay), 32)]
        loss = sum(
            ((net(torch.tensor(c, dtype=torch.float32)) - r) ** 2).mean()
            for c, _, r in batch
            for net in self.networks
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Offline Evaluation and Off-Policy Learning

In production systems, you often want to evaluate a new bandit policy using **logged data** collected by a previous policy (the logging policy $\pi_0$). Naively averaging rewards from logged data is biased because the logging policy may have systematically avoided certain actions.

### Inverse Propensity Scoring (IPS)

IPS re-weights logged rewards by how likely the evaluation policy would have chosen the same action:

$$\hat{V}_{\text{IPS}}(\pi) = \frac{1}{T} \sum_{t=1}^{T} \frac{\pi(a_t \mid \mathbf{x}_t)}{\pi_0(a_t \mid \mathbf{x}_t)} r_t$$

```python
def ips_estimate(eval_policy_probs, logging_policy_probs, actions, rewards, clip=10.0):
    """
    eval_policy_probs: probability of chosen action under eval policy
    logging_policy_probs: probability of chosen action under logging policy
    """
    importance_weights = eval_policy_probs / (logging_policy_probs + 1e-8)
    importance_weights = np.clip(importance_weights, 0, clip)  # Clip for variance reduction
    return np.mean(importance_weights * rewards)
```

### Doubly Robust Estimation

Doubly Robust (DR) combines IPS with a reward model $\hat{r}(a, \mathbf{x})$ — consistent if either the reward model or propensities are correct:

$$\hat{V}_{\text{DR}}(\pi) = \frac{1}{T}\sum_t \left[ \hat{r}(a_t, \mathbf{x}_t) + \frac{\pi(a_t \mid \mathbf{x}_t)}{\pi_0(a_t \mid \mathbf{x}_t)} (r_t - \hat{r}(a_t, \mathbf{x}_t)) \right]$$

## Applications

### Recommendation Systems

News and content recommendation is a canonical bandit application. Microsoft News (Microsoft Research, 2010) deployed LinUCB for personalized news article selection, treating each article as an arm and user/article features as context — demonstrating significant CTR improvements over non-contextual methods.

### Clinical Trial Adaptive Design

Adaptive clinical trials use bandit algorithms to allocate more patients to treatments showing promise during the trial itself, rather than fixing allocation upfront. This raises ethical considerations (patients in the trial benefit from exploration) and statistical challenges (sequential testing while controlling false discovery rate).

### Online Advertising

Bid optimization and ad selection are bandit problems: each ad auction is a round, the context is the user and page features, the action is the bid or selected creative, and the reward is click/conversion. UCB and TS variants are widely deployed in display advertising systems.

### Hyperparameter Tuning as a Bandit

Successive Halving and Hyperband formulate hyperparameter search as a bandit problem: each configuration is an arm, the reward is validation performance, and the explore-exploit tradeoff determines how many resources to allocate to each configuration (see Multi-Fidelity Optimization).

## Regret Bounds Summary

| Algorithm | Regret Bound | Assumptions |
| --- | --- | --- |
| $\varepsilon$-greedy | $\mathcal{O}(T^{2/3})$ | Fixed $\varepsilon$, stochastic rewards |
| LinUCB | $\mathcal{O}(d\sqrt{T \log T})$ | Linear rewards, $d$ features |
| Linear TS | $\mathcal{O}(d\sqrt{T})$ | Linear rewards, Gaussian noise |
| NeuralUCB | $\mathcal{O}(\tilde{d}\sqrt{T})$ | Overparameterized network, NTK regime |
| Lower bound | $\Omega(\sqrt{KT})$ | Any algorithm, $K$ arms |

## Summary

Contextual bandits provide a principled framework for sequential decision-making with partial feedback:

- **LinUCB** adds an optimism bonus based on linear uncertainty estimates — regret scales as $\mathcal{O}(d\sqrt{T})$
- **Thompson Sampling** maintains a posterior over reward parameters and acts optimally under posterior samples — empirically strong and Bayesian-principled
- **Neural bandits** extend these ideas to non-linear reward functions via gradient features (NeuralUCB) or ensemble uncertainty (Neural TS)
- **Offline evaluation** via IPS and doubly robust estimation allows safe evaluation and learning from logged historical data

In practice, bandit algorithms outperform supervised approaches in personalization settings precisely because they adapt to individual preferences over time — learning not just from what worked, but from the uncertainty about what hasn't been tried.
