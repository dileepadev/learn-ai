---
title: Trust Region Policy Optimization (TRPO)
description: Understand Trust Region Policy Optimization (TRPO), monotonic improvement guarantees, KL-divergence constraints, Natural Policy Gradient, and conjugate gradient optimization.
---

In reinforcement learning, standard policy gradient methods update policy parameters in the direction of steepest ascent: $\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)$. However, taking steps directly in **parameter space** is dangerous: a small change in weight parameters can result in an unpredictably large change in the resulting action **probability distribution**, destabilizing the policy and causing training performance to collapse.

**Trust Region Policy Optimization (TRPO)**, developed by Schulman et al. (2015) based on theoretical foundations by Kakade & Langford, resolved this issue by enforcing a constraint on policy changes directly in **probability distribution space** via Kullback-Leibler (KL) divergence, providing theoretical guarantees of **monotonic policy improvement**.

---

## Monotonic Improvement Theory

For any two arbitrary policies $\pi$ and $\tilde{\pi}$, their expected performance $J(\tilde{\pi})$ relates to $J(\pi)$ via the following identity:

$$J(\tilde{\pi}) = J(\pi) + \mathbb{E}_{\tau \sim \tilde{\pi}}\left[ \sum_{t=0}^\infty \gamma^t A_\pi(s_t, a_t) \right]$$

Because sampling directly from the candidate policy $\tilde{\pi}$ during optimization is intractable, TRPO optimizes a **local surrogate objective** $L_\pi(\tilde{\pi})$ that approximates $J(\tilde{\pi})$ using state distributions sampled from the current policy $\pi$:

$$L_\pi(\tilde{\pi}) = J(\pi) + \sum_{s} \rho_\pi(s) \sum_{a} \tilde{\pi}(a \mid s) A_\pi(s, a)$$

Kakade and Langford proved that $J(\tilde{\pi})$ is bounded from below by the surrogate objective minus a penalty proportional to the maximum KL-divergence between the policies:

$$J(\tilde{\pi}) \ge L_\pi(\tilde{\pi}) - C \cdot D_{\text{KL}}^{\max}(\pi, \tilde{\pi})$$

where $C = \frac{4 \epsilon \gamma}{(1 - \gamma)^2}$ and $\epsilon = \max_{s, a} |A_\pi(s, a)|$. 

While this guarantee ensures that optimizing the lower bound never degrades policy performance, the theoretical penalty term $C$ is too large in practice, leading to tiny, overly conservative updates.

---

## The TRPO Constrained Optimization Problem

Rather than penalizing KL-divergence with a fixed multiplier, TRPO reformulates policy updates as a **hard-constrained optimization problem** within a trust region:

$$\max_{\theta} \; \mathbb{E}_{s \sim \rho_{\theta_k}, a \sim \pi_{\theta_k}}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)} A_{\theta_k}(s, a) \right]$$

$$\text{subject to} \quad \bar{D}_{\text{KL}}(\theta_k, \theta) \le \delta$$

where $\bar{D}_{\text{KL}}(\theta_k, \theta) = \mathbb{E}_{s \sim \rho_{\theta_k}}\left[ D_{\text{KL}}\left(\pi_{\theta_k}(\cdot \mid s) \;\parallel\; \pi_\theta(\cdot \mid s)\right) \right]$ is the average KL-divergence across visited states, and $\delta > 0$ defines the radius of the trust region (typically $\delta \approx 0.01$).

```
                      Parameter Space vs. Distribution Space
Parameter Space:
               Δθ
   θ_old ─────────────► θ_new  (Small step in Euclidean weights can drastically
                                 alter action outputs)

Trust Region Distribution Space:
               ┌───────────────────────────┐
               │    Trust Region:          │
               │   D_KL(π_old || π_new) ≤ δ│
               │         • π_new           │
               │        /                  │
               │       /                   │
               │      • π_old              │
               └───────────────────────────┘
```

---

## Practical Solution: Second-Order Approximation

Solving this non-linear constrained optimization problem on deep neural networks requires expanding both the objective and the constraint around the current policy $\theta_k$ using Taylor approximations:

1. **Linear Approximation of Objective:**
   $$L(\theta) \approx L(\theta_k) + g^\top (\theta - \theta_k)$$
   where $g = \nabla_\theta L(\theta)\big|_{\theta = \theta_k}$ is the standard policy gradient vector.

2. **Quadratic Approximation of KL Constraint:**
   Since $D_{\text{KL}}(\theta_k \parallel \theta_k) = 0$ and its first derivative is zero at $\theta = \theta_k$:
   $$\bar{D}_{\text{KL}}(\theta_k, \theta) \approx \frac{1}{2} (\theta - \theta_k)^\top H (\theta - \theta_k)$$
   where $H = \nabla^2_\theta \bar{D}_{\text{KL}}(\theta_k, \theta)\big|_{\theta = \theta_k}$ is the **Fisher Information Matrix (FIM)**.

This simplifies TRPO to a quadratic programming problem:

$$\max_{\Delta \theta} \; g^\top \Delta \theta \quad \text{subject to} \quad \frac{1}{2} \Delta \theta^\top H \Delta \theta \le \delta$$

Using the method of Lagrange multipliers, the analytical solution is:

$$\Delta \theta = \sqrt{\frac{2 \delta}{g^\top H^{-1} g}} \, H^{-1} g$$

---

## The Conjugate Gradient Method & Backtracking Line Search

For a neural network with millions of parameters, directly computing and inverting the $N \times N$ Fisher Information Matrix $H$ is computationally impossible. TRPO circumvents this with two algorithms:

### 1. Conjugate Gradient (CG) for Hessian-Vector Products
Instead of inverting $H$, TRPO computes the vector $x = H^{-1} g$ by solving the linear system $H x = g$ using the **Conjugate Gradient method**. CG requires only matrix-vector products $H v$, which can be computed efficiently via automatic differentiation without ever instantiating $H$:

$$H v = \nabla_\theta \left( \left(\nabla_\theta \bar{D}_{\text{KL}}\right)^\top v \right)$$

Typically, $10\text{ to }20$ CG iterations provide an accurate estimate of $x \approx H^{-1} g$.

### 2. Backtracking Line Search
Because Taylor approximations are only locally valid, taking the full step $\Delta \theta$ might violate the KL constraint or fail to improve the surrogate objective. TRPO verifies the proposed step using a backtracking line search:

$$\theta_{k+1} = \theta_k + \alpha^j \Delta \theta, \quad j \in \{0, 1, 2, \ldots\}$$

where decay factor $\alpha \in (0, 1)$. The search accepts the smallest $j$ that satisfies both:
1. Improvement in the surrogate objective: $L(\theta_{k+1}) > L(\theta_k)$
2. Satisfaction of the trust region constraint: $\bar{D}_{\text{KL}}(\theta_k, \theta_{k+1}) \le \delta$

---

## TRPO vs. PPO Comparison

| Aspect | TRPO | PPO |
| :--- | :--- | :--- |
| **Optimization Order** | Second-order (Fisher Information Matrix approximation) | First-order (Standard SGD / Adam) |
| **Constraint Enforcement** | Hard constraint solved via Conjugate Gradient & line search | Soft clipping of probability ratio $r_t(\theta)$ |
| **Computational Overhead** | High (multiple Hessian-vector product passes per batch) | Low (standard backward pass) |
| **Implementation Complexity**| Complex ($\sim 500+$ lines of specialized solver code) | Simple (standard loss function addition) |
| **Compatibility** | Struggles with recurrent policies (RNNs/LSTMs) or massive models | Seamlessly scales to multi-billion parameter LLMs |
