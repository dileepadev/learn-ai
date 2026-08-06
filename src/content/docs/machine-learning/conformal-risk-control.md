---
title: "Conformal Risk Control: Statistical Guarantees Beyond Classification"
description: Learn how Conformal Risk Control extends conformal prediction to arbitrary loss functions, delivering rigorous statistical error bounds for complex ML and LLM outputs.
---

Machine learning models deployed in high-stakes domains — such as medical diagnosis, autonomous vehicle perception, financial risk assessment, and legal LLM generation — must provide reliable uncertainty estimates.

While standard **Conformal Prediction (CP)** provides distribution-free, finite-sample coverage guarantees for classification (e.g., *"the true label lies within this set of classes with 95% probability"*), traditional CP cannot handle arbitrary, non-bounded continuous loss functions or complex structured outputs like bounding boxes, image segmentation, and LLM text generation.

**Conformal Risk Control (CRC)**, developed by Angelopoulos et al. (2023), extends conformal prediction to guarantee that the **expected loss (risk)** of a model's prediction set or output stays below a user-specified threshold $\alpha$.

## The Core Concept: From Coverage to Risk Control

In standard conformal classification:
$$\mathbb{P}(Y \in C(X)) \ge 1 - \alpha$$
The goal is controlling the probability that the true label $Y$ is included in prediction set $C(X)$.

In **Conformal Risk Control**:
We define any bounded or monotonic loss function $\ell(C(X), Y) \in (-\infty, B]$ that quantifies prediction error (e.g., false negative rate, token hallucination rate, or bounding box intersection-over-union error).

CRC algorithms guarantee that the expected risk over unseen test samples is upper-bounded by a target level $\alpha$:
$$\mathbb{E}[\ell(C_\lambda(X), Y)] \le \alpha$$

where $C_\lambda(X)$ is a set-valued or thresholded predictor parameterized by a tuning parameter $\lambda$.

## How Split Conformal Risk Control Works

Given $n$ calibration data points $(X_1, Y_1), \dots, (X_n, Y_n)$ drawn i.i.d. from an unknown distribution:

1. **Parameterize Predictor Set:** Define a family of prediction sets $C_\lambda(x)$ indexed by a parameter $\lambda \in \mathbb{R}$. Larger values of $\lambda$ make the set $C_\lambda(x)$ larger and reduce the loss $\ell(C_\lambda(X), Y)$.
2. **Compute Empirical Risk:** For any candidate value of $\lambda$, compute the average calibration loss:
   $$\hat{R}_n(\lambda) = \frac{1}{n} \sum_{i=1}^n \ell(C_\lambda(X_i), Y_i)$$
3. **Upper Confidence Bound Selection:** Find the smallest $\hat{\lambda}$ that satisfies a statistical upper bound condition:
   $$\hat{\lambda} = \inf \left\{ \lambda : \frac{n}{n+1} \hat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha \right\}$$
4. **Deploy at Test Time:** For any new test input $X_{n+1}$, output $C_{\hat{\lambda}}(X_{n+1})$. The expected test loss is strictly guaranteed to satisfy $\mathbb{E}[\ell(C_{\hat{\lambda}}(X_{n+1}), Y_{n+1})] \le \alpha$.

## Applications of Conformal Risk Control

### 1. Controlling LLM Hallucination Rates
- **Problem:** LLMs generate plausible but unverified facts.
- **CRC Formulation:** Let $C_\lambda(X)$ be the set of generated claims retained after dropping claims with confidence scores below threshold $\lambda$. Let $\ell(C_\lambda(X), Y)$ be the fraction of unsupported claims in the generation.
- **Guarantee:** CRC guarantees that the average hallucination rate across generated responses stays strictly below $\alpha = 5\%$.

### 2. Multi-Label Classification & Gene Function Annotation
- **Problem:** Predicting multi-label tags where missing a true label (False Negative) is far worse than adding a spurious tag (False Positive).
- **CRC Formulation:** Use False Negative Rate (FNR) as the loss function $\ell(C_\lambda(X), Y) = \frac{|Y \setminus C_\lambda(X)|}{|Y|}$.
- **Guarantee:** CRC sets $\lambda$ so that average FNR is guaranteed to be under 2%.

### 3. Object Detection Bounding Box Tightness
- **Problem:** Ensuring object detection bounding boxes encompass targets with guaranteed coverage.

## Step-by-Step Python Implementation Example

```python
import numpy as np

def split_conformal_risk_control(calib_losses_by_lambda: np.ndarray, lambdas: np.ndarray, alpha: float, B: float = 1.0):
    """
    Computes optimal threshold lambda_hat guaranteeing expected loss <= alpha.
    
    Parameters:
    - calib_losses_by_lambda: shape (n_samples, n_lambdas) loss values for each sample across lambdas
    - lambdas: array of candidate lambda parameter values (sorted)
    - alpha: targeted maximum risk bound (e.g. 0.05 for 5% risk limit)
    - B: upper bound of the loss function
    """
    n = calib_losses_by_lambda.shape[0]
    
    # Compute empirical mean loss for each candidate lambda across calibration set
    empirical_risks = np.mean(calib_losses_by_lambda, axis=0)
    
    # Apply Benton-Hoeffding / CRC upper bound formula
    risk_upper_bounds = (n / (n + 1)) * empirical_risks + (B / (n + 1))
    
    # Find smallest lambda satisfying the risk condition
    valid_indices = np.where(risk_upper_bounds <= alpha)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("Target risk alpha is too strict for the provided calibration set.")
        
    optimal_idx = valid_indices[0]
    return lambdas[optimal_idx], empirical_risks[optimal_idx]

# Example Usage:
n_calib = 1000
n_lambdas = 100
candidate_lambdas = np.linspace(0.01, 0.99, n_lambdas)

# Simulate calibration losses (higher lambda -> lower loss)
simulated_losses = np.random.beta(0.5, 2.0, size=(n_calib, n_lambdas)) * (1.0 - candidate_lambdas)

target_alpha = 0.05  # Target maximum expected risk = 5%
hat_lambda, empirical_risk = split_conformal_risk_control(simulated_losses, candidate_lambdas, alpha=target_alpha)

print(f"Selected Lambda Threshold: {hat_lambda:.4f}")
print(f"Empirical Calibration Risk: {empirical_risk:.4f} (Guaranteed <= {target_alpha})")
```

## Comparison: Conformal Prediction vs. Conformal Risk Control

| Attribute | Standard Conformal Prediction | Conformal Risk Control (CRC) |
|---|---|---|
| **Target Quantity** | Set Coverage Probability $\mathbb{P}(Y \in C(X)) \ge 1-\alpha$ | Expected Loss Bounds $\mathbb{E}[\ell(C(X), Y)] \le \alpha$ |
| **Loss Function** | Binary Indicator Loss ($1_{Y \notin C(X)}$) | Any Bounded / Monotonic Continuous Loss |
| **Output Types** | Classification Sets, Interval Regression | LLM Generation, Segmentation Masks, Multi-label, Graphs |
| **Distribution Assumptions** | Distribution-free (i.i.d. exchangeability) | Distribution-free (i.i.d. exchangeability) |

## Summary

Conformal Risk Control bridges machine learning deployment and statistical safety. By allowing practitioners to place rigorous upper bounds on arbitrary loss functions — from LLM hallucination rates to false negative metrics — CRC provides finite-sample distribution-free reliability for real-world AI applications.

## Further Reading

- Angelopoulos, Bates, Jordan, & Malik (2023), *Conformal Risk Control*
- Angelopoulos & Bates (2022), *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*
- Learn-Then-Test Framework for Multiple Hypothesis Risk Control
