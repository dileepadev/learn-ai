---
title: Causal Discovery
description: A comprehensive guide to causal discovery algorithms that learn causal structure from observational data, covering constraint-based, score-based, and functional causal model approaches.
---

# Causal Discovery

Causal discovery is the task of **learning causal structure from data** — inferring which variables causally influence others, without requiring randomized experiments. Unlike causal inference (which estimates effect sizes given a known graph), causal discovery recovers the graph itself. It underpins explainable AI, scientific hypothesis generation, and robust decision-making under distribution shift.

## Why Causal Discovery Matters

Observational datasets capture correlations, but correlation does not imply causation. Consider:

- $X \to Y$: X causes Y (drug causes recovery)
- $Y \to X$: Y causes X (recovery improves drug compliance)
- $X \leftarrow Z \rightarrow Y$: common cause Z confounds (genetics affects both)

Standard ML cannot distinguish these from correlation alone. Causal discovery uses additional assumptions — faithfulness, Markov condition, functional causal models — to resolve the ambiguity.

## Core Assumptions

### Causal Markov Condition

A variable $X_i$ is independent of its non-descendants given its parents $\text{Pa}(X_i)$ in the causal graph $\mathcal{G}$:

$$X_i \perp \!\!\! \perp \text{Non-descendants}(X_i) \mid \text{Pa}(X_i)$$

### Faithfulness

Every conditional independence in the data is entailed by the graph — no independence arises by coincidental parameter cancellation.

### Causal Sufficiency

No unmeasured common causes (confounders) exist among observed variables.

## Markov Equivalence Classes

Many graphs produce identical conditional independencies and are **Markov equivalent** — indistinguishable from observational data alone. The set of equivalent graphs forms a **Markov Equivalence Class (MEC)**, represented by a **Completed Partially Directed Acyclic Graph (CPDAG)**.

Example: $A \to B \to C$, $A \leftarrow B \to C$, and $A \to B \leftarrow C$ all have the same skeleton but different v-structures — only $A \to B \leftarrow C$ (a collider) can be identified.

## Constraint-Based Methods

### PC Algorithm

The **PC algorithm** (Peter & Clark) learns the CPDAG using conditional independence tests.

**Phase 1 — Skeleton learning:**

1. Start with a complete undirected graph
2. For each pair $(X, Y)$, test $X \perp \!\!\! \perp Y \mid \mathbf{Z}$ for conditioning sets $\mathbf{Z}$ of increasing size
3. Remove edge if independence found; record separating set $\text{Sep}(X, Y) = \mathbf{Z}$

**Phase 2 — V-structure orientation:**

For each unshielded triple $X - Z - Y$ (no edge $X$–$Y$): if $Z \notin \text{Sep}(X, Y)$, orient as $X \to Z \leftarrow Y$.

**Phase 3 — Meek rules:** propagate orientations via four deterministic rules to avoid new v-structures or cycles.

```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

# data: (n_samples, n_variables) numpy array
cg = pc(data, alpha=0.05, indep_test=fisherz)
cg.draw_pydot_graph()
```

**Complexity:** $O(p^k)$ where $p$ = number of variables, $k$ = max conditioning set size.

### FCI — Fast Causal Inference

Extends PC to handle **latent confounders** (relaxes causal sufficiency). Outputs a **Partial Ancestral Graph (PAG)** with circle marks indicating uncertainty about edge orientation.

```python
from causallearn.search.ConstraintBased.FCI import fci

g, edges = fci(data, fisherz, 0.05)
```

## Score-Based Methods

Score-based methods assign a goodness-of-fit score to each DAG and search for the highest-scoring structure.

### BIC Score

$$\text{BIC}(\mathcal{G}, \mathcal{D}) = \log P(\mathcal{D} \mid \hat{\theta}_{\mathcal{G}}) - \frac{d}{2}\log n$$

where $d$ is the number of free parameters and $n$ is sample size. BIC penalizes complexity to avoid overfitting.

### GES — Greedy Equivalence Search

GES operates in the CPDAG space with two phases:

1. **Forward phase**: greedily add edges that maximally increase score
2. **Backward phase**: greedily remove edges

GES is **consistent** — recovers the true CPDAG as $n \to \infty$ under faithfulness.

```python
from causallearn.search.ScoreBased.GES import ges

Record = ges(data, score_func="local_score_BIC")
G = Record["G"]
```

### NOTEARS — Continuous Optimization

Zheng et al. (2018) reformulated DAG learning as a **continuous optimization problem** by encoding the acyclicity constraint:

$$\min_{W} \frac{1}{n}\|X - XW\|_F^2 + \lambda\|W\|_1 \quad \text{s.t.} \quad h(W) = 0$$

where $h(W) = \text{tr}(e^{W \circ W}) - d = 0$ is a smooth acyclicity constraint.

```python
import numpy as np
from notears import notears_linear

W_est = notears_linear(data, lambda1=0.1, loss_type="l2")
# W_est[i, j] != 0 implies X_i → X_j
```

This enables gradient-based DAG learning, opening the door to neural extensions.

### DAG-GNN and GRAN

Neural extensions of NOTEARS replace the linear structural equation with a graph neural network, enabling nonlinear causal discovery.

## Functional Causal Model Methods

FCM methods exploit **asymmetries** in the joint distribution to distinguish cause from effect — impossible with constraint-based methods in the Gaussian linear case.

### LiNGAM — Linear Non-Gaussian Acyclic Model

If noise terms are **non-Gaussian**, the full DAG is identifiable (not just the CPDAG):

$$\mathbf{X} = B\mathbf{X} + \boldsymbol{\varepsilon}, \qquad \varepsilon_i \sim \text{non-Gaussian}$$

ICA decomposition recovers $B$ uniquely.

```python
from lingam import DirectLiNGAM

model = DirectLiNGAM()
model.fit(data)
print(model.adjacency_matrix_)  # (p, p) — B[i,j] is effect of j on i
```

### ANM — Additive Noise Models

For bivariate causal discovery, fits $Y = f(X) + \varepsilon$ and $X = g(Y) + \eta$, checking which direction has **independent residuals**:

$$X \to Y \iff \hat{\varepsilon} = Y - \hat{f}(X) \perp \!\!\! \perp X$$

```python
from causallearn.search.FCMBased.ANM.ANM import ANM

anm = ANM()
p_value_xy, p_value_yx = anm.cause_or_effect(x, y)
direction = "X→Y" if p_value_xy > p_value_yx else "Y→X"
```

### RESIT — Regression with Subsequent Independence Test

Extends ANM to multivariate settings by regressing each variable on all others and testing residual independence.

## Evaluation Metrics

| Metric | Description |
|---|---|
| SHD (Structural Hamming Distance) | Edge insertions + deletions + reversals needed |
| Precision / Recall | Over directed/undirected edges |
| F1 score | Harmonic mean of precision and recall |
| SID (Structural Intervention Distance) | Error in interventional distributions |

```python
from causallearn.utils.GraphUtils import GraphUtils

# Compare estimated vs true adjacency matrices
shd = GraphUtils.structural_hamming_distance(est_graph, true_graph)
```

## Handling Real-World Challenges

### Non-Stationarity

**CD-NOD** (Causal Discovery from Non-stationary Data) models distribution shift as an observed domain variable, identifying edges that are stable across environments.

### Hidden Variables

**RFCI** (Really Fast Causal Inference) is a computationally efficient alternative to FCI that outputs a PAG under latent confounding with fewer independence tests.

### Time Series

**Granger causality** is the classical approach: $X$ Granger-causes $Y$ if past values of $X$ improve prediction of $Y$ beyond past $Y$ alone.

**PCMCI** (Runge et al.) extends PC to temporal data:

$$P(\mathbf{X}_t \mid \mathbf{X}_{t-1}, \mathbf{X}_{t-2}, \ldots) = \prod_i P(X_{i,t} \mid \text{Pa}(X_{i,t}))$$

```python
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)
```

## Software Ecosystem

| Library | Methods | Language |
|---|---|---|
| `causal-learn` | PC, FCI, GES, ANM, LiNGAM | Python |
| `lingam` | DirectLiNGAM, ICALiNGAM | Python |
| `notears` | NOTEARS linear/nonlinear | Python |
| `tigramite` | PCMCI, PCMCI+ | Python |
| `gCastle` | 20+ algorithms | Python |
| `bnlearn` | BN structure learning | R / Python |

## Applications in Machine Learning

- **Domain generalization**: discovering invariant causal features across environments
- **Feature selection**: including only causal predictors improves out-of-distribution robustness
- **Reinforcement learning**: causal world models for faster credit assignment
- **Fairness**: identifying causal pathways from protected attributes to outcomes
- **Gene regulatory networks**: inferring which genes regulate others from expression data

## Identifiability Summary

| Setting | Identifiable | Algorithm |
|---|---|---|
| Gaussian linear, no latents | CPDAG only | PC, GES |
| Non-Gaussian linear, no latents | Full DAG | LiNGAM |
| Nonlinear additive noise, no latents | Full DAG | ANM, RESIT |
| Any distribution, with latents | PAG only | FCI, RFCI |
| Non-stationary, no latents | Full DAG | CD-NOD |

## Summary

Causal discovery bridges observational data and causal structure, enabling machines to reason about interventions and counterfactuals. Constraint-based methods (PC, FCI) use conditional independence tests; score-based methods (GES, NOTEARS) optimize graph scores; functional causal models (LiNGAM, ANM) exploit distributional asymmetries. Real-world applications span genomics, time series analysis, fairness-aware ML, and reinforcement learning — all requiring knowledge of causal structure that cannot be read off from correlation alone.
