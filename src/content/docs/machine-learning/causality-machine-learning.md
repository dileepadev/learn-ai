---
title: Causality in Machine Learning
description: Moving beyond correlation to causation — causal inference methods, causal discovery, and applications in ML and decision-making.
---

**Causality** in machine learning addresses a fundamental challenge: most AI systems learn correlations from data, but the real world often demands causal understanding. "Does X cause Y, or do they both result from a hidden confounder?"

Causal reasoning is essential for policy making (will this intervention improve outcomes?), scientific discovery, and robust machine learning systems.

## Correlation vs. Causation

### The Confounder Problem

Consider predicting hospital readmission risk. A model trained on historical data finds: "Wearing an oxygen mask increases readmission risk." This is not causal. The confounder is severity of illness: severe patients are more likely to wear oxygen masks *and* to be readmitted.

A naive policy ("remove oxygen masks to reduce readmission") would harm patients.

### Why This Matters

- **Fairness**: Predicting credit risk conditioned on race captures correlation, not causation — can perpetuate historical discrimination.
- **Robustness**: A model trained on correlations may fail when the environment changes (distribution shift).
- **Intervention**: "What if we changed X?" requires causal reasoning, not correlation.

## Causal Graphs

**Causal graphs** represent causal relationships using directed acyclic graphs (DAGs):
- **Nodes**: Variables.
- **Edges**: Direct causal effects (if X → Y, changing X directly affects Y).

### Example

```
Confounder (U)
    ↓     ↓
    X  →  Y
```

X causes Y, but both are influenced by U. Ignoring U leads to confounding bias.

### Markov Assumption

A key assumption: each variable is independent of non-descendants given its parents. This enables conditional independence reasoning.

## Causal Inference Methods

### Randomized Experiments

Gold standard: randomize the treatment (X) and observe outcomes (Y). Removes confounding:

$$P(Y | \text{do}(X = x)) = P(Y | X = x)$$

The $\text{do}$-operator represents an intervention, distinct from observation.

**Limitation**: Expensive, infeasible for many questions (e.g., "Does education cause higher income?").

### Observational Adjustment (Backdoor Criterion)

If we observe all confounders, we can eliminate bias by conditioning:

$$P(Y | \text{do}(X = x)) = \sum_u P(Y | X = x, U = u) P(U = u)$$

**Requirement**: We must observe all confounders $U$. This is often violated in practice (unmeasured confounding).

### Instrumental Variables

When unobserved confounders exist, use an **instrumental variable** $Z$:
- $Z$ affects $Y$ only through $X$ (no direct effect).
- $Z$ is independent of unobserved confounders.

Under these conditions, $Z$ enables causal effect estimation even with unobserved confounders.

**Example**: Geographic proximity to universities can instrument education (affects education, but affects income only through education, not through other factors).

### Difference-in-Differences

Exploit variation over time and across groups:

$$\text{CausalEffect} = (Y_{\text{treated, after}} - Y_{\text{treated, before}}) - (Y_{\text{control, after}} - Y_{\text{control, before}})$$

Assumes parallel trends: absent the treatment, both groups would evolve similarly.

Used in policy evaluation (e.g., did a new law affect employment?).

### Propensity Score Matching

When randomization is infeasible, match treated and control units with similar propensity scores (probability of treatment given observed covariates). Treated and control groups become more similar, reducing confounding bias.

## Causal Discovery

**Causal discovery** infers the causal graph from observational data — a harder problem than causal inference.

### Constraint-Based Methods

Test conditional independencies implied by the causal graph. For example, if $Z$ is a confounder of $X$ and $Y$, then $X \perp Y | Z$ (X and Y are independent given Z).

**Algorithm**: PC algorithm iteratively removes edges whose conditional independence is confirmed by data.

**Limitation**: Can only identify causal structure up to Markov equivalence (multiple DAGs consistent with observed independencies).

### Score-Based Methods

Search over possible DAGs, scoring each by how well it fits the data:

$$\text{Score}(\text{DAG}) = -\text{BIC}(\text{DAG}) = -k \log n + 2 \log \hat{L}(\text{DAG})$$

where $k$ is the number of parameters and $\hat{L}$ is the likelihood.

**Trade-off**: Search space is exponential in the number of variables.

### Functional Causal Models

Assume each variable is a function of its parents plus independent noise:

$$Y = f(X, U)$$

where $U$ is exogenous noise, independent of all other variables' parents. Under this model and smoothness assumptions, certain directions of causality are identifiable.

## Applications

### Personalized Medicine

Infer which treatment is best for a patient given their characteristics. Causal reasoning ensures treatment recommendations are not merely correlational.

### Recommendation Systems

Standard collaborative filtering learns correlations (users who liked item A also like item B). Causal approaches ask: "Will recommending this item cause engagement, or is engagement driven by factors independent of the recommendation?"

### Policy Optimization

A government program aims to increase employment. Causal inference estimates the causal effect; observational data alone could mislead due to selection bias (those already likely to find jobs self-select into the program).

### Fairness

Causal reasoning helps identify discrimination. If a model's decisions are causally affected by a protected attribute (race, gender), it's discriminatory. If the protected attribute affects decisions only through legitimate mediators (e.g., race affects income prediction only through education), it may be acceptable.

## Causal Machine Learning

**Causal ML** integrates causal reasoning into standard ML pipelines:

### Heterogeneous Treatment Effects

Estimate causal effects that vary by subgroup:

$$\tau(x) = \mathbb{E}[Y | \text{do}(X = 1), Z = x] - \mathbb{E}[Y | \text{do}(X = 0), Z = x]$$

**Methods**: Causal forests, X-learner, R-learner.

**Application**: Personalize interventions based on individual characteristics.

### Debiased ML

Correct for confounding bias in ML predictions using causal weighting:

$$\hat{\theta} = \arg \min_\theta \mathbb{E}[(Y - \theta X) / P(X | Z)]$$

where the weights $1 / P(X | Z)$ account for confounding through the propensity score.

## Challenges

**Unidentifiability**: Even with perfect data and large sample sizes, some causal effects cannot be identified. Observational data has fundamental limits.

**Model misspecification**: Causal inference is fragile to assumption violations (wrong causal graph, unobserved confounding). Sensitivity analysis assesses robustness to violations.

**Scalability**: Causal discovery in high-dimensional settings remains open; computational and sample complexity scale poorly.

**Interpretability-causality gap**: Causal graphs require domain knowledge; ML practitioners often lack such knowledge.

## Research Directions

- **Learning causal representations**: Deep learning approaches that discover causal factors directly from high-dimensional data.
- **Adaptive experimental design**: Iteratively choose treatments to maximize learning efficiency.
- **Causal transfer learning**: Transfer causal knowledge across domains.
- **Explainability through causality**: Using causal reasoning to explain model predictions.

Causality is becoming increasingly central to responsible AI development, especially for high-stakes domains where decisions affect people's lives.
