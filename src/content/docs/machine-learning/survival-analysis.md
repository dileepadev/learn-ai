---
title: Survival Analysis with Machine Learning
description: Learn how machine learning extends classical survival analysis — combining Cox proportional hazards, survival forests, and deep survival models to predict time-to-event outcomes in healthcare, finance, and engineering.
---

**Survival analysis** is a branch of statistics focused on modeling **time-to-event** data — the time until a specific event of interest occurs. Despite the name, the "event" need not be death: it could be equipment failure, customer churn, loan default, disease recurrence, employee resignation, or any other transition from one state to another. Machine learning has significantly extended the classical survival analysis toolkit, enabling more flexible models that capture complex relationships between covariates and event timing.

## Core Concepts

### The Survival Function

Given a random variable $T$ representing the time until an event, the **survival function** $S(t)$ gives the probability that the event has not yet occurred by time $t$:

$$S(t) = P(T > t)$$

$S(0) = 1$ (the event hasn't happened at the start) and $S(t)$ is monotonically decreasing toward 0 as $t \to \infty$.

### The Hazard Function

The **hazard function** $h(t)$ captures the instantaneous rate of event occurrence at time $t$, conditional on surviving until $t$:

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t \mid T \geq t)}{\Delta t}$$

Intuitively, $h(t)$ answers: "given that the subject has survived until time $t$, what is the instantaneous risk of the event occurring right now?" The hazard function can be constant (exponential distribution), increasing (equipment wearing out), decreasing (infant mortality period), or bathtub-shaped (both high early risk and high late risk).

The survival function and hazard function are related by:

$$S(t) = \exp\left(-\int_0^t h(u)\, du\right) = \exp(-H(t))$$

where $H(t) = \int_0^t h(u)\, du$ is the **cumulative hazard function**.

### Censoring

A defining challenge of survival analysis is **censoring** — situations where the exact event time is unknown:

- **Right censoring** (most common): The study ends or the subject is lost to follow-up before the event occurs. We know the event time is greater than the observed follow-up time, but we don't know by how much.
- **Left censoring**: The event occurred before the observation period began.
- **Interval censoring**: We know the event occurred within a time interval but not the exact time.

Standard regression models cannot handle censored observations correctly — they would either exclude the censored cases (introducing bias) or treat the censoring time as the event time (also biased). Survival analysis methods are specifically designed to handle censoring through the **partial likelihood** function.

## Classical Survival Models

### Kaplan-Meier Estimator

The **Kaplan-Meier (KM) estimator** is a nonparametric estimate of the survival function from observed data. It is the survival analysis equivalent of an empirical distribution function, incorporating censored observations:

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

where $d_i$ is the number of events at time $t_i$ and $n_i$ is the number of subjects at risk just before $t_i$.

The KM curve is the standard first visualization in any survival analysis — plotted separately for groups (e.g., treatment vs. control) to compare survival experience. The **log-rank test** assesses whether KM curves from two or more groups differ significantly.

### Cox Proportional Hazards Model

The **Cox proportional hazards model** (Cox, 1972) is the workhorse of survival regression. It models the hazard for subject $i$ with covariate vector $\mathbf{x}_i$ as:

$$h(t \mid \mathbf{x}_i) = h_0(t) \cdot \exp(\boldsymbol{\beta}^\top \mathbf{x}_i)$$

The **baseline hazard** $h_0(t)$ is an unspecified function of time shared by all subjects. The exponential term captures individual risk — subjects with higher $\boldsymbol{\beta}^\top \mathbf{x}_i$ have proportionally higher hazard at all times.

The key **proportional hazards assumption**: the ratio of hazards between any two subjects is constant over time. This is testable using Schoenfeld residuals.

Cox models are fit using the **partial likelihood** — which conditions out the baseline hazard, enabling estimation of $\boldsymbol{\beta}$ without specifying $h_0(t)$. This is Cox's key insight: you can estimate covariate effects without parametric assumptions about the shape of the hazard.

## Machine Learning Extensions

### Survival Forests

**Random Survival Forests (RSF)** (Ishwaran & Kogalur, 2008) extend random forests to survival outcomes:

- Trees are built by splitting on covariates to maximize the difference in survival curves between child nodes (using the log-rank test as the splitting criterion).
- Terminal node predictions are ensemble survival curves — averaging the Nelson-Aalen estimators across all trees.
- Feature importance is computed from the out-of-bag concordance index after permuting each feature.

RSF makes no proportional hazards assumption, handles high-dimensional feature spaces, and captures nonlinear interactions naturally. The **concordance index (C-index)** — the probability that the model assigns higher risk to the subject who experiences the event first in a randomly chosen pair — is the primary evaluation metric.

```python
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
import numpy as np

# Survival labels: array of (event_indicator, event_time) tuples
y_train = np.array([(True, 365), (False, 180), (True, 720), ...],
                   dtype=[('event', bool), ('time', float)])

model = RandomSurvivalForest(
    n_estimators=300,
    min_samples_leaf=10,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

# Predict survival function for new subjects
survival_fns = model.predict_survival_function(X_test)
for fn in survival_fns[:3]:
    print(fn.x, fn.y)  # time points, survival probabilities

# C-index
c_index = model.score(X_test, y_test)
```

### Gradient Boosting for Survival

**Gradient boosting survival models** treat the survival problem as minimizing a loss function derived from the Cox partial likelihood:

- **CoxBoost**: Gradient boosted Cox model — iteratively fits regression trees to the negative partial likelihood gradient.
- **XGBoost / LightGBM with Cox objective**: Both support the Cox partial likelihood loss natively:

```python
import xgboost as xgb

# For Cox objective, labels are: positive = event occurred, negative = censored
# The absolute value is the event time
dtrain = xgb.DMatrix(X_train, label=y_cox_format)

params = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

model = xgb.train(params, dtrain, num_boost_round=500)
```

### DeepSurv and Neural Survival Models

**DeepSurv** (Katzman et al., 2018) replaces the linear predictor in the Cox model with a deep neural network:

$$h(t \mid \mathbf{x}) = h_0(t) \cdot \exp(f_\theta(\mathbf{x}))$$

where $f_\theta(\mathbf{x})$ is a feedforward neural network. DeepSurv retains the partial likelihood objective, enabling end-to-end training while capturing highly nonlinear covariate effects.

**DRSA (Deep Recurrent Survival Analysis)** uses RNNs to model longitudinal covariates — covariate values that change over time (e.g., repeated lab measurements in clinical studies). The recurrent structure captures temporal dependencies in covariate trajectories.

**SurvTRACE** and **Transformer-based survival models** apply self-attention to tabular survival data, capturing complex covariate interactions in high-dimensional settings.

### Discrete-Time Survival Models

When time can be discretized into intervals (years, quarters), survival can be framed as a **sequence of binary classification problems**:

- At each time interval, the model predicts the probability of the event occurring *in that interval*, conditional on surviving to that point.
- Standard classifiers (logistic regression, neural networks, gradient boosting) apply directly with standard binary cross-entropy loss.
- The survival curve is reconstructed as the product of one-minus-hazard estimates across intervals.

This approach scales naturally to large datasets, leverages any binary classification model, and handles time-varying covariates straightforwardly.

## Competing Risks

In many real applications, subjects can experience one of **multiple mutually exclusive events** — and experiencing one event precludes the others. Examples:

- A patient can die from cancer (event 1), die from cardiovascular disease (event 2), or be administratively censored.
- A customer can churn to Competitor A (event 1), churn to Competitor B (event 2), or remain active.
- A machine can fail due to bearing wear (event 1) or electrical fault (event 2).

The **cause-specific hazard** approach models a separate Cox (or ML) model for each event type, treating the other event types as censored. The **Fine-Gray subdistribution hazard** model directly models the cumulative incidence function for each event type, accounting for the fact that a subject who has experienced a competing event cannot experience the focal event.

## Time-Varying Covariates

When covariate values change during follow-up (blood pressure measured at each clinic visit, loan payment behavior, machine sensor readings), the Cox model extends to handle them via the **counting process** formulation:

Each subject contributes multiple rows, one per time interval, with covariate values current at that interval. This enables incorporating longitudinal data without requiring a separate recurrent model.

## Applications

### Healthcare

- **Patient prognosis**: Predicting time to disease progression, death, or hospital readmission from clinical covariates and biomarkers.
- **Clinical trial analysis**: Primary endpoint analysis (time to event) for randomized controlled trials.
- **Cancer survival**: Predicting 5-year survival probability from tumor characteristics, staging, and treatment.
- **Readmission risk**: Predicting time to hospital readmission to prioritize post-discharge interventions.

### Finance and Insurance

- **Credit risk**: Predicting time to default for loans and mortgages — capturing that default risk varies over the loan's life.
- **Customer churn**: Modeling time to subscription cancellation as a survival problem, enabling cohort-level survival curves by customer segment.
- **Insurance claims**: Modeling time between insurance claims for frequency-severity models.

### Engineering and Reliability

- **Predictive maintenance**: Predicting remaining useful life (RUL) of mechanical components from sensor data — framed as a survival problem with "failure" as the event.
- **Product reliability**: Estimating warranty claim rates and product lifetimes from accelerated testing data.

## Evaluation Metrics

- **Concordance index (C-index)**: Probability that the model assigns higher risk to the subject who experiences the event first in a randomly chosen concordant pair. 0.5 = random, 1.0 = perfect. The survival equivalent of AUC.
- **Integrated Brier Score (IBS)**: Measures the mean squared error of survival probability predictions across all time points — lower is better. Accounts for calibration as well as discrimination.
- **Time-dependent AUC**: AUC computed at each time point $t$ — assessing how well the model distinguishes subjects who experience the event by $t$ from those who don't.

Survival analysis provides the principled statistical framework for time-to-event questions that are ubiquitous in medicine, business, and engineering — and ML extensions have dramatically expanded its modeling capacity beyond what traditional parametric and semiparametric models can achieve.
