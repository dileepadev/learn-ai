---
title: "ML Observability: Monitoring Models in Production"
description: "Learn how to monitor machine learning models in production — detecting data drift, model degradation, and silent failures before they impact users."
---

A model that works well at launch can silently degrade over time as the world changes. **ML observability** is the practice of instrumenting your ML systems so you can detect, diagnose, and respond to these changes before they cause significant harm.

## Why Models Degrade

### Data Drift
The statistical distribution of input features changes over time. A fraud detection model trained on 2022 transaction patterns may perform poorly on 2024 patterns as fraud tactics evolve.

### Concept Drift
The relationship between inputs and the correct output changes. A demand forecasting model trained before a major economic shift will have learned relationships that no longer hold.

### Upstream Data Changes
A feature pipeline changes its logic, a data source changes its schema, or a third-party API starts returning different values. The model receives inputs it was never trained on.

### Feedback Loop Effects
The model's predictions influence future data. A recommendation system that promotes certain content changes user behavior, which changes the distribution of future training data.

## The Three Pillars of ML Observability

### 1. Data Monitoring

Monitor the statistical properties of inputs and outputs:

- **Distribution monitoring**: Track mean, variance, percentiles, and histograms of each feature. Alert when they deviate significantly from training distribution.
- **Schema validation**: Detect missing features, type changes, unexpected null rates.
- **Statistical tests**: Population Stability Index (PSI), Kolmogorov-Smirnov test, Jensen-Shannon divergence for detecting distribution shift.

### 2. Model Performance Monitoring

Track prediction quality over time:

- **Labeled data**: If ground truth labels are available (even with delay), compute accuracy, F1, AUC, RMSE directly.
- **Proxy metrics**: Business metrics correlated with model quality (click-through rate, conversion rate, customer complaints).
- **Prediction distribution**: Monitor the distribution of model outputs. A sudden shift in prediction confidence or class distribution often signals a problem.

### 3. Infrastructure Monitoring

Standard software observability:
- Latency (P50, P95, P99).
- Error rates and exception types.
- Throughput and queue depth.
- GPU/CPU utilization and memory.

## Alerting Strategy

Not every drift requires immediate action. A tiered alerting strategy:

1. **Warning**: Drift detected, within acceptable bounds. Log and monitor.
2. **Alert**: Drift exceeds threshold. Investigate root cause.
3. **Critical**: Performance degradation confirmed. Trigger retraining or rollback.

Set thresholds based on business impact, not just statistical significance. A statistically significant drift that doesn't affect business metrics may not warrant action.

## Handling Delayed Labels

Ground truth labels are often delayed or unavailable:
- A loan default label may take months to materialize.
- User satisfaction may only be measurable through indirect signals.

Strategies:
- **Proxy labels**: Use faster-available signals as proxies.
- **Active labeling**: Sample predictions for human review.
- **Simulation**: Use historical data to estimate current performance.

## Tools for ML Observability

| Tool | Focus |
|---|---|
| Evidently AI | Open-source data and model monitoring |
| Arize Phoenix | LLM and ML observability |
| WhyLabs | Data monitoring and drift detection |
| Fiddler AI | Enterprise model monitoring |
| Grafana + Prometheus | Infrastructure metrics |
| MLflow | Experiment tracking + basic monitoring |

## LLM-Specific Observability

For LLM applications, additional monitoring dimensions:
- **Latency per token**: Track generation speed.
- **Token usage**: Monitor cost and context window utilization.
- **Hallucination rate**: Sample responses for factual accuracy.
- **Guardrail trigger rate**: How often safety filters activate.
- **User feedback signals**: Thumbs up/down, regeneration rate, session abandonment.

The goal of ML observability is not just detecting problems — it's building the feedback loops that make your models continuously improve.
