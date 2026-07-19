---
title: Introduction to Evidently AI
description: Get started with Evidently AI for data-quality checks, model monitoring, drift reports, and evaluation dashboards.
---

Evidently AI is an open-source Python library for evaluating and monitoring machine-learning and AI systems. It compares current data or predictions to a reference period and produces reports, test suites, and metrics that teams can use in notebooks, CI, or monitoring pipelines.

## The Monitoring Question

Monitoring begins with a concrete question:

- Has the input distribution changed?
- Are predictions becoming less reliable?
- Is a feature missing or outside its expected range?
- Does quality differ across important cohorts?

A drift score by itself is not an incident. It is a signal to investigate whether a change affects the model and users.

## Basic Report

```python
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report([DataDriftPreset()])
snapshot = report.run(
    reference_data=reference_df,
    current_data=current_df,
)

snapshot.save_html("data_drift.html")
```

The reference dataset should represent a known-good period, not merely the latest batch. Choose it deliberately and version its schema and selection criteria.

## Useful Checks

Evidently can surface missing values, duplicate rows, feature distributions, prediction distributions, classification quality when labels arrive, and LLM-oriented metrics. Add thresholds appropriate to each feature and model; a small shift in an ID field is different from a small shift in a safety-critical input.

## Production Pattern

1. log inputs, outputs, versions, and eventual outcomes with privacy controls
2. compute checks on a schedule or for each batch
3. send actionable alerts to an owner
4. investigate with slices and examples
5. retrain, roll back, or update thresholds only with evidence

Do not log sensitive prompts or personal data by default. Monitoring is effective when it connects technical signals to an explicit response process.

