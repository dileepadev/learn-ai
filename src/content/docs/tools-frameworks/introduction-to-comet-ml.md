---
title: Introduction to Comet ML
description: An overview of Comet ML — an experiment tracking and model monitoring platform — covering experiment logging, model registry, and comparison across training runs.
---

Comet ML is a machine learning experiment tracking platform that logs metrics, hyperparameters, code state, and artifacts for every training run, letting teams compare experiments systematically instead of relying on scattered notebooks and spreadsheets.

## Core Concepts

### Experiments
An **experiment** is a single tracked training run. Comet automatically captures the metrics you log, along with the git commit, installed dependencies, and system information, so a past run can be reproduced or audited later.

### Projects
Experiments are grouped into **projects**, which correspond to a specific model or task being developed. Within a project, Comet provides a comparison view across all logged experiments.

### Panels
**Panels** are customizable visualizations (line charts, parallel coordinate plots, confusion matrices) that can be arranged into dashboards for monitoring training progress or comparing runs side by side.

## Getting Started

### Install and Configure

```bash
pip install comet_ml
export COMET_API_KEY="your-api-key"
```

### Logging an Experiment

```python
from comet_ml import Experiment

experiment = Experiment(project_name="image-classifier")

experiment.log_parameter("learning_rate", 0.001)
experiment.log_parameter("batch_size", 64)

for epoch in range(epochs):
    train_loss = train_one_epoch(model, data)
    experiment.log_metric("train_loss", train_loss, step=epoch)

experiment.log_model("final-model", "model.pt")
experiment.end()
```

## Comparing Experiments

Comet's comparison view overlays metrics from multiple experiments on the same chart, making it straightforward to see which hyperparameter combination produced the best validation performance:

```text
experiment A: lr=0.001, batch=64  -> val_acc=0.91
experiment B: lr=0.0005, batch=32 -> val_acc=0.93
experiment C: lr=0.001, batch=128 -> val_acc=0.89
```

Parallel coordinate plots visualize how multiple hyperparameters jointly relate to the target metric across many runs at once, which is harder to see from a flat table.

## Model Registry

Comet includes a **model registry** that versions trained models, tracks which experiment produced each version, and records the stage a model is in (staging, production, archived). This gives a clear audit trail connecting a deployed model back to the exact training run, code, and data that produced it.

## Model Monitoring

Beyond training, Comet can track live production model performance — logging prediction distributions and drift metrics over time so degradation can be caught before it significantly affects downstream decisions.

## Comet ML vs. Other Tracking Tools

| Feature | Comet ML | MLflow | Weights & Biases |
|---------|----------|--------|-------------------|
| Managed cloud offering | ✓ | Limited | ✓ |
| Self-hostable | ✓ | ✓ | Limited |
| Built-in model registry | ✓ | ✓ | ✓ |
| Production monitoring | ✓ | Limited | Partial |
| Parallel coordinate plots | ✓ | Limited | ✓ |

## Common Use Cases

- **Hyperparameter comparison:** tracking many runs to find the best-performing configuration.
- **Reproducibility:** recreating a past experiment's exact code, data, and environment.
- **Team collaboration:** sharing experiment dashboards across a team without manually exporting charts.
- **Production monitoring:** tracking a deployed model's live performance against its training-time baseline.

Comet ML's combination of experiment tracking, model registry, and production monitoring makes it a full-lifecycle tool for teams that need traceability from initial training run through deployed model.
