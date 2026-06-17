---
title: ML Experiment Tracking
description: Why experiment tracking is essential for reproducible ML, what to track, how MLflow and Weights & Biases work, and best practices for managing experiments at scale.
---

ML experiment tracking is the practice of systematically recording everything about a training run — hyperparameters, metrics, artifacts, code versions, and environment — so that any experiment can be reproduced, compared, and understood after the fact.

Without experiment tracking, ML development degrades into "I ran a lot of experiments but can't remember which config produced the best model."

## What to Track

A complete experiment record captures four things:

### 1. Hyperparameters
Every configuration choice that affects training:
- Learning rate, batch size, optimizer, weight decay.
- Architecture choices: layer sizes, dropout rates, attention heads.
- Data preprocessing decisions: normalization, augmentation settings.
- Training duration: epochs, early stopping patience.

### 2. Metrics
Performance measurements over time:
- Training and validation loss per epoch/step.
- Task-specific metrics: accuracy, F1, BLEU, ROUGE, AUC, NDCG.
- System metrics: GPU memory, training throughput (steps/sec), wall time.

### 3. Artifacts
Files produced by the experiment:
- Model checkpoints (saved model weights).
- Training datasets and their versions.
- Evaluation results, confusion matrices, prediction samples.
- Plots and visualizations.

### 4. Code and Environment
What code and software stack produced the result:
- Git commit hash (ensures exact code reproducibility).
- Python and library versions (requirements.txt or conda env).
- Hardware configuration (GPU type, number).
- Random seeds.

## MLflow

MLflow is the dominant open-source experiment tracking framework. Its core abstraction is a **run**: a single execution of a training script with its own log of parameters, metrics, and artifacts.

```python
import mlflow

mlflow.set_experiment("text-classifier-v2")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        "learning_rate": 3e-4,
        "batch_size": 32,
        "model": "bert-base-uncased",
    })

    # Train model...
    for epoch in range(num_epochs):
        train_loss, val_loss, val_f1 = train_epoch(model, data)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }, step=epoch)

    # Log the final model
    mlflow.pytorch.log_model(model, "model")
```

MLflow stores runs locally or on a remote tracking server. The UI allows filtering, sorting, and comparing runs across any logged parameter or metric.

**MLflow also provides:**
- **Model Registry:** Version and stage-manage models (Staging, Production, Archived).
- **Projects:** Reproducible packaging of ML code.
- **Deployments:** Model serving integrations.

## Weights & Biases (W&B)

Weights & Biases is a commercial platform (with a free tier) offering richer collaboration features and deeper integrations. It logs everything MLflow does, plus:

- **System metrics** (GPU/CPU utilization, memory) logged automatically.
- **Media logging:** Images, audio, video, 3D point clouds, HTML, tables.
- **Hyperparameter sweeps:** Automated search (grid, random, Bayesian) with the W&B `sweep` API.
- **Artifacts with lineage:** Tracks how datasets, models, and downstream artifacts relate to each other.
- **Reports:** Interactive shareable documents mixing text, charts, and experiment comparisons.

```python
import wandb

wandb.init(project="text-classifier", config={
    "learning_rate": 3e-4,
    "batch_size": 32,
})

for epoch in range(num_epochs):
    train_loss, val_f1 = train_epoch(model, data)
    wandb.log({"train_loss": train_loss, "val_f1": val_f1})

wandb.finish()
```

## Comparing Experiments

The primary workflow after tracking experiments:
1. Filter runs by metric threshold (e.g., "show all runs with val_f1 > 0.85").
2. Sort by the target metric to find top candidates.
3. Group by a key hyperparameter to understand its effect.
4. Plot metric curves to compare learning trajectories.
5. Identify which hyperparameters correlate most with performance (parallel coordinates plots, importance analysis).

## Best Practices

- **Track every run, not just the good ones.** Failed experiments contain information about what doesn't work.
- **Use descriptive run names and tags.** "lr-warmup-cosine-bert-large" is more useful than "run_47."
- **Log git commit hash automatically.** Many frameworks do this; don't rely on manual documentation.
- **Version your datasets.** Log a hash or DVC reference to the exact dataset used, not just a path.
- **Set random seeds and log them.** Reproducibility requires all sources of randomness to be fixed and recorded.
- **Log system metrics.** GPU OOM errors and training slowdowns are often more important to debug than model metrics.

## Choosing a Tool

| Criteria | MLflow | W&B | Neptune | Comet |
|----------|--------|-----|---------|-------|
| Open-source | ✓ | ✗ | ✗ | ✗ |
| Self-hostable | ✓ | ✓ (enterprise) | ✓ | ✗ |
| Free tier | ✓ | ✓ (limited) | ✓ | ✓ |
| Collaboration | Limited | Strong | Strong | Moderate |
| Sweeps/HPO | Limited | Built-in | Limited | Limited |

MLflow is the standard choice for teams that need self-hosting and model registry integration (e.g., with Databricks). W&B is preferred for teams prioritizing collaboration, visualization, and sweeps.
