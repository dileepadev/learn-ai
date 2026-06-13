---
title: Introduction to Weights & Biases (W&B)
description: A practical introduction to Weights & Biases, the MLOps platform for experiment tracking, model visualization, and collaboration.
---

Weights & Biases (W&B, pronounced "wand-b") is a popular MLOps platform that helps machine learning practitioners track experiments, visualize training, manage datasets, and collaborate on models. It integrates with virtually every major ML framework and is widely used in research and production.

## What W&B Does

At its core, W&B solves the reproducibility and visibility problem in ML experimentation. When training models, you typically want to track:

- **Hyperparameters** (learning rate, batch size, architecture choices)
- **Metrics** (loss, accuracy, F1 over time)
- **Model artifacts** (saved checkpoints)
- **System metrics** (GPU usage, memory)
- **Media** (images, audio, plots, tables)

W&B logs all of this automatically and stores it in a centralized, searchable dashboard.

## Core Features

### Experiment Tracking (Runs)
Every training run is logged as a **Run**. You initialize a run, log metrics as training progresses, and finish when done. Runs are grouped into **Projects**.

```python
import wandb

wandb.init(project="my-project", config={"lr": 0.001, "epochs": 10})

for epoch in range(10):
    loss = train_one_epoch()
    wandb.log({"loss": loss, "epoch": epoch})

wandb.finish()
```

### Hyperparameter Sweeps
W&B Sweeps automates hyperparameter search. Define the search space in a config, and W&B coordinates multiple agents to explore it, logging all results centrally.

```yaml
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
```

### Artifacts
Artifacts track and version datasets, model checkpoints, and any file. This creates a lineage graph showing which model was trained on which data.

### W&B Tables
Interactive tables for logging predictions, ground truths, and rich media (images, text) side by side. Excellent for debugging model outputs.

### Reports
Shareable, collaborative documentation that embeds live charts and narrative text — like a Jupyter notebook for ML results.

## Framework Integrations

W&B integrates natively with:
- **PyTorch** (`wandb.watch(model)` logs gradients and weights)
- **Keras / TensorFlow** (WandbCallback)
- **Hugging Face Transformers** (`report_to="wandb"` in TrainingArguments)
- **scikit-learn, XGBoost, LightGBM, PyTorch Lightning**

Most integrations require just one or two lines of code.

## Getting Started

```bash
pip install wandb
wandb login
```

Then add `wandb.init()` and `wandb.log()` to your training script. Everything is automatically synced to your W&B dashboard at wandb.ai.

## When to Use W&B

- When you're running multiple experiments and need to compare them.
- When you want to reproduce a previous run's configuration.
- When collaborating with a team on model development.
- When you need to track dataset versions alongside model versions.

W&B has a generous free tier suitable for individual researchers and small teams.
