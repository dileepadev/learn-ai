---
title: Introduction to Weights & Biases
description: Learn how to use Weights & Biases (W&B) for experiment tracking, dataset and model artifact versioning, hyperparameter optimization with Sweeps, and model registry — the essential MLOps platform for machine learning teams.
---

**Weights & Biases (W&B)** is a machine learning platform designed to help researchers and engineers track experiments, visualize training metrics, version datasets and models, and optimize hyperparameters at scale. Originally launched in 2018 as an experiment tracker, W&B has evolved into a comprehensive MLOps platform used by organizations ranging from academic research labs to production AI teams at major technology companies.

The core value proposition of W&B is **reproducibility and collaboration**: every experiment run is logged with its hyperparameters, metrics, system utilization, and code state — creating a permanent, searchable record that enables teams to understand why a model performs the way it does and reproduce any result exactly.

## Core Concepts

### Runs

A **run** is the fundamental unit of W&B — a single execution of your training script or experiment. Each run automatically captures:

- **Hyperparameters**: Configuration values passed to the run (learning rate, batch size, model architecture, optimizer).
- **Metrics**: Any values you log during training (loss, accuracy, F1, BLEU, perplexity) at each step or epoch.
- **System metrics**: CPU/GPU utilization, memory usage, disk I/O, and network — automatically collected without instrumentation.
- **Code state**: The git commit hash, uncommitted changes (diff), and command used to launch the run.
- **Output files**: Model checkpoints, plots, and any other files you save during the run.

Runs are organized into **Projects** — logical groupings of related experiments (e.g., all experiments for a specific model architecture or dataset).

### Basic Integration

Integrating W&B into a training script requires minimal code:

```python
import wandb

# Initialize a run
wandb.init(
    project="my-image-classifier",
    config={
        "learning_rate": 1e-3,
        "batch_size": 64,
        "architecture": "resnet50",
        "epochs": 20,
    }
)

# Access config values
config = wandb.config

# Log metrics during training
for epoch in range(config.epochs):
    train_loss = train_one_epoch(model, train_loader, config.learning_rate)
    val_accuracy = evaluate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/accuracy": val_accuracy,
    })

# Save a model artifact
wandb.save("model_checkpoint.pt")

wandb.finish()
```

W&B provides framework-specific integrations for PyTorch, TensorFlow/Keras, Hugging Face Transformers, PyTorch Lightning, and others — enabling automatic logging with a single line of code.

## Experiment Tracking

### The W&B Dashboard

The W&B web dashboard provides a rich interface for analyzing experiments:

- **Run table**: Compare hyperparameters and final metrics across all runs in a project. Sort, filter, and group runs to identify which configurations perform best.
- **Charts and panels**: Interactive time-series charts of metrics across training steps. Multiple runs overlay on the same chart for direct comparison.
- **Parallel coordinates plot**: Visualizes the relationship between hyperparameters and metrics across many runs simultaneously — revealing which hyperparameter ranges consistently produce good results.
- **Scatter plots**: Any metric against any other metric or hyperparameter, helping identify correlations.

### Media Logging

Beyond scalar metrics, W&B logs rich media:

```python
# Log images (e.g., validation samples with predictions)
wandb.log({
    "val/samples": [
        wandb.Image(img, caption=f"Pred: {pred}, GT: {label}")
        for img, pred, label in zip(val_images, predictions, labels)
    ]
})

# Log audio
wandb.log({"generated_audio": wandb.Audio(audio_array, sample_rate=22050)})

# Log video
wandb.log({"rollout": wandb.Video(video_frames, fps=24)})

# Log matplotlib/plotly figures
fig = create_confusion_matrix(y_true, y_pred)
wandb.log({"confusion_matrix": wandb.Image(fig)})

# Log tables for structured data comparison
columns = ["image", "predicted", "actual", "confidence"]
data = [[wandb.Image(img), pred, label, conf] for ...]
wandb.log({"val/predictions": wandb.Table(columns=columns, data=data)})
```

### Alerts

W&B can send alerts when training metrics cross defined thresholds:

```python
# Alert if validation accuracy drops unexpectedly
if val_accuracy < best_val_accuracy - 0.05:
    wandb.alert(
        title="Validation accuracy drop detected",
        text=f"Accuracy fell from {best_val_accuracy:.3f} to {val_accuracy:.3f}",
        level=wandb.AlertLevel.WARN,
    )
```

## Artifacts: Dataset and Model Versioning

**Artifacts** are W&B's version control system for datasets, models, and other large files. Every artifact version is immutable and permanently tracked — creating a complete lineage from raw data to trained model.

### Creating and Logging Artifacts

```python
# Log a dataset artifact
with wandb.init(project="my-project", job_type="data-prep") as run:
    artifact = wandb.Artifact(
        name="imagenet-subset",
        type="dataset",
        description="ImageNet validation set, 10k samples",
        metadata={"num_samples": 10000, "split": "val"}
    )
    artifact.add_dir("./data/imagenet_val/")
    run.log_artifact(artifact)
```

```python
# Log a model artifact
with wandb.init(project="my-project", job_type="training") as run:
    # ... training code ...

    model_artifact = wandb.Artifact(
        name="resnet50-classifier",
        type="model",
        metadata={"val_accuracy": 0.923, "architecture": "resnet50"}
    )
    model_artifact.add_file("model.pt")
    run.log_artifact(model_artifact)
```

### Consuming Artifacts

```python
# Download and use a specific artifact version
with wandb.init(project="my-project", job_type="evaluation") as run:
    artifact = run.use_artifact("resnet50-classifier:v3")
    artifact_dir = artifact.download()
    model = load_model(f"{artifact_dir}/model.pt")
```

### Artifact Lineage

W&B automatically tracks the lineage between artifacts — knowing which dataset version was used to train which model version, and which model was used to generate which evaluation results. The **artifact graph** visualizes these relationships as a DAG, enabling complete reproducibility tracing.

## Sweeps: Hyperparameter Optimization

**W&B Sweeps** automates hyperparameter search with support for grid search, random search, and Bayesian optimization — parallelizing across multiple agents and visualizing results as they arrive.

### Defining a Sweep

```python
import wandb

sweep_config = {
    "method": "bayes",  # or "grid", "random"
    "metric": {
        "name": "val/accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2
        },
        "batch_size": {
            "values": [32, 64, 128, 256]
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5
        },
        "optimizer": {
            "values": ["adam", "sgd", "adamw"]
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 2
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")
```

### Running Sweep Agents

```python
def train():
    with wandb.init() as run:
        config = run.config

        model = build_model(config.dropout)
        optimizer = get_optimizer(config.optimizer, config.learning_rate)

        for epoch in range(20):
            train_loss = train_epoch(model, optimizer, config.batch_size)
            val_acc = validate(model)
            wandb.log({"val/accuracy": val_acc, "train/loss": train_loss})

# Launch multiple agents in parallel (on different machines/GPUs)
wandb.agent(sweep_id, function=train, count=50)
```

Bayesian optimization sweeps use a **Gaussian process** to model the relationship between hyperparameters and the target metric, intelligently selecting new hyperparameter combinations to evaluate based on previous results — converging to the optimum faster than random search.

### Early Termination

The **Hyperband** early termination policy in the example above automatically terminates poor-performing runs early, freeing compute for promising configurations. This dramatically increases the efficiency of large sweeps.

## Model Registry

The **W&B Model Registry** provides a centralized catalog for managing model versions across their lifecycle — from experimentation through production:

- **Model staging**: Models progress through stages (Candidate → Staging → Production → Archived) with tracked transitions and audit logs.
- **Linking runs to registry**: Trained model artifacts from experiment runs are linked to registry entries, maintaining traceability.
- **Metadata and evaluation results**: Each registry entry stores evaluation metrics, benchmark results, and usage documentation.
- **Downstream notifications**: Teams subscribed to a model can be notified when new versions are registered or promoted.

```python
# Register a model from an artifact
run = wandb.init(project="my-project")
artifact = run.use_artifact("resnet50-classifier:v5")

# Link to the model registry
run.link_artifact(
    artifact,
    target_path="my-org/model-registry/image-classifier"
)
```

## Reports

**W&B Reports** are collaborative documents that combine experiment charts, code, and narrative text — serving as living research documents:

- Embed live charts that update as new runs are logged.
- Document research findings, methodology, and conclusions.
- Share results with stakeholders who don't have ML expertise.
- Create experiment comparison reports for model review meetings.

Reports are increasingly used as **model cards** — documents accompanying model releases that describe training data, evaluation results, limitations, and intended use cases.

## Framework Integrations

### Hugging Face Transformers

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # Single line enables W&B logging
    run_name="bert-finetuning-experiment-1",
    # ... other training args
)
```

### PyTorch Lightning

```python
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(project="my-project", log_model="all")
trainer = Trainer(logger=logger, max_epochs=20)
```

### Keras/TensorFlow

```python
from wandb.keras import WandbCallback

model.fit(
    train_data,
    callbacks=[WandbCallback(monitor="val_accuracy")]
)
```

## Privacy and Self-Hosting

W&B offers **W&B Server** for organizations that require on-premises or private cloud deployment of the W&B platform — ensuring experiment data and model artifacts never leave the organization's infrastructure. This is particularly important for:

- Organizations with sensitive research data.
- Defense and intelligence applications.
- Healthcare organizations with HIPAA requirements.
- Financial services with regulatory data residency requirements.

## W&B in the MLOps Ecosystem

W&B occupies a central position in the MLOps toolchain:

- Integrates with **training infrastructure** (SLURM clusters, Kubernetes, AWS SageMaker, Google Vertex AI, Azure ML).
- Connects to **data platforms** (S3, GCS, Azure Blob Storage, Hugging Face Hub) for artifact storage.
- Works alongside **serving infrastructure** (MLflow, BentoML, Seldon) for production deployment.
- Feeds into **monitoring tools** that track production model performance over time.

By maintaining a continuous record from experiment to production, W&B provides the observability foundation that makes ML systems reproducible, debuggable, and improvable at scale.
