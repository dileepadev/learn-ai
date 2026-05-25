---
title: Introduction to Metaflow
description: Get started with Metaflow — Netflix's open-source ML workflow framework — covering FlowSpec, @step decorators, branching with foreach, artifact persistence, @resources for cloud scaling, @retry and @timeout for resilience, reproducible environments with @pypi, and running workflows on AWS Batch and Kubernetes.
---

Netflix built Metaflow to solve a specific problem: data scientists writing code that worked in Jupyter notebooks but failed when the data scientist left, scaled up, or changed environments. **Metaflow** is an open-source Python framework that makes ML workflows reproducible, scalable, and resilient by providing a thin, opinionated abstraction over your code, data, and compute infrastructure.

## Core Philosophy

Metaflow is designed around three principles:

- **Versioning by default**: every run, every step, every artifact is automatically versioned and stored persistently — you can always reproduce any past result
- **Local-first development**: code runs identically on a laptop and on cloud compute; the only change is adding `--with batch` to the command
- **Python-native**: no YAML, no DSLs — workflows are Python classes using standard decorators

## Basic Flow Structure

A Metaflow workflow is a Python class that inherits from `FlowSpec`. Each step is a method decorated with `@step`:

```python
from metaflow import FlowSpec, step


class TrainingFlow(FlowSpec):

    @step
    def start(self):
        """Load data and define parameters."""
        import pandas as pd
        self.train_df = pd.read_parquet("s3://my-bucket/train.parquet")
        self.test_df = pd.read_parquet("s3://my-bucket/test.parquet")
        self.learning_rate = 0.01
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Feature engineering and train/test split."""
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.train_df.drop("label", axis=1))
        self.y_train = self.train_df["label"].values
        self.X_test = scaler.transform(self.test_df.drop("label", axis=1))
        self.y_test = self.test_df["label"].values
        self.scaler = scaler
        self.next(self.train)

    @step
    def train(self):
        """Train model."""
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=1.0 / self.learning_rate, max_iter=500)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Compute final metrics."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        preds = self.model.predict(self.X_test)
        proba = self.model.predict_proba(self.X_test)[:, 1]
        self.accuracy = accuracy_score(self.y_test, preds)
        self.auc = roc_auc_score(self.y_test, proba)
        print(f"Accuracy: {self.accuracy:.4f}, AUC: {self.auc:.4f}")
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Training complete. Final AUC: {self.auc:.4f}")


if __name__ == "__main__":
    TrainingFlow()
```

Run locally:

```bash
python flow.py run
```

## Artifact Persistence

Every attribute assigned to `self` within a step becomes a **Metaflow artifact** — automatically serialized and stored in the artifact store (local filesystem, S3, or Azure Blob). Artifacts are versioned per run and accessible after the run completes:

```python
from metaflow import Flow, Run

# Access the latest run's artifacts
run = Flow("TrainingFlow").latest_run
print(f"AUC: {run["evaluate"].task.data.auc}")
print(f"Model: {run["train"].task.data.model}")

# Access a specific past run
past_run = Run("TrainingFlow/1234")
past_model = past_run["train"].task.data.model
```

Artifacts are content-addressed — if two steps produce identical artifacts, the data is stored only once.

## Branching with foreach

Metaflow's `foreach` enables parallel branches — essential for hyperparameter sweeps, cross-validation folds, and data partitions:

```python
from metaflow import FlowSpec, step, Parameter


class HyperparamSweepFlow(FlowSpec):

    @step
    def start(self):
        self.configs = [
            {"n_estimators": 100, "max_depth": 4},
            {"n_estimators": 200, "max_depth": 6},
            {"n_estimators": 300, "max_depth": 8},
            {"n_estimators": 500, "max_depth": 10},
        ]
        # fan out: one branch per config
        self.next(self.train, foreach="configs")

    @step
    def train(self):
        # self.input = the current foreach element
        from sklearn.ensemble import RandomForestClassifier

        config = self.input
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)
        self.auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        self.config = config
        self.model = model
        self.next(self.join)

    @step
    def join(self, inputs):
        # inputs = list of all branch artifacts
        best = max(inputs, key=lambda x: x.auc)
        self.best_model = best.model
        self.best_config = best.config
        self.best_auc = best.auc
        print(f"Best AUC: {self.best_auc:.4f} with config: {self.best_config}")
        self.next(self.end)

    @step
    def end(self):
        pass
```

Metaflow executes `foreach` branches in parallel — either as local processes or as independent cloud jobs depending on the execution environment.

## Parameters

Flows accept typed command-line parameters:

```python
from metaflow import FlowSpec, step, Parameter


class ParameterizedFlow(FlowSpec):

    learning_rate = Parameter(
        "learning_rate",
        help="Learning rate for the optimizer",
        default=0.01,
        type=float,
    )

    n_estimators = Parameter(
        "n_estimators",
        help="Number of trees",
        default=100,
        type=int,
    )

    dataset_path = Parameter(
        "dataset_path",
        help="S3 path to training data",
        required=True,
    )

    @step
    def start(self):
        print(f"Learning rate: {self.learning_rate}")
        self.next(self.train)
```

```bash
python flow.py run --learning_rate 0.001 --n_estimators 500 --dataset_path s3://bucket/data.parquet
```

## Cloud Scaling with @resources and @batch

Add `@resources` to allocate more compute and `@batch` to run on AWS Batch — no code changes:

```python
from metaflow import FlowSpec, step, resources, batch, retry, timeout


class ProductionFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.train)

    @retry(times=3)
    @timeout(minutes=60)
    @resources(cpu=8, memory=32000, gpu=1)
    @batch(image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0-gpu-py310")
    @step
    def train(self):
        """Runs on AWS Batch with 8 CPUs, 32GB RAM, 1 GPU."""
        import torch
        # ... heavy training code
        self.model = trained_model
        self.next(self.end)

    @step
    def end(self):
        pass
```

Run with remote execution:

```bash
python flow.py --environment=pypi run --with batch
```

The `@retry` decorator automatically retries failed steps (cloud spot interruptions, transient errors). The `@timeout` decorator kills steps that take too long and marks them as failed rather than hanging indefinitely.

## Reproducible Environments with @pypi

Pin exact dependencies per step using `@pypi`:

```python
from metaflow import pypi, step


class EnvironmentFlow(FlowSpec):

    @pypi(packages={
        "scikit-learn": "1.4.0",
        "pandas": "2.1.0",
        "xgboost": "2.0.3",
    })
    @step
    def train(self):
        import xgboost as xgb
        # Runs with exactly the pinned versions regardless of host environment
        ...
```

Metaflow creates an isolated environment per step — different steps can use different library versions. This solves the "it works on my machine" problem.

## Accessing Run History

The `Client API` provides programmatic access to past runs for analysis, comparison, and artifact retrieval:

```python
from metaflow import Flow, Run, namespace
import pandas as pd

# Compare AUC across all production runs
flow = Flow("TrainingFlow")

records = []
for run in flow.runs("production"):
    try:
        records.append({
            "run_id": run.id,
            "created_at": run.created_at,
            "auc": run["evaluate"].task.data.auc,
            "accuracy": run["evaluate"].task.data.accuracy,
        })
    except Exception:
        continue

df = pd.DataFrame(records).sort_values("created_at")
print(df.tail(10))
```

## Tagging and Namespaces

Runs can be tagged for organization and filtered by tag:

```bash
# Tag a run
python flow.py run --tag production --tag model_v2

# Resume a failed run from a specific step
python flow.py resume --origin-run-id 1234 train
```

```python
# Filter runs by tag
for run in flow.runs("production"):
    print(run.id, run.created_at)
```

## Metaflow vs Other ML Workflow Frameworks

| Feature | Metaflow | ZenML | Kubeflow Pipelines | Prefect |
| --- | --- | --- | --- | --- |
| Python-native | Yes | Yes | No (YAML DSL) | Yes |
| Local-to-cloud | Seamless | Via stacks | Requires K8s | Yes |
| Artifact versioning | Yes (automatic) | Yes | Partial | Partial |
| foreach / fan-out | Yes (native) | Via steps | Yes | Yes |
| Reproducible envs | `@pypi` / `@conda` | Docker images | Docker images | Docker images |
| ML-specific tooling | Strong | Strong | Moderate | Moderate |
| Open source | Yes | Yes | Yes | Yes |
| Managed offering | Outerbounds | ZenML Cloud | Google Vertex | Prefect Cloud |

## Summary

Metaflow makes ML workflows production-ready through a minimal, Python-native API:

- **FlowSpec and @step** turn ML scripts into versioned, reproducible DAGs
- **Automatic artifact persistence** stores every intermediate result — runs are always reproducible
- **foreach** enables data-parallel and hyperparameter-parallel execution with automatic fan-out/join
- **@resources and @batch** scale individual steps to cloud compute with a single decorator — zero code changes
- **@retry and @timeout** make workflows resilient to cloud transience without defensive boilerplate
- **@pypi and @conda** pin exact dependencies per step, solving environment reproducibility at the source

Metaflow's philosophy — workflows should feel like Python, not configuration — makes it particularly well-suited for data science teams where the priority is moving from experiment to production quickly.
