---
title: Introduction to ZenML
description: Get started with ZenML — the open-source MLOps framework for building portable, production-ready ML pipelines — covering steps, pipelines, stacks, artifact stores, model registry, experiment tracking integrations, and deployment to cloud infrastructure.
---

ML models frequently work in notebooks but break down when moved to production. Pipelines that were tightly coupled to a local environment, manual experiment tracking spread across spreadsheets, and deployment processes that only one person understands are common failure patterns. **ZenML** is an open-source MLOps framework that addresses these problems by providing a standard abstraction layer between your ML code and the underlying infrastructure — making pipelines portable, reproducible, and auditable across any cloud or orchestrator.

## Core Concepts

ZenML centers on four abstractions:

- **Steps**: individual Python functions decorated with `@step` that perform a single ML task (data loading, preprocessing, training, evaluation)
- **Pipelines**: sequences of steps decorated with `@pipeline`, wired together by ZenML's artifact passing system
- **Artifacts**: versioned outputs produced by steps (datasets, models, metrics) automatically tracked in a metadata store
- **Stacks**: composable collections of infrastructure components (orchestrator, artifact store, experiment tracker, model deployer) that define where and how pipelines run

## Installation

```bash
pip install zenml

# Initialize a ZenML project in the current directory
zenml init

# Start the ZenML dashboard (optional)
pip install "zenml[server]"
zenml up
```

## Your First Pipeline

```python
from zenml import step, pipeline
from zenml.logger import get_logger
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

logger = get_logger(__name__)


@step
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and return features and labels."""
    iris = load_iris()
    return iris.data, iris.target


@step
def split_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)


@step
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train and return a RandomForest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


@step
def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Evaluate model accuracy."""
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Model accuracy: {accuracy:.4f}")
    return accuracy


@pipeline
def training_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    training_pipeline()
```

Run it:

```bash
python run.py
```

ZenML automatically tracks every artifact (the dataset splits, trained model, accuracy metric), links them to the pipeline run, and stores metadata in the local SQLite store.

## Artifact Versioning

Every step output becomes a versioned **artifact** in ZenML's artifact store. You can retrieve past artifacts programmatically:

```python
from zenml.client import Client

client = Client()

# Get the latest run of a pipeline
pipeline_run = client.get_pipeline("training_pipeline").last_run

# Access a specific step's output
model_artifact = pipeline_run.steps["train_model"].outputs["output"].load()
accuracy = pipeline_run.steps["evaluate_model"].outputs["output"].load()

print(f"Loaded model: {model_artifact}")
print(f"Accuracy: {accuracy:.4f}")
```

Artifacts are automatically serialized/deserialized using materializers. ZenML ships materializers for common types (NumPy arrays, pandas DataFrames, scikit-learn models, PyTorch models) and you can write custom ones.

## Custom Materializers

```python
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
import joblib
import os


class SklearnModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (RandomForestClassifier,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type):
        model_path = os.path.join(self.uri, "model.joblib")
        return joblib.load(model_path)

    def save(self, model):
        os.makedirs(self.uri, exist_ok=True)
        joblib.dump(model, os.path.join(self.uri, "model.joblib"))
```

## Stacks: Composable Infrastructure

A **stack** defines the infrastructure components for pipeline execution. Stacks are configured via the CLI and can be switched without changing pipeline code.

### Stack Components

| Component | Role | Examples |
| --- | --- | --- |
| Orchestrator | Runs pipeline steps | Local, Airflow, Kubeflow, Vertex AI, Sagemaker |
| Artifact Store | Stores pipeline artifacts | Local filesystem, S3, GCS, Azure Blob |
| Experiment Tracker | Logs metrics and parameters | MLflow, W&B, Comet |
| Model Registry | Versions and stages models | MLflow Model Registry, W&B |
| Model Deployer | Serves models | BentoML, Seldon, Sagemaker Endpoints |
| Data Validator | Validates datasets | Great Expectations, Evidently |
| Alert | Sends notifications | Slack, PagerDuty |

### Configuring a Stack with MLflow and S3

```bash
# Register artifact store (S3)
zenml artifact-store register s3_store \
  --flavor=s3 \
  --path=s3://my-ml-artifacts/zenml

# Register experiment tracker (MLflow)
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri=http://mlflow.internal:5000 \
  --tracking_username=admin \
  --tracking_password=secret

# Register model registry (MLflow)
zenml model-registry register mlflow_registry \
  --flavor=mlflow \
  --tracking_uri=http://mlflow.internal:5000

# Compose the stack
zenml stack register production_stack \
  -o default \
  -a s3_store \
  -e mlflow_tracker \
  -r mlflow_registry

# Activate the stack
zenml stack set production_stack
```

Now any pipeline run automatically logs to MLflow and stores artifacts in S3 — with no changes to pipeline code.

## Experiment Tracking Integration

When a stack includes an experiment tracker, ZenML automatically logs step parameters and outputs. You can also log custom metrics:

```python
from zenml.client import Client
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
import mlflow


@step(
    experiment_tracker="mlflow_tracker",
    settings={
        "experiment_tracker.mlflow": MLFlowExperimentTrackerSettings(
            experiment_name="iris_classification",
            tags={"framework": "sklearn", "dataset": "iris"},
        )
    },
)
def train_model_tracked(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    mlflow.log_param("n_estimators", n_estimators)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    return model
```

## Model Registry and Versioning

ZenML's model registry tracks model versions with stages (staging, production, archived):

```python
from zenml import step, pipeline, Model
from zenml.enums import ModelStages

# Associate a pipeline run with a named model
@pipeline(
    model=Model(
        name="iris_classifier",
        version="1.2.0",
        tags=["production-candidate"],
    )
)
def training_pipeline_with_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)


# Promote a model version to production via CLI or Python
client = Client()
model_version = client.get_model_version("iris_classifier", "1.2.0")
model_version.set_stage(ModelStages.PRODUCTION, force=True)
```

## Pipeline Configuration and Parameterization

ZenML pipelines accept runtime configurations via YAML:

```yaml
# config.yaml
steps:
  train_model_tracked:
    parameters:
      n_estimators: 200
  split_data:
    parameters:
      test_size: 0.15
```

```bash
python run.py --config config.yaml
```

This enables running the same pipeline with different parameters without code changes — useful for hyperparameter sweeps.

## Step Caching

ZenML caches step outputs by default. If the step's code, inputs, and parameters are unchanged, the step is skipped and the cached artifact is reused:

```python
@step(enable_cache=True)   # Default: True
def load_data() -> tuple[np.ndarray, np.ndarray]:
    ...

# Disable caching for a step that should always re-run (e.g., data ingestion)
@step(enable_cache=False)
def fetch_latest_data() -> pd.DataFrame:
    ...
```

Caching dramatically speeds up development: iterating on the training step doesn't re-run data loading and preprocessing.

## Cloud Orchestration with Kubeflow

Switching to Kubeflow Pipelines on Kubernetes requires only stack configuration — the pipeline code is unchanged:

```bash
zenml integration install kubeflow kubernetes

zenml orchestrator register kubeflow_orchestrator \
  --flavor=kubeflow \
  --kubeflow_hostname=https://kubeflow.internal

zenml stack update production_stack -o kubeflow_orchestrator
```

On the next pipeline run, ZenML packages each step as a Docker container, submits the DAG to Kubeflow, and streams logs back to the ZenML dashboard.

## ZenML vs Other MLOps Frameworks

| Feature | ZenML | MLflow | Metaflow | Prefect |
| --- | --- | --- | --- | --- |
| Pipeline orchestration | Yes | No (tracking only) | Yes | Yes |
| Artifact versioning | Yes (native) | Yes | Yes | Partial |
| Stack abstraction | Yes | No | No | No |
| Multi-cloud portability | Yes | Partial | Partial | Yes |
| Model registry | Yes | Yes | No | No |
| Open source | Yes | Yes | Yes | Yes |
| ML-specific primitives | Yes | Yes | No | No |

## Summary

ZenML makes ML pipelines production-ready through clean abstractions:

- **Steps and pipelines** convert ML code into reusable, versioned, cached units of work
- **Artifacts** are automatically tracked and versioned — every run is reproducible
- **Stacks** decouple pipeline logic from infrastructure — swap orchestrators and artifact stores without touching pipeline code
- **Model registry** provides a unified model lifecycle from experiment to production staging
- **Integrations** with MLflow, W&B, Kubeflow, Sagemaker, and others mean ZenML fits into existing toolchains rather than replacing them

For teams moving from notebook-driven ML to production systems, ZenML provides the scaffold that makes pipelines repeatable, auditable, and deployable across any cloud environment.
