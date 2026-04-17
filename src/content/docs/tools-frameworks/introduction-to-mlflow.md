---
title: Introduction to MLflow
description: A practical guide to MLflow, the open-source platform for managing the machine learning lifecycle, covering experiment tracking, model registry, MLflow Projects, model deployment, and the unified MLflow AI Gateway.
---

MLflow is an open-source platform developed by Databricks that addresses the operational complexity of machine learning projects. It provides a consistent, reproducible workflow from experimentation through deployment, compatible with any ML library, cloud provider, or infrastructure.

The platform is organised around four core components: **Tracking**, **Projects**, **Models**, and the **Model Registry**.

## Experiment Tracking

Tracking is the most widely adopted MLflow feature. Every training run can log parameters, metrics, artefacts, and environment metadata to a central server or local filesystem.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="rf-baseline"):
    params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, artifact_path="model")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### What Can Be Logged

| Item | API | Examples |
| --- | --- | --- |
| Parameters | `mlflow.log_param()` | Learning rate, batch size, feature set |
| Metrics | `mlflow.log_metric()` | Accuracy, RMSE, F1 at each epoch |
| Artefacts | `mlflow.log_artifact()` | Plots, confusion matrices, data samples |
| Models | `mlflow.log_model()` | Serialised model with schema and dependencies |
| Tags | `mlflow.set_tag()` | Team, dataset version, environment |

### Auto-Logging

```python
mlflow.autolog()  # Automatically captures supported framework parameters and metrics
```

Auto-logging supports scikit-learn, TensorFlow, Keras, PyTorch Lightning, XGBoost, LightGBM, Spark MLlib, and more. It requires no changes to existing training code beyond a single function call.

### The MLflow UI

`mlflow ui` launches a local web server at `http://localhost:5000` where runs can be compared across experiments with parallel coordinates plots, metric graphs, and artefact viewers. Hosted alternatives include Databricks Managed MLflow, Azure ML MLflow integration, and AWS SageMaker Experiments.

## MLflow Projects

An MLflow Project is a directory containing a `MLproject` file that specifies the environment and entry points for reproducible execution.

```yaml
# MLproject
name: churn-prediction

conda_env: conda.yaml

entry_points:
  train:
    parameters:
      learning_rate: {type: float, default: 0.01}
      max_iter: {type: int, default: 100}
    command: "python train.py --lr {learning_rate} --max_iter {max_iter}"
```

Projects can be run locally or on remote compute:

```bash
mlflow run . -P learning_rate=0.05
mlflow run https://github.com/org/project -P learning_rate=0.05
```

This enables reproducibility across machines, hyperparameter sweeps, and programmatic orchestration from tools like Airflow or Prefect.

## MLflow Models

The MLflow Models format packages a trained model with its dependencies, input/output schema (MLflow signature), and serving code into a portable directory.

```python
from mlflow.models import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    signature=signature,
    input_example=X_train[:5]
)
```

### Model Flavours

MLflow models support multiple **flavours** — serialisation formats that a model can be loaded as:

| Flavour | Use case |
| --- | --- |
| `python_function` | Universal serving interface |
| `sklearn` | scikit-learn models |
| `pytorch` | PyTorch `nn.Module` |
| `transformers` | Hugging Face models |
| `langchain` | LangChain chains and agents |
| `tensorflow` | TF SavedModel |
| `spark` | Spark MLlib pipelines |

### Model Serving

Logged models can be served as REST APIs:

```bash
mlflow models serve -m runs:/<run_id>/model -p 1234 --no-conda
```

The server exposes a `/invocations` endpoint accepting JSON payloads matching the logged input schema, making deployment during evaluation and staging straightforward.

## Model Registry

The Model Registry provides versioned, stage-managed storage for production-bound models.

```python
import mlflow

# Register model from a completed run
mlflow.register_model("runs:/<run_id>/model", name="FraudDetector")

# Transition to staging or production
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="FraudDetector", version=3, stage="Production"
)
```

### Registry Stages

| Stage | Purpose |
| --- | --- |
| None / Staging | Candidate model under evaluation |
| Production | Currently serving live traffic |
| Archived | Retired; kept for audit and rollback |

Registry metadata supports descriptions, tags, and lineage links back to training runs, datasets, and Git commits.

## MLflow AI Gateway

The AI Gateway (formerly MLflow Deployments) provides a unified API layer across multiple LLM providers, enabling teams to switch providers without changing application code:

```python
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={"messages": [{"role": "user", "content": "Summarise this document."}]}
)
```

It supports rate limiting, authentication centralisation, and usage tracking across teams.

## Integration with CI/CD

MLflow integrates with MLOps pipelines to automate model promotion:

1. Training job logs run to MLflow Tracking Server
2. Evaluation job compares new run against the current production model on a hold-out dataset
3. If metrics improve beyond a threshold, the model is registered and transitioned to Staging
4. Integration tests run against the Staging endpoint
5. Manual or automated approval promotes the model to Production

```python
# Compare new model to current production model
client = mlflow.MlflowClient()
production_versions = client.get_latest_versions("FraudDetector", stages=["Production"])
prod_run = client.get_run(production_versions[0].run_id)
prod_acc = prod_run.data.metrics["accuracy"]

if new_acc > prod_acc + 0.005:
    client.transition_model_version_stage("FraudDetector", new_version, "Production")
```

## Deployment Targets

| Target | Method |
| --- | --- |
| Local REST API | `mlflow models serve` |
| Docker container | `mlflow models build-docker` then deploy |
| Azure ML | `mlflow.deployments` with AzureML plugin |
| AWS SageMaker | `mlflow.sagemaker.deploy()` |
| Databricks | Model Serving with auto-scaling |
| Kubernetes | KServe + MLflow model format |

MLflow's framework-agnostic design and broad ecosystem integrations make it a practical foundation for teams that want reproducible ML workflows without locking into a proprietary platform.
