---
title: Introduction to Evidently AI
description: Get started with Evidently AI — the open-source framework for ML model monitoring and data quality evaluation — covering data drift detection, model performance reports, test suites, the monitoring dashboard, and integration with production ML pipelines.
---

Models degrade silently in production. Input data distributions shift, upstream pipelines change, user behavior evolves — and accuracy quietly erodes long before anyone notices. **Evidently AI** is an open-source Python library for monitoring ML models and data quality, providing pre-built reports, statistical tests, and a real-time dashboard to detect and diagnose these issues before they impact users.

## Core Concepts

Evidently is organized around three layers:

- **Metrics**: individual calculations (e.g., share of drifted features, mean prediction value)
- **Reports**: collections of metrics rendered as interactive HTML or returned as Python dictionaries
- **Test Suites**: collections of assertions that pass or fail — suitable for CI/CD quality gates

All three operate on the same abstraction: compare a **reference dataset** (training data, a recent production window) against a **current dataset** (new production data, a deployment batch).

## Installation

```bash
pip install evidently
```

For the monitoring dashboard and self-hosted service:

```bash
pip install "evidently[service]"
```

## Quick Start: Data Drift Report

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Reference: training data
reference_data = pd.read_csv("train.csv")

# Current: recent production data
current_data = pd.read_csv("production_batch.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save interactive HTML
report.save_html("drift_report.html")

# Or get results as a dictionary
results = report.as_dict()
print(results["metrics"][0]["result"]["share_of_drifted_columns"])
```

The report compares distributions of every feature and outputs per-feature drift scores, detected drift (boolean), and a summary.

## Data Drift Detection

### How Drift is Detected

Evidently selects a statistical test automatically based on feature type and dataset size:

| Feature Type | Default Test | Alternative |
| --- | --- | --- |
| Numerical (large $n$) | Wasserstein distance | KS test, PSI |
| Numerical (small $n$) | KS test | Anderson-Darling |
| Categorical | Chi-squared test | Jensen-Shannon |
| Text | Domain classifier AUC | Embedding distance |

The drift threshold (p-value or distance threshold) is configurable per-feature.

### Configuring Drift Tests

```python
from evidently.report import Report
from evidently.metrics import DataDriftTable, ColumnDriftMetric
from evidently.tests import TestColumnDrift

# Granular column-level drift
report = Report(metrics=[
    ColumnDriftMetric(column_name="age", stattest="ks", stattest_threshold=0.05),
    ColumnDriftMetric(column_name="income", stattest="wasserstein", stattest_threshold=0.1),
    ColumnDriftMetric(column_name="category", stattest="chisquare"),
    DataDriftTable(),
])

report.run(reference_data=reference_data, current_data=current_data)
```

### Population Stability Index (PSI)

PSI is widely used in credit scoring and finance to detect distribution shift. Evidently supports PSI as a drift statistic:

$$\text{PSI} = \sum_i (P_i - Q_i) \cdot \ln\frac{P_i}{Q_i}$$

- PSI < 0.1: no significant shift
- 0.1 ≤ PSI < 0.2: moderate shift — investigate
- PSI ≥ 0.2: major shift — action required

```python
from evidently.metrics import ColumnDriftMetric

metric = ColumnDriftMetric(
    column_name="credit_score",
    stattest="psi",
    stattest_threshold=0.2,
)
```

## Model Performance Reports

When labels are available (delayed ground truth), Evidently evaluates model degradation:

```python
from evidently.metric_preset import ClassificationPreset, RegressionPreset

# Classification monitoring
clf_report = Report(metrics=[ClassificationPreset()])
clf_report.run(
    reference_data=reference_with_labels,
    current_data=current_with_labels,
    column_mapping=column_mapping,
)

# Regression monitoring
reg_report = Report(metrics=[RegressionPreset()])
reg_report.run(
    reference_data=reference_with_labels,
    current_data=current_with_labels,
    column_mapping=column_mapping,
)
```

`ClassificationPreset` includes accuracy, precision, recall, F1, confusion matrix, class distribution, and prediction drift. `RegressionPreset` covers MAE, RMSE, MAPE, error distribution, and actual vs. predicted plots.

## Column Mapping

Column mapping tells Evidently which columns are predictions, targets, IDs, and features:

```python
from evidently import ColumnMapping

column_mapping = ColumnMapping(
    target="loan_default",           # Ground truth label column
    prediction="model_score",        # Model prediction column
    id="customer_id",                # Identifier (excluded from analysis)
    datetime="timestamp",            # Datetime column
    numerical_features=["age", "income", "credit_score"],
    categorical_features=["employment_type", "loan_purpose"],
    text_features=["application_notes"],
)
```

## Test Suites for CI/CD

Test suites replace visual inspection with programmatic pass/fail checks — suitable for automated quality gates in ML pipelines:

```python
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, DataDriftTestPreset
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfMissingValues,
    TestColumnDrift,
    TestValueMeanInNSigmas,
)

suite = TestSuite(tests=[
    DataStabilityTestPreset(),            # Checks missing values, constant cols, etc.
    DataDriftTestPreset(),                # Checks for drift in all features
    TestNumberOfDriftedColumns(lte=3),    # Fail if more than 3 features drift
    TestShareOfMissingValues(lt=0.05),    # Fail if >5% missing values
    TestColumnDrift("age"),               # Specific column must not drift
    TestValueMeanInNSigmas("income", n=2),  # Mean within 2 sigma of reference
])

suite.run(reference_data=reference_data, current_data=current_data)

# In CI: exit with failure if tests fail
if not suite.as_dict()["summary"]["all_passed"]:
    print(suite.as_dict()["summary"])
    raise ValueError("Data quality tests failed — block deployment")
```

## Data Quality Monitoring

Beyond drift, Evidently monitors dataset health:

```python
from evidently.metric_preset import DataQualityPreset

quality_report = Report(metrics=[DataQualityPreset()])
quality_report.run(reference_data=reference_data, current_data=current_data)
```

Checks include:

- Missing value rates per column
- Constant and near-constant features
- Duplicate rows
- Feature correlation changes
- Out-of-range values (compared to reference min/max)
- Class imbalance (for classification targets)

## Text and Embedding Drift

For NLP models, Evidently monitors text feature drift using:

- **Domain classifier**: trains a binary classifier to distinguish reference from current text samples — AUC measures separability
- **Embedding drift**: computes embeddings with a chosen model and measures distance between reference/current embedding distributions

```python
from evidently.descriptors import TextLength, Sentiment, OOV, NonLetterCharacterPercentage
from evidently.metrics import ColumnSummaryMetric, TextDescriptorsDistribution

text_report = Report(metrics=[
    TextDescriptorsDistribution(column_name="review_text"),
    ColumnSummaryMetric(
        column_name="review_text",
        descriptors={
            "length": TextLength(),
            "sentiment": Sentiment(),
            "oov_rate": OOV(),
        }
    ),
])
```

## Evidently Service: Self-Hosted Dashboard

For continuous production monitoring, Evidently Service provides a web dashboard with historical tracking:

```python
# workspace.py — define a project
from evidently.ui.workspace import Workspace

ws = Workspace.create("./monitoring_workspace")
project = ws.create_project("Loan Default Model")
project.description = "Production monitoring for loan default classifier"

# Scheduled job adds snapshots over time
from evidently.ui.dashboards import DashboardConfig, CounterAgg, PanelValue

project.dashboard.add_panel(
    DashboardConfig(
        title="Data Drift Share",
        values=[
            PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="Drifted Features %",
            )
        ],
    )
)
```

Run the dashboard:

```bash
evidently ui --workspace ./monitoring_workspace --port 8080
```

## Integration with ML Pipelines

### Apache Airflow

```python
from airflow.decorators import task

@task
def monitor_data_drift(reference_path: str, current_path: str):
    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(current_path)

    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(reference_data=reference, current_data=current)

    results = suite.as_dict()
    if not results["summary"]["all_passed"]:
        raise ValueError(f"Drift detected: {results['summary']}")

    return results
```

### Batch Scoring Pipelines

```python
# After daily batch scoring — check before deploying predictions
def validate_batch(predictions_df, reference_df):
    suite = TestSuite(tests=[
        TestShareOfMissingValues(lt=0.01),
        TestNumberOfDriftedColumns(lte=2),
        TestValueMeanInNSigmas("prediction_score", n=3),
    ])
    suite.run(reference_data=reference_df, current_data=predictions_df)

    if not suite.as_dict()["summary"]["all_passed"]:
        alert_team("Prediction drift detected — pausing batch deployment")
        return False
    return True
```

## Evidently vs Other Monitoring Tools

| Feature | Evidently | Great Expectations | Whylogs | Arize Phoenix |
| --- | --- | --- | --- | --- |
| Data drift | Yes (rich) | Partial | Yes | Yes |
| Model performance | Yes | No | Partial | Yes |
| Text/embedding drift | Yes | No | Partial | Yes |
| Self-hosted dashboard | Yes | No | No | Yes |
| CI/CD test suites | Yes | Yes | No | Partial |
| LLM monitoring | Partial | No | No | Yes (strong) |
| Open source | Yes | Yes | Yes | Yes |

## Summary

Evidently AI provides a complete toolkit for ML monitoring and data quality evaluation:

- **Reports** generate visual, interactive analyses of drift, quality, and model performance
- **Test Suites** translate monitoring checks into automated pass/fail gates for CI/CD pipelines
- **Column mapping** cleanly separates features, targets, predictions, and metadata
- **Text descriptors** extend drift detection to unstructured data without custom code
- **Evidently Service** provides a self-hosted dashboard for continuous historical monitoring

For teams deploying models to production, integrating Evidently into data ingestion and batch scoring pipelines provides an early warning system for the silent degradation that affects every model over time.
