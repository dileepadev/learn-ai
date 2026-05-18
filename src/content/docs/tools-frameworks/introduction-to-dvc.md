---
title: Introduction to DVC (Data Version Control)
description: Get started with DVC — the open-source tool for versioning large datasets, tracking ML experiments, and building reproducible pipelines — covering core commands, remote storage configuration, pipeline stages, experiment management, and Git integration.
---

Git excels at versioning code but breaks down with large binary files: datasets of gigabytes or terabytes cannot be committed to a repository without bloating clone sizes and slowing every operation. **DVC (Data Version Control)** solves this by storing large files in external storage (S3, GCS, Azure, local disk) while keeping lightweight `.dvc` pointer files in Git. The result is reproducible ML workflows where code, data, models, and experiment results are all versioned together.

## Core Concepts

DVC introduces two key abstractions:

- **Data versioning**: `.dvc` files tracked by Git point to specific content-addressed versions of large files stored remotely
- **Pipelines**: `dvc.yaml` defines a DAG of stages with tracked inputs, outputs, and commands — enabling `dvc repro` to re-run only what changed

Together these allow a collaborator to run `git checkout <commit> && dvc pull` to reproduce the exact data state of any historical experiment.

## Installation

```bash
pip install dvc

# With cloud storage support
pip install "dvc[s3]"       # AWS S3
pip install "dvc[gs]"       # Google Cloud Storage
pip install "dvc[azure]"    # Azure Blob Storage
pip install "dvc[all]"      # All remote backends
```

## Initializing a Project

```bash
cd my-ml-project
git init
dvc init
git commit -m "Initialize DVC"
```

`dvc init` creates a `.dvc/` directory (committed to Git) containing DVC configuration, cache location, and internal metadata.

## Tracking Data Files

### Adding Files and Directories

```bash
dvc add data/train.csv
dvc add data/images/       # Track an entire directory
```

This creates:

- `data/train.csv.dvc` — a pointer file containing the MD5 hash and size of the data
- An entry in `.gitignore` so the actual data file is not committed to Git

The `.dvc` file is small (a few lines of YAML) and goes into Git:

```yaml
outs:
- md5: d8e8fca2dc0f896fd7cb4cb0031ba249
  size: 1234567
  path: train.csv
```

The actual data goes into DVC's local cache (`~/.dvc/cache` by default) and later to a remote.

```bash
git add data/train.csv.dvc data/.gitignore
git commit -m "Track training dataset with DVC"
```

### Pulling and Pushing Data

```bash
dvc push   # Upload cached data to remote storage
dvc pull   # Download data from remote to local cache
dvc fetch  # Download to cache only (not workspace)
```

## Configuring Remote Storage

```bash
# AWS S3
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote region us-east-1

# Google Cloud Storage
dvc remote add -d myremote gs://my-bucket/dvc-store

# Azure Blob Storage
dvc remote add -d myremote azure://mycontainer/dvc-store

# Local or SSH
dvc remote add -d myremote /mnt/shared/dvc-cache
dvc remote add -d myremote ssh://user@host:/path/to/cache

git add .dvc/config
git commit -m "Configure DVC remote storage"
```

The `-d` flag sets this as the default remote. Credentials are handled through standard cloud CLI tools (AWS CLI, `gcloud auth`, Azure CLI) — DVC does not store secrets.

## Building Reproducible Pipelines

DVC pipelines define ML workflows as directed acyclic graphs. Each stage declares its command, inputs (dependencies), and outputs.

### Creating a Pipeline

```bash
dvc run -n prepare \
  -d src/prepare.py -d data/raw.csv \
  -o data/prepared/ \
  python src/prepare.py data/raw.csv data/prepared/

dvc run -n train \
  -d src/train.py -d data/prepared/ -d params.yaml \
  -o models/model.pkl \
  -m metrics/train_metrics.json \
  python src/train.py

dvc run -n evaluate \
  -d src/evaluate.py -d models/model.pkl -d data/prepared/ \
  -m metrics/eval_metrics.json \
  python src/evaluate.py
```

This writes a `dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python src/prepare.py data/raw.csv data/prepared/
    deps:
      - src/prepare.py
      - data/raw.csv
    outs:
      - data/prepared/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/prepared/
      - params.yaml
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/prepared/
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

### Running the Pipeline

```bash
dvc repro         # Re-run only stages whose dependencies changed
dvc repro -f      # Force re-run all stages
dvc dag           # Print the pipeline DAG
```

DVC tracks checksums of all dependencies. If `data/raw.csv` hasn't changed but `src/train.py` has, only the `train` and `evaluate` stages re-run.

## Parameters

`params.yaml` stores hyperparameters that are tracked as pipeline dependencies:

```yaml
train:
  learning_rate: 0.001
  batch_size: 64
  epochs: 50
  hidden_dim: 256

data:
  test_split: 0.2
  random_seed: 42
```

Reference parameters in `dvc.yaml`:

```yaml
train:
  cmd: python src/train.py
  params:
    - train.learning_rate
    - train.batch_size
    - train.epochs
```

DVC tracks parameter values in `dvc.lock` — any change to tracked parameters marks the stage as outdated and triggers re-execution.

## Experiment Management

DVC experiments run variations of the pipeline with different parameters, storing results without creating Git commits:

```bash
# Run with modified parameters
dvc exp run --set-param train.learning_rate=0.01

# Run a grid search
dvc exp run --set-param train.learning_rate=0.001,0.01,0.1 \
            --set-param train.batch_size=32,64

# Name an experiment
dvc exp run --name "lr-sweep-v1" --set-param train.learning_rate=0.005
```

### Comparing Experiments

```bash
# Show all experiments
dvc exp show

# Compact table output
dvc exp show --md          # Markdown table
dvc exp show --csv         # CSV output

# Compare two experiments
dvc exp diff exp1 exp2
```

Sample output of `dvc exp show`:

```text
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Experiment  ┃ Created   ┃ learning_rate┃ batch_size   ┃ val_acc    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ workspace   │ -         │ 0.001        │ 64           │ 0.923      │
│ main        │ 05/17     │ 0.001        │ 64           │ 0.917      │
│ exp-a1b2    │ 05/17     │ 0.01         │ 64           │ 0.891      │
│ exp-c3d4    │ 05/17     │ 0.001        │ 32           │ 0.921      │
└─────────────┴───────────┴──────────────┴──────────────┴────────────┘
```

### Promoting an Experiment

```bash
# Apply experiment changes to working directory
dvc exp apply exp-a1b2

# Commit to Git branch
dvc exp branch exp-a1b2 my-feature-branch
```

## DVCLive: Real-Time Experiment Logging

DVCLive integrates with training code to log metrics during runs:

```python
from dvclive import Live

with Live("dvclive") as live:
    for epoch in range(100):
        train_loss = train_epoch()
        val_acc = evaluate()

        live.log_metric("train/loss", train_loss)
        live.log_metric("val/accuracy", val_acc)
        live.next_step()
```

DVCLive writes metrics to files that DVC tracks, enabling real-time monitoring and post-run comparison in `dvc exp show`.

## CI/CD Integration

```yaml
# .github/workflows/train.yml
name: Train and Evaluate

on:
  push:
    paths:
      - "src/**"
      - "params.yaml"

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: iterative/setup-dvc@v1

      - name: Pull data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: dvc pull

      - name: Reproduce pipeline
        run: dvc repro

      - name: Push results
        run: dvc push

      - name: Report metrics
        run: dvc metrics show
```

DVC integrates with the [Iterative Studio](https://studio.iterative.ai/) web interface for team-wide experiment tracking and model registry.

## DVC vs Other Experiment Tracking Tools

| Feature | DVC | MLflow | Weights & Biases | Aim |
| --- | --- | --- | --- | --- |
| Data versioning | Yes | No | No | No |
| Pipeline DAG | Yes | No | No | No |
| Git integration | Deep | Partial | Partial | No |
| Experiment tracking | Yes | Yes | Yes | Yes |
| Remote storage | S3/GCS/Azure/SSH | S3/Azure | Cloud only | Self-host |
| Self-hostable | Yes | Yes | No | Yes |
| UI | Iterative Studio | MLflow UI | W&B UI | Aim UI |
| Code-first | Yes | Yes | Yes | Yes |

DVC's primary differentiator is **data versioning tied to Git history** — a capability none of the experiment tracking tools provide. For teams where reproducibility of training data is critical (regulated industries, research), DVC is the natural choice. For pure experiment tracking without data versioning concerns, MLflow or W&B are simpler.

## Common Workflows

### Onboarding a Collaborator

```bash
git clone https://github.com/org/ml-project
cd ml-project
pip install -r requirements.txt

# Configure remote (credentials from env vars or cloud CLI)
dvc remote modify myremote access_key_id "$AWS_ACCESS_KEY_ID"

# Pull all data at the current commit
dvc pull

# Run the full pipeline
dvc repro
```

### Switching Experiments

```bash
# Checkout a historical experiment from Git
git checkout <experiment-commit>
dvc checkout    # Swap data files to match the commit's .dvc files

# Return to latest
git checkout main
dvc checkout
```

## Summary

DVC brings version control discipline to ML data and pipelines:

- **`dvc add`** creates Git-trackable `.dvc` pointers for large files, storing actual data in remote storage
- **`dvc.yaml` pipelines** define reproducible DAG workflows; `dvc repro` re-runs only changed stages
- **`params.yaml`** integrates hyperparameter tracking with pipeline dependency management
- **`dvc exp run`** enables lightweight experiment sweeps without Git commits; `dvc exp show` compares results
- **DVCLive** provides real-time metric logging integrated with the DVC experiment system

By treating data as a first-class version-controlled artifact alongside code, DVC enables reproducible ML development across teams, environments, and time — closing the gap between ad hoc notebook experiments and production-ready workflows.
