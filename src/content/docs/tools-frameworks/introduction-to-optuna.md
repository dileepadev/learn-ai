---
title: Introduction to Optuna
description: Learn how to use Optuna — an automatic hyperparameter optimization framework — to tune machine learning and deep learning models efficiently using Tree-structured Parzen Estimators, pruning, multi-objective optimization, and distributed search.
---

**Optuna** is an open-source automatic hyperparameter optimization (HPO) framework developed by Preferred Networks. Unlike grid search (which exhaustively evaluates all parameter combinations) or random search (which samples blindly), Optuna uses **Bayesian optimization** with the **Tree-structured Parzen Estimator (TPE)** algorithm to intelligently sample the most promising hyperparameter configurations — finding better solutions with fewer evaluations.

Optuna's design philosophy is **define-by-run**: the search space is defined imperatively within the objective function using `trial.suggest_*` calls, making it trivial to define complex conditional and nested search spaces. Combined with built-in integration for PyTorch, scikit-learn, XGBoost, LightGBM, and other frameworks, Optuna is one of the most versatile HPO tools available.

## Installation

```bash
pip install optuna

# Optional: visualization dependencies
pip install optuna-dashboard plotly kaleido
```

## Core Concepts

### Study and Trial

- **Study**: An optimization session — a collection of trials directed toward optimizing an objective.
- **Trial**: A single evaluation of the objective function with one set of hyperparameter values.
- **Objective function**: A Python function that takes a `Trial` object, defines the search space, trains the model, and returns the metric to optimize.

## Basic Usage

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    # 5-fold cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Create a study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=300)  # 100 trials or 5 minutes

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Suggest Methods

Optuna provides typed suggest methods that define the search space:

```python
def objective(trial):
    # Integer — uniform over [low, high]
    n_layers = trial.suggest_int("n_layers", 1, 5)
    
    # Float — uniform or log-uniform
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Categorical — one of a list of choices
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"])
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "tanh"])
    
    # Conditional parameters (only relevant when condition is met)
    if optimizer == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
    
    # Nested search spaces
    hidden_sizes = [
        trial.suggest_int(f"hidden_{i}", 32, 512, log=True)
        for i in range(n_layers)
    ]
    
    return train_and_evaluate(n_layers, learning_rate, dropout, optimizer, hidden_sizes)
```

Log-uniform sampling (by setting `log=True`) is the right choice for learning rates, regularization coefficients, and other parameters that span orders of magnitude.

## PyTorch Integration with Pruning

**Pruning** (early stopping of unpromising trials) dramatically accelerates optimization by terminating trials that are clearly underperforming early in training. Optuna provides pruning callbacks for PyTorch, Keras, and other frameworks:

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    layers = [nn.Linear(784, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_size, 10))
    
    return nn.Sequential(*layers)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    model = create_model(trial)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(20):
        # Training loop
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        val_accuracy = evaluate(model, val_loader)
        
        # Report intermediate value for pruning decision
        trial.report(val_accuracy, epoch)
        
        # Prune trial if it's clearly worse than others at this epoch
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_accuracy

# Use Hyperband pruner for aggressive early stopping
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=3,
        max_resource=20,
        reduction_factor=3
    )
)
study.optimize(objective, n_trials=200)
```

Pruning with the **Hyperband** or **MedianPruner** strategy can reduce total compute by 5-10x compared to running all trials to completion.

## Sampler Algorithms

Optuna provides multiple sampling algorithms:

```python
# Tree-structured Parzen Estimator (default) — good for most tasks
study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=10))

# CMA-ES — effective for continuous, correlated parameter spaces
study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())

# Random search — baseline, no learning between trials
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())

# Grid search — exhaustive, requires explicit search space
search_space = {"lr": [1e-4, 1e-3, 1e-2], "batch_size": [32, 64, 128]}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))

# QMC (Quasi-Monte Carlo) — better coverage than random in early trials
study = optuna.create_study(sampler=optuna.samplers.QMCSampler())
```

TPE is the recommended default — it builds separate probability models for good and bad parameter configurations and samples from regions predicted to be good.

## Multi-Objective Optimization

Optuna supports simultaneous optimization of multiple objectives — finding the Pareto front of trade-offs:

```python
def multi_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 300)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    # Objective 2: model size (number of parameters) — minimize
    n_nodes = sum(estimator.tree_.node_count for estimator in model.estimators_)
    
    return accuracy, -n_nodes  # Maximize accuracy, minimize model size

# Multi-objective study with NSGA-II
study = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler()
)
study.optimize(multi_objective, n_trials=200)

# Get Pareto-optimal trials
pareto_front = study.best_trials
for trial in pareto_front[:5]:
    print(f"Accuracy: {trial.values[0]:.4f}, Size score: {trial.values[1]:.0f}")
```

## Distributed Optimization

Optuna scales to distributed HPO across multiple processes or machines via a shared database backend:

```python
# All workers share the same study via a PostgreSQL database
import optuna

storage = "postgresql://user:password@localhost/optuna_db"

# Worker 1 (and 2, 3, ...) — run on separate machines or processes
study = optuna.load_study(
    study_name="distributed_hpo",
    storage=storage
)
study.optimize(objective, n_trials=50)  # Each worker runs 50 trials

# Or create if not exists
study = optuna.create_study(
    study_name="distributed_hpo",
    storage=storage,
    load_if_exists=True,
    direction="maximize"
)
```

SQLite works for single-machine parallelism (`sqlite:///optuna.db`); PostgreSQL or MySQL is recommended for true multi-machine distributed search.

## Visualization and Analysis

```python
import optuna.visualization as viz

# Optimization history
fig = viz.plot_optimization_history(study)
fig.show()

# Parameter importance — which hyperparameters matter most?
fig = viz.plot_param_importances(study)
fig.show()

# Parameter interactions — contour plot for two parameters
fig = viz.plot_contour(study, params=["lr", "n_layers"])
fig.show()

# Parallel coordinate plot — see patterns across all parameters
fig = viz.plot_parallel_coordinate(study)
fig.show()
```

Parameter importance plots (using functional ANOVA or Shapley values) identify which hyperparameters most affect model performance — guiding where to focus manual tuning effort.

## Integration with MLflow

```python
import mlflow
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        result = train_and_evaluate(lr)
        mlflow.log_metric("accuracy", result)
    
    return result

with mlflow.start_run(run_name="optuna_hpo"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)
```

Optuna + MLflow provides end-to-end experiment tracking — every trial is logged with its parameters and metrics, and the best configuration is persisted for reproducibility.

## When to Use Optuna

| Scenario | Recommendation |
|----------|----------------|
| < 5 hyperparameters, small grid | Grid search is fine |
| 5-15 hyperparameters, moderate budget | Optuna TPE — significant gains over random |
| Deep learning, training is expensive | Optuna + Hyperband pruner — 5-10x speedup |
| Multi-objective (accuracy vs. latency) | Optuna NSGA-II |
| Large cluster available | Distributed Optuna with shared DB |
| Need reproducible results | Set `seed` in `TPESampler` |

Optuna's combination of intelligent sampling, pruning, and a clean Python API makes it the practical default for hyperparameter optimization across the full range of ML workflows — from scikit-learn models to large-scale deep learning.
