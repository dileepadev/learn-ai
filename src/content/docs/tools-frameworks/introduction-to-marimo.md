---
title: Introduction to Marimo - Reactive Python Notebooks
description: Learn how marimo's reactive execution model, pure-Python file format, and built-in UI elements address the reproducibility problems of traditional notebooks.
---

Marimo is an open-source notebook environment for Python that treats a notebook as a reactive program rather than a collection of manually run cells. It was designed to fix long-standing Jupyter pain points: hidden state, out-of-order execution, and notebooks that cannot be version-controlled cleanly.

## Reactive Execution

Marimo builds a dependency graph from the variables each cell defines and reads. When you change a cell, every cell that depends on it re-runs automatically, and stale outputs cannot linger.

```python
import marimo as mo

threshold = mo.ui.slider(0, 1, step=0.05, value=0.5)
threshold
```

```python
# This cell re-runs whenever the slider changes
preds = (scores > threshold.value)
preds.mean()
```

A variable can be defined in only one cell, which removes a major source of hidden-state bugs.

## Notebooks as Python Files

Notebooks are stored as plain `.py` files. This gives:

- Clean, readable diffs in Git
- Ability to run as a script: `python notebook.py`
- Easy testing and linting with standard tools

## Apps and Interactivity

The same notebook can be served as an interactive web app with `marimo run`, hiding code and exposing only UI elements. This makes it convenient for sharing model demos and data-exploration tools.

## Typical ML Uses

- Hyperparameter exploration with sliders and dropdowns
- Interactive dataset inspection
- Lightweight dashboards over model outputs
- Reproducible experiment write-ups

## Caveats

Reactivity can trigger expensive recomputation. Use caching decorators or disable automatic execution for heavy cells while developing.
