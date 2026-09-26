---
title: Introduction to Prefect - Python-Native Workflow Orchestration
description: Learn how Prefect orchestrates data and ML pipelines with plain Python functions, dynamic flows, and built-in retries and observability.
---

Prefect is a workflow orchestration tool for building, scheduling, and monitoring data and ML pipelines using plain Python functions rather than a separate DSL or config format.

## Flows and Tasks

A Prefect pipeline is built from flows (the overall workflow) composed of tasks (individual units of work), defined with simple decorators:

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=10)
def extract_data():
    return fetch_from_api()

@task
def transform(data):
    return clean(data)

@flow
def etl_pipeline():
    raw = extract_data()
    transform(raw)
```

Because flows are ordinary Python functions, you can use standard control flow — loops, conditionals, dynamic task creation based on runtime data — without needing a static DAG defined up front, unlike orchestration tools that require the full task graph to be known before execution starts.

## Why Prefect for AI Pipelines

AI and ML pipelines often need retries around flaky external calls (LLM APIs, vector database writes), scheduled batch jobs (nightly re-embedding of new documents), and observability into which step failed and why. Prefect provides retries, caching, and a UI showing flow run history and logs out of the box, without requiring a Kubernetes cluster or a heavyweight scheduler to get started — a single Python process can run flows locally, and the same code can later be deployed to distributed infrastructure.

## Prefect vs. Airflow

Airflow requires defining a static DAG of tasks ahead of time and historically had a steeper operational footprint (a dedicated scheduler, webserver, and metadata database). Prefect's dynamic, Python-native flow definition and lighter deployment model make it a common choice for teams that want orchestration without adopting Airflow's full operational surface, particularly for newer ML and LLM pipeline teams starting from scratch.

## Practical Guidance

Start with local flow runs during development, then move to Prefect Cloud or a self-hosted server only once you need scheduling, team visibility, or distributed execution. Wrap external API calls (embedding models, LLM providers, vector stores) as tasks with retries and reasonable timeouts, since these are the most common failure points in AI pipelines.
