---
title: "ML System Design: From Prototype to Production"
description: "A practical framework for designing machine learning systems that work reliably in production — covering data pipelines, training infrastructure, serving, monitoring, and feedback loops."
---

Building a model that works in a notebook is the easy part. Building an ML system that works reliably in production, at scale, over time, is a fundamentally different engineering challenge.

## The ML System Stack

A production ML system has several interconnected layers:

```
Data Sources → Feature Store → Training Pipeline → Model Registry
                                                         ↓
User Requests → Feature Serving → Inference Service → Monitoring
                                                         ↓
                                              Feedback Loop → Retraining
```

Each layer has its own failure modes and design considerations.

## Data Pipeline Design

**Offline (batch) pipelines** process historical data for training and batch inference. Key concerns:
- Idempotency: re-running the pipeline produces the same result.
- Backfilling: ability to reprocess historical data when logic changes.
- Data quality checks: schema validation, null rate monitoring, distribution drift detection.

**Online (streaming) pipelines** compute features in real time for low-latency inference. Key concerns:
- Feature freshness: how stale can features be?
- Training-serving skew: features computed differently offline vs. online.

## Feature Stores

A feature store solves the training-serving skew problem by providing a single source of truth for feature computation. Features are computed once and served consistently to both training jobs and inference services.

Popular options: Feast (open source), Tecton, Vertex AI Feature Store.

## Training Infrastructure

- **Experiment tracking**: Log hyperparameters, metrics, and artifacts for every run (MLflow, W&B).
- **Reproducibility**: Pin library versions, seed random number generators, version datasets.
- **Distributed training**: Data parallelism for large datasets; model parallelism for large models.
- **Compute management**: Spot/preemptible instances for cost savings; checkpointing for fault tolerance.

## Model Registry and Deployment

A model registry stores versioned model artifacts with metadata (training data version, metrics, lineage). Deployment patterns include:

- **Shadow mode**: New model runs alongside the old one; outputs are logged but not served.
- **A/B testing**: Traffic is split between model versions; metrics are compared.
- **Canary deployment**: New model receives a small percentage of traffic, gradually increasing.
- **Blue/green deployment**: Instant cutover with easy rollback.

## Inference Serving

Key decisions:
- **Batch vs. real-time**: Batch inference is cheaper; real-time is necessary for interactive applications.
- **Latency SLAs**: P50, P95, P99 latency targets drive infrastructure choices.
- **Scaling**: Horizontal scaling for stateless models; careful state management for models with KV caches.

## Monitoring and Observability

Monitor at three levels:
1. **Infrastructure**: CPU/GPU utilization, memory, latency, error rates.
2. **Data**: Input feature distributions, missing values, schema violations.
3. **Model**: Prediction distributions, confidence scores, business metrics.

**Data drift** (input distribution changes) and **concept drift** (relationship between inputs and outputs changes) are the most common causes of silent model degradation.

## The Feedback Loop

The most important and most neglected part of ML system design. How do you know if the model is working? How do you collect labels for retraining?

- **Explicit feedback**: User ratings, corrections, thumbs up/down.
- **Implicit feedback**: Click-through rates, conversion rates, session length.
- **Human-in-the-loop**: Sampling predictions for human review.

A system without a feedback loop will degrade silently over time.
