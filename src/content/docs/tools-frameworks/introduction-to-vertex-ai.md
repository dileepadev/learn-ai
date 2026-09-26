---
title: Introduction to Google Vertex AI
description: Learn how Vertex AI unifies model training, tuning, deployment, and Google's Gemini models into one managed ML platform on Google Cloud.
---

Vertex AI is Google Cloud's unified machine learning platform, covering the full lifecycle from custom model training and fine-tuning to deploying both custom models and Google's own foundation models like Gemini behind managed endpoints.

## Accessing Foundation Models

```python
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

aiplatform.init(project="my-project", location="us-central1")
model = GenerativeModel("gemini-2.5-pro")

response = model.generate_content("Draft a data retention policy summary.")
```

Because Vertex AI runs inside a Google Cloud project, generated content and any custom training data stay within the project's existing IAM permissions, VPC Service Controls, and audit logging — the same appeal as AWS Bedrock's integration story, but for teams already standardized on Google Cloud.

## Model Garden and Fine-Tuning

Vertex AI's Model Garden catalogs both Google's proprietary models and a curated set of open-weight models available for deployment or fine-tuning on managed infrastructure, and Vertex AI's tuning pipelines support supervised fine-tuning and reinforcement learning from human feedback workflows on top of several of these base models without managing training infrastructure directly.

## Vertex AI Search and RAG Engine

Vertex AI Search provides managed retrieval infrastructure — document ingestion, chunking, embedding, and search — as a packaged service, and Vertex AI's RAG Engine gives a more configurable managed RAG pipeline for teams that want retrieval-augmented generation without assembling and operating the underlying vector infrastructure themselves.

## MLOps Tooling Beyond LLMs

Distinct from most LLM-provider platforms, Vertex AI also supports classic ML workflows end-to-end: custom model training on managed compute, a feature store, model monitoring for drift, and pipeline orchestration (Vertex AI Pipelines, built on Kubeflow Pipelines) — relevant for organizations running both traditional ML models and LLM-based applications on one platform.

## Practical Guidance

Choose Vertex AI when you want foundation models and classic ML infrastructure managed together inside Google Cloud's compliance boundary, particularly if you're already running other ML workloads there. For teams whose primary need is just calling Gemini without the surrounding MLOps platform, the standalone Gemini API is a lighter-weight entry point with less initial setup.
