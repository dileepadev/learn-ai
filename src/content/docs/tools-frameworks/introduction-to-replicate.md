---
title: Introduction to Replicate - Run Any Open-Source Model via API
description: Learn how Replicate packages open-source models as callable API endpoints with automatic scaling and no infrastructure management.
---

Replicate lets developers run open-source machine learning models — image generation, video generation, speech, and language models — through a simple API call, without provisioning GPUs, managing dependencies, or building serving infrastructure.

## Basic Usage

```python
import replicate

output = replicate.run(
    "black-forest-labs/flux-schnell",
    input={"prompt": "a lighthouse at dawn, watercolor style"}
)
```

Each model on Replicate is packaged with Cog, an open-source tool that wraps a model and its dependencies into a standardized container with a defined input/output schema, which is what lets Replicate run an enormous variety of community-contributed models behind one consistent calling convention.

## Automatic Scaling and Cold Starts

Replicate scales model instances up and down automatically based on demand, including scaling to zero when a model isn't being used, which minimizes cost for intermittent workloads but introduces cold-start latency (loading model weights onto a fresh GPU instance) for the first request after an idle period. High-traffic production models can be configured to keep a minimum number of instances warm to avoid this latency at additional cost.

## Publishing Your Own Models

Beyond consuming public models, Replicate lets developers package and publish their own models using Cog, making a custom or fine-tuned model callable through the same API pattern as Replicate's public catalog, which is useful for sharing research models or deploying an internal fine-tune without building separate serving infrastructure from scratch.

## Practical Guidance

Replicate is particularly strong for quickly trying out or building on top of the latest open-source generative models (image, video, audio) without setting up GPU infrastructure yourself, which matters most during prototyping or for workloads with unpredictable, spiky traffic. For high-volume, latency-sensitive production use of a single well-known model, compare Replicate's per-second GPU pricing against self-hosting with vLLM or a dedicated inference platform, since dedicated infrastructure often becomes more cost-effective at sustained high volume.
