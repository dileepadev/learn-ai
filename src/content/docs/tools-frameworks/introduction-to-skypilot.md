---
title: Introduction to SkyPilot - Cloud-Agnostic GPU Job Orchestration
description: Learn how SkyPilot runs ML training and batch jobs across multiple clouds, automatically finding the cheapest available GPU capacity.
---

SkyPilot is an open-source framework for running machine learning jobs — training, batch inference, hyperparameter sweeps — across multiple cloud providers without manually provisioning and tearing down cloud infrastructure for each job.

## Declarative Job Specification

A SkyPilot task is defined in a YAML file describing the resources needed and the commands to run, independent of any specific cloud provider's particular instance types or APIs:

```yaml
resources:
  accelerators: A100:8
  cloud: any

setup: |
  pip install -r requirements.txt

run: |
  python train.py --config configs/large_run.yaml
```

Running `sky launch train.yaml` provisions matching GPU capacity on whichever configured cloud (AWS, GCP, Azure, and others) currently has it available at the best price, runs the job, and can automatically tear down the instance afterward to avoid paying for idle compute.

## Spot Instance Management

SkyPilot supports automatically using discounted spot/preemptible instances for training jobs, along with automatic checkpointing and recovery when a spot instance is reclaimed by the cloud provider — a meaningful cost saving for long training runs, provided the training code checkpoints frequently enough that a preemption doesn't lose substantial progress.

## Multi-Cloud Cost Optimization

Because GPU availability and pricing fluctuate significantly across regions and providers, especially for high-demand accelerators, SkyPilot's ability to search across configured clouds for the cheapest available matching capacity can meaningfully reduce both cost and the time spent waiting for capacity on a single, potentially GPU-constrained provider.

## Practical Guidance

Use SkyPilot for training and batch workloads where job portability across clouds and automatic cost optimization matter more than staying within a single cloud's native tooling — this is especially valuable for research teams and startups without negotiated capacity reservations with one specific provider. For always-on production inference serving rather than discrete training or batch jobs, a dedicated serving platform is generally a better fit than a job-orchestration tool built around ephemeral compute.
