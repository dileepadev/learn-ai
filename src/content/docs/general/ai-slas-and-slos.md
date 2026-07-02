---
title: "AI SLAs and SLOs"
description: "How service-level objectives help teams run AI features with clear expectations for latency, quality, and reliability."
---

As AI features move into production, teams need a way to define what "reliable enough" means. That is where **SLOs** (service-level objectives) and **SLAs** (service-level agreements) become useful. They bring operational discipline to systems whose behavior is partly probabilistic.

## What to Measure

For AI applications, useful objectives often include:

- **Latency:** how quickly the system returns a response
- **Availability:** how often the feature is usable
- **Output validity:** whether the response matches the required schema or format
- **Task success:** whether the answer actually helps complete the user goal

## Why AI Needs More Than Uptime

A system can be technically available while still failing users. If a chatbot answers slowly, returns malformed JSON, or produces low-quality results, the experience is degraded even though the server never went down.

## The Practical Benefit

SLOs help teams decide when a release is good enough, when a regression is serious, and what tradeoffs are acceptable between cost, quality, and speed. They turn AI operations into something measurable instead of purely subjective.
