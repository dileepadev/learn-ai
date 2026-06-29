---
title: "Synthetic Data Quality"
description: "Why generated data is only useful when teams measure diversity, realism, and task relevance."
---

Synthetic data can expand scarce datasets, protect privacy, and accelerate model development. But generated data is not automatically good data. Quality determines whether synthetic examples improve a model or quietly make it worse.

## What Quality Means

- **Relevance:** the examples match the task you care about
- **Diversity:** they cover meaningful variation instead of repeating the same pattern
- **Realism:** they resemble the structure of real-world inputs
- **Correctness:** labels, answers, or actions are accurate

## Main Risk

Low-quality synthetic data can create shortcut learning, amplify bias, or teach a model unrealistic patterns. This is especially risky when the synthetic set is much larger than the real one.

## Best Practice

Treat synthetic data as something to curate, filter, and evaluate carefully. It works best as a supplement to strong real-world data, not a blind replacement for it.
