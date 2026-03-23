---
title: Prompt Evaluation Metrics
description: Overview of metrics and approaches for evaluating prompts and LLM outputs.
---

Evaluating prompts and model outputs requires a mix of automatic and human-centered metrics. This guide summarizes common metrics and when to use them.

## Automatic Metrics

- **Perplexity:** Measures how well a model predicts a sequence; useful for language modeling tasks.
- **BLEU / ROUGE:** N-gram overlap metrics for tasks with reference outputs (translation, summarization).
- **BERTScore / MoverScore:** Embedding-based similarity metrics that capture semantic similarity better than simple overlap.
- **Factuality Metrics:** Use QA-based checks or fact extraction to measure hallucination rates.

## Human-Centered Metrics

- **Helpfulness:** Human raters judge how useful the response is for the task.
- **Correctness / Accuracy:** Judges verify factual claims or task outputs.
- **Safety / Toxicity:** Assess whether outputs contain unsafe or biased content.
- **Fluency & Coherence:** Quality of language and logical flow.

## Best Practices

- Combine automatic metrics with targeted human evaluations for high-risk tasks.
- Use task-specific metrics where possible (e.g., exact match for QA, BLEU/ROUGE for summarization).
- Run A/B evaluations for prompt variants and collect both quantitative and qualitative feedback.

## Quick Checklist

- Define success criteria before evaluating prompts
- Use a mix of automated checks and human reviews
- Track metrics over time to detect regressions
