---
title: "Fine-tuning vs In-Context Learning"
description: "Compare fine-tuning models and using in-context examples (few-shot) with tradeoffs and use cases."
date: "2026-03-19"
tags: ["introduction", "llm", "fine-tuning", "few-shot"]
---

Large language models can be adapted to tasks either by fine-tuning the model weights or by prompting it with examples (in-context learning). Each approach has tradeoffs.

## Fine-tuning

- **What:** Update model weights on task-specific data.
- **Pros:** Often higher accuracy, reproducible behavior, and lower per-request cost after deployment.
- **Cons:** Requires data, compute, and a retraining pipeline; risk of catastrophic forgetting.

## In-context learning (few-shot)

- **What:** Provide a small set of examples in the prompt without changing model weights.
- **Pros:** Fast to iterate, no retraining required, good for low-data scenarios.
- **Cons:** Prompt length limitations, higher per-call cost, and more variable outputs.

## Choosing an approach

- If you need consistent high-quality outputs and have labeled data, consider fine-tuning.
- For rapid prototyping or when data is scarce, prefer in-context learning.

## Hybrid strategies

- Use prompt engineering for quick exploration, then fine-tune a smaller model for production.
- Consider parameter-efficient fine-tuning (LoRA, adapters) to reduce cost and preserve base model behavior.

Next steps: run a small A/B test comparing a tuned model vs few-shot prompts on your target metric.
