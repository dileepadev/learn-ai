---
title: "Evaluation: Measuring AI Performance"
description: "Why LLM evaluation is the hardest part of AI development and the strategies used to solve it."
---

"I poked it and it looked okay" is no longer a viable strategy for building AI applications. As models move into production, rigorous evaluation (Eval) becomes the most critical part of the development lifecycle.

## Why Eval is Hard

Unlike traditional software where outputs are predictable, LLM outputs are:

- **Non-deterministic:** The same input can produce different outputs.
- **Unstructured:** Free-form text is harder to validate than a JSON response.
- **Subjective:** "Quality" or "Tone" are difficult to measure mathematically.

## Types of AI Evaluation

1. **Deterministic Tests:** Checking for specific keywords, valid JSON format, or exact matches in code.
2. **Model-Based Evals (LLM-as-a-Judge):** Using a highly capable model (like GPT-4o) to grade the outputs of a smaller model based on a rubric.
3. **Benchmarks:** Standardized tests like MMLU (knowledge), GSM8K (math), or HumanEval (coding).
4. **Human Evaluation:** The gold standard, but the slowest and most expensive.

## The Golden Dataset

The key to good evaluation is building a "Golden Dataset"—a curated list of inputs and their "correct" or "ideal" outputs. This allows developers to measure "Regression" (knowing if a change to a prompt actually made the model worse in some areas).

## Metrics to Watch

- **Faithfulness:** Does the answer actually come from the provided context? (Crucial for RAG).
- **Relevance:** Does the answer actually address the user's question?
- **Perplexity:** A mathematical measure of how "confused" the model is by the text.
