---
title: Introduction to DeepEval
description: Learn how DeepEval brings test cases, metrics, and regression checks to LLM application evaluation.
---

DeepEval is an open-source Python framework for testing LLM applications. It lets teams define test cases, apply task-specific metrics, and run evaluations in local development or continuous integration.

## Evaluation as a Test Suite

An LLM test should include the user input, actual output, and the evidence needed to judge it:

```python
from deepeval.test_case import LLMTestCase

case = LLMTestCase(
    input="What is the refund period?",
    actual_output="You can request a refund within 30 days.",
    retrieval_context=["Refunds are available within 30 days of purchase."],
)
```

This makes an expectation explicit and repeatable rather than relying on a one-off chat inspection.

## Useful Metrics

DeepEval provides metrics for answer relevance, faithfulness to retrieval context, contextual precision, toxicity, and task-specific criteria. A retrieval-grounded application may require both:

- **faithfulness:** the answer is supported by retrieved evidence
- **answer relevance:** the answer directly addresses the user’s question

No single score captures application quality. Combine deterministic checks for schemas and policies with model-judged metrics for semantic quality, and sample results for human review.

## A Practical Testing Pattern

1. create a compact set of representative and adversarial cases
2. include known production failures after removing sensitive data
3. set thresholds based on observed human judgment
4. run the suite on prompt, model, retrieval, and tool changes
5. investigate regressions with traces and example outputs

Avoid treating a score threshold as a release guarantee. Judge models can be inconsistent and can miss domain-specific errors. Calibrate metrics against the real decision the application supports.

## CI and Data Handling

Run deterministic and inexpensive checks on every change, then run costlier model-graded suites on a suitable schedule or release gate. Keep API keys out of source control, minimize sensitive test data, and version the cases with the prompt and application code. An evaluation framework is valuable when it shortens the feedback loop while preserving meaningful quality standards.

