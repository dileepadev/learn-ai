---
title: Introduction to Great Expectations
description: Learn how Great Expectations defines data-quality contracts, validates datasets, and documents pipeline assumptions.
---

Great Expectations is an open-source framework for expressing and checking assumptions about data. Instead of discovering malformed data after a training run or production incident, teams encode expectations close to their pipelines.

## Expectations as Contracts

An expectation is a testable statement about a dataset:

```python
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_values_to_be_in_set(
    "country_code", value_set=["IN", "US", "GB"]
)
```

These checks can cover schema, completeness, uniqueness, ranges, categories, and distributional properties.

## Why Data Validation Matters

ML systems often fail quietly when upstream data changes: a unit changes from dollars to cents, a feature becomes null, a join duplicates records, or a target leaks future information. A model can still produce a number even when the input is invalid.

## A Practical Workflow

1. profile representative data and identify assumptions
2. write an expectation suite that captures those assumptions
3. validate each input batch before it reaches training or inference
4. store validation results with data and model versions
5. fail, quarantine, or escalate according to the severity of the violation

Avoid copying every auto-generated expectation into a pipeline. Focus on properties that protect real decisions and that have a clear owner when they fail.

## Limits

Passing data checks does not prove model quality, fairness, or causal validity. Great Expectations complements model evaluation and monitoring; it does not replace them. Treat quality checks as a living contract and update them intentionally when source systems evolve.

