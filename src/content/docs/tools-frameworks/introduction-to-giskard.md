---
title: Introduction to Giskard
description: Learn how Giskard helps test ML and LLM applications for quality regressions, vulnerabilities, and failure patterns.
---

Giskard is an open-source quality-assurance platform for machine-learning and LLM applications. It helps teams turn known requirements and discovered weaknesses into repeatable tests, then run those tests before a release.

## What to Test

An AI application needs more than a single accuracy score. Useful tests cover:

- factual correctness and grounded answers
- structured-output validity
- robustness to spelling, phrasing, and distribution shifts
- prompt-injection and data-exposure attempts
- performance across relevant user or data slices

The appropriate suite depends on the system’s intended use and the harm caused by an error.

## Test-Driven AI Quality

Start by writing an explicit behavior that must hold:

```text
Given: a support question and the approved policy document
Expect: an answer grounded in that document
Reject: invented policy details or unrelated instructions in the document
```

Representative examples make this behavior executable. When a real failure occurs, add a minimized, privacy-safe version to the test set so it cannot silently return after a prompt or model change.

## Scanning and Human Review

Automated scans can propose weaknesses such as hallucination, harmful output, or injection susceptibility. These results are leads for investigation, not a complete security assessment. Review the examples, confirm the threat model, and implement controls in the application itself: authorization, trusted tool boundaries, output validation, and least privilege.

## Release Workflow

1. define critical behaviors and test data
2. run deterministic checks on every change
3. run model-graded, adversarial, and slice-based tests before release
4. compare results with the previous version
5. require an owner to approve or investigate meaningful regressions

Quality testing becomes durable when tests are versioned alongside prompts, models, data assumptions, and application code. The goal is not a perfect score; it is evidence that the system remains fit for its intended use.

