---
title: "Prompt Regression Testing"
description: "Why every prompt change should be tested against a stable set of examples."
---

Prompt regression testing checks whether a new prompt version improves the behavior you want without breaking behavior you already had. Because prompt changes are easy to make and hard to reason about, regression tests are essential.

## What to Test

- Factual accuracy on known examples
- Output structure and schema adherence
- Refusal behavior for unsafe prompts
- Tone, brevity, and instruction following
- Edge cases that have failed before

## Building a Prompt Test Set

A useful test set contains both common cases and tricky ones. Good teams save real failures from production and turn them into regression examples so the same mistake is less likely to return later.

## Why It Works

Regression testing gives prompt work a feedback loop. Instead of relying on intuition, you can measure whether the new wording actually made the system better.
