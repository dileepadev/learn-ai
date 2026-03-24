---
title: "A Practical Checklist for Evaluating LLM Outputs"
description: "A concise checklist to assess correctness, relevance, safety, and formatting of LLM responses."
date: "2026-03-24"
tags: ["evaluation", "llm", "quality-assurance"]
---

## 1. Correctness

- Verify factual claims against reliable sources when possible.
- Check numbers, dates, names, and any specific details for accuracy.

## 2. Relevance

- Ensure the response addresses the user's question and scope.
- Remove or rewrite extraneous information that could confuse the user.

## 3. Completeness

- Confirm the answer covers required subtopics and edge cases.
- If partial, ask the model to list what's missing or provide follow-ups.

## 4. Clarity & Format

- Ensure the output is readable: short paragraphs, headings, and lists where helpful.
- Enforce required formatting (JSON, CSV, bullet lists) with explicit constraints.

## 5. Safety & Bias

- Screen for harmful, biased, or disallowed content.
- If sensitive topics appear, prefer safe defaults and ask for human review.

## 6. Source & Attribution

- Ask the model to provide sources or cite evidence when making factual claims.
- Prefer verifiable references; treat unreferenced facts as unconfirmed.

## 7. Reproducibility

- Record prompt, model settings (temperature, max tokens), and any few-shot examples used.
- Use variations to test stability of important outputs.

## 8. Actionability

- If the output includes instructions or code, run a quick validation or linting step.
- For code, prefer minimal reproducible examples and unit tests where feasible.

## Quick workflow

1. Run the model and capture the response.
2. Apply this checklist top-to-bottom.
3. If any item fails, refine the prompt and re-run.
4. Request a human review for high-risk or high-impact outputs.

## Next steps

- Add this checklist as part of your review process for production RAG and automation flows.
- Automate lightweight checks (format, JSON validity, banned words) where possible.

References

- Keep a running list of authoritative sources and internal validation steps tailored to your domain.
