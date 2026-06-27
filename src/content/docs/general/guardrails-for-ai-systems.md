---
title: "Guardrails for AI Systems"
description: "The rules, checks, and control layers that keep AI applications useful and safe."
---

Guardrails are the mechanisms that constrain what an AI system can say or do. They exist because even strong models can hallucinate, ignore instructions, or act unsafely when placed in real workflows.

## Types of Guardrails

1. **Input guardrails:** block harmful or out-of-scope requests.
2. **Generation guardrails:** use system prompts, schemas, or constrained decoding.
3. **Output guardrails:** scan responses for policy violations or invalid structure.
4. **Action guardrails:** require approval before risky tool use or irreversible actions.

## Why They Matter

Guardrails create defense in depth. If one layer fails, another can still prevent a harmful outcome. This is more reliable than trusting the model alone to behave correctly in every situation.

## A Useful Mindset

Think of guardrails as product architecture, not just safety add-ons. They improve reliability, consistency, and trust in addition to reducing harm.
