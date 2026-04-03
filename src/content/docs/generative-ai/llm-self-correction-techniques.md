---
title: "Self-Correction in LLMs: Teaching Models to Review Their Own Work"
description: "Explore the techniques used to help LLMs identify and fix their own errors during generation, from basic prompting to iterative refinement."
---

One of the most promising areas of AI research is **Self-Correction**—the ability for a model to evaluate its own output, identify mistakes (hallucinations, logic errors, or formatting issues), and fix them before the user sees the result.

## Why Models Fail Initially

LLMs are "next-token predictors." Once they start down a wrong path of reasoning, they often continue to hallucinate to remain consistent with their previous tokens. Self-correction breaks this cycle.

## Common Techniques

1. **Self-Reflection**: Asking the model to "List any errors in your previous response and provide an improved version."
2. **Multi-Model Debating**: Using one model to generate and a different, perhaps more specialized model (like a "critic") to provide feedback.
3. **Chain-of-Verification (CoVe)**: The model generates an answer, creates verify-questions to check its own facts, and then synthesizes a final verified response.

## Applications

- **Coding**: Running generated code in a sandbox, catching the error message, and feeding it back to the model for a fix.
- **Mathematics**: Double-checking each step of a calculation before providing the final answer.
- **Safety**: Filtering out biased or harmful content by having a secondary "guardrail" pass.

## The Bottleneck

Self-correction is computationally expensive as it requires multiple passes. However, as inference becomes faster and cheaper (e.g., via Groq or Speculative Decoding), autonomous refinement will become the default behavior.
