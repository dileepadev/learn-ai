---
title: "LLM Evaluation Benchmarks: A Practical Guide"
description: "Navigate the landscape of LLM benchmarks — from MMLU and HumanEval to GPQA and LiveBench — understanding what each measures, their limitations, and how to choose the right ones."
---

Benchmarks are how the field measures progress. But with dozens of benchmarks in common use, understanding what each actually measures — and where each falls short — is essential for making informed decisions about model selection and evaluation.

## Why Benchmarks Are Hard

A good benchmark should be:
- **Valid**: Actually measuring the capability it claims to measure.
- **Reliable**: Producing consistent results across runs.
- **Contamination-free**: Not present in the training data of models being evaluated.
- **Discriminative**: Separating models of different capability levels.

In practice, most benchmarks fail at least one of these criteria for at least some models.

## Core Knowledge and Reasoning Benchmarks

### MMLU (Massive Multitask Language Understanding)
57 subjects from elementary math to professional law. Tests broad knowledge recall. Widely used but heavily contaminated in recent models — many frontier models score 85–90%+, reducing its discriminative power.

### GPQA (Graduate-Level Google-Proof Q&A)
PhD-level questions in biology, chemistry, and physics that are designed to be unsearchable. Much harder than MMLU. Human experts score ~65%; frontier models now exceed this.

### ARC-Challenge
Grade-school science questions requiring reasoning beyond simple recall. Still useful for comparing smaller models.

## Coding Benchmarks

### HumanEval
164 Python programming problems with unit tests. The original coding benchmark. Now largely saturated — top models score 90%+.

### MBPP (Mostly Basic Python Problems)
500 crowd-sourced Python problems. Similar saturation issues to HumanEval.

### SWE-bench
Real GitHub issues from popular Python repositories. The model must generate a patch that passes the existing test suite. Much harder and more realistic than HumanEval. SWE-bench Verified is the curated subset with confirmed correct solutions.

### LiveCodeBench
Continuously updated with new competitive programming problems, reducing contamination risk.

## Math Benchmarks

### GSM8K
Grade-school math word problems. Saturated for frontier models (99%+).

### MATH
Competition mathematics (AMC, AIME level). Still discriminative for frontier models.

### AIME
American Invitational Mathematics Examination problems. Extremely hard; used to evaluate o1/o3-class models.

## Instruction Following and Alignment

### MT-Bench
Multi-turn conversation quality, judged by GPT-4. Useful but subject to judge model bias.

### IFEval
Tests precise instruction following (e.g., "write exactly 3 paragraphs"). Objective and hard to game.

### AlpacaEval
Pairwise comparison against a reference model, judged by an LLM. Fast but biased toward verbose responses.

## Contamination and the Benchmark Treadmill

As models are trained on more internet data, benchmark contamination becomes a serious problem. Solutions include:

- **Dynamic benchmarks** (LiveBench, LiveCodeBench): Continuously add new problems.
- **Private test sets**: Held-out evaluation sets not released publicly.
- **Contamination detection**: Checking for n-gram overlap between training data and benchmarks.

## Practical Advice

Don't rely on a single benchmark. Use a portfolio:
- GPQA for knowledge depth.
- SWE-bench for real-world coding.
- MATH/AIME for reasoning.
- IFEval for instruction following.
- A domain-specific benchmark relevant to your use case.

And always evaluate on your own data — benchmark performance and production performance often diverge significantly.
