---
title: Evaluating AI Agents - Benchmarks and Methodology
description: Learn how agent benchmarks measure multi-step tool use, what makes agent evaluation harder than static LLM evaluation, and how to design reliable agent evals.
---

Evaluating an agent is harder than evaluating a single model response. Agents take multiple steps, call tools, change environment state, and may reach the same goal through many different paths.

## What Is Different About Agents

- **Trajectories**: success depends on a sequence of actions, not one output
- **State changes**: the environment (files, databases, web pages) is modified
- **Non-determinism**: runs vary, so one pass is a noisy measurement
- **Compounding errors**: a small mistake early can derail later steps
- **Cost and time**: each run may consume many model calls

## Well-Known Benchmark Categories

| Category | Example benchmarks | What it measures |
|----------|--------------------|------------------|
| Software engineering | SWE-bench | Resolve real GitHub issues, verified by tests |
| Web navigation | WebArena, Mind2Web | Complete tasks on real or simulated websites |
| Tool use / customer service | tau-bench | Follow policies while calling APIs with a simulated user |
| General assistant | GAIA | Multi-step questions needing browsing and tools |
| Computer use | OSWorld | Operate a desktop through screenshots and actions |
| Research / terminal | Terminal-style and deep-research suites | Long-horizon command-line or search tasks |

## Grading Approaches

1. **Outcome-based**: check final environment state (tests pass, record exists). Most reliable when possible.
2. **Trajectory-based**: score the steps taken, for efficiency or policy compliance.
3. **LLM-as-judge**: rate open-ended results against a rubric; calibrate it against human labels.

## Reliability Metrics

A single success rate hides inconsistency. Run each task several times and report:

```text
pass@k   : at least one of k attempts succeeds     (capability)
pass^k   : all k attempts succeed                  (reliability)
```

For production agents, pass^k matters more, because users expect consistent behavior.

## Designing Your Own Evals

1. Collect real failure cases and representative tasks from logs
2. Make each task self-contained, with a verifiable success condition
3. Reset the environment to a clean state per run
4. Track cost, latency, and number of steps alongside accuracy
5. Separate capability evals (hard, low pass rate) from regression evals (should stay near 100%)

## Pitfalls

- Benchmark contamination or leaked solutions in training data
- Flawed or underspecified tasks and graders that mark valid solutions wrong
- Agents exploiting the environment rather than solving the task
- Overfitting prompts to a public benchmark that does not match your use case

Read agent transcripts regularly; aggregate scores cannot tell you why an agent failed.
