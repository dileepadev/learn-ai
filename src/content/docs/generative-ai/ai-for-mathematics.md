---
title: AI for Mathematics and Automated Theorem Proving
description: Explore how AI systems are tackling mathematical reasoning — from neural theorem provers and formal verification in Lean 4 to AlphaProof's olympiad-level solutions — and the open challenges of bridging intuitive and rigorous mathematical thinking.
---

Mathematics has long been considered a frontier uniquely resistant to automation. Unlike game-playing or image recognition, mathematical reasoning demands **logical rigor**, **creative insight**, and **multi-step deduction** over abstract objects — with zero tolerance for error. Yet in 2024–2025, AI systems crossed a threshold: for the first time, machine systems began solving competition-level mathematical problems at rates approaching or exceeding human expert performance, validated not by approximate benchmarks but by formal proof verifiers.

## The Two Paradigms: Informal and Formal

Mathematical AI operates in two fundamentally different modes:

### Informal Reasoning (Natural Language Math)

LLMs operate in natural language or mathematical notation (LaTeX, equations) and produce solutions as text. Correctness is verified by evaluating the final numerical answer against a ground truth. **The proof steps are not formally verified** — a model can produce a plausible-sounding but logically flawed derivation and still report the correct answer.

**Key benchmarks:**

| Benchmark | Description | Difficulty |
| --- | --- | --- |
| GSM8K | Grade-school arithmetic word problems | Easy |
| MATH | High-school competition mathematics | Medium |
| AIME | American Invitational Mathematics Examination | Hard |
| HMMT / USAMO | Harvard-MIT and US Math Olympiad | Expert |
| Putnam | Undergraduate competition | Expert |

State-of-the-art LLMs (GPT-4o, Gemini 1.5 Pro, Claude 3.5) achieve 90%+ on MATH and 50–80% on AIME, with performance improving rapidly as models are trained with more mathematical data and reasoning-focused RLHF.

### Formal Reasoning (Proof Assistants)

Proof assistants — **Lean 4**, **Coq**, **Isabelle**, **Agda** — are programming languages where every proof step is verified by a type checker. A "proof" that does not type-check is definitively wrong, regardless of how plausible it looks. This provides **machine-checkable ground truth** for mathematical correctness.

Formal proofs are verbose: a one-line informal argument may require hundreds of lines in Lean. But they provide an unambiguous correctness signal — exactly what AI training needs.

## AlphaProof and AlphaGeometry

DeepMind's **AlphaProof** (2024) is the most significant recent advance in formal mathematical AI:

- **Task:** Solve International Mathematical Olympiad (IMO) problems by producing formally verified Lean 4 proofs.
- **Architecture:** Combines an LLM-based proof search with reinforcement learning: the model generates candidate proof steps; a Lean verifier accepts or rejects each step; successful proofs become training data.
- **Result:** Solved 4 out of 6 problems at IMO 2024, achieving a score equivalent to a Silver Medal — the first AI result at this level for formal competition mathematics.

**AlphaGeometry** (2024) tackles Euclidean geometry problems specifically, using a combination of:

1. A **symbolic engine** that applies geometric inference rules exhaustively.
2. An LLM that proposes **auxiliary constructions** — adding new points or lines to the diagram — when the symbolic engine is stuck.

This neuro-symbolic architecture solved 25 out of 30 IMO geometry problems from 2000–2023, outperforming the previous record of 10 and exceeding the average Gold Medalist score on geometry.

## Neural Theorem Provers

Several architectures have been developed specifically for neural theorem proving:

### Tactic-Based Proof Search

Interactive theorem provers (Lean, Coq) expose a **tactic API**: a sequence of commands that reduce the current goal to simpler sub-goals. Proof search is a tree search problem:

- **State:** Current proof context (hypotheses + goal).
- **Action:** Apply a tactic (e.g., `rw`, `apply`, `simp`, `ring`, `norm_num`).
- **Reward:** $+1$ if the goal is closed (proof complete), $0$ otherwise.

Neural provers learn a policy $\pi_\theta(\text{tactic} | \text{context})$ and a value function $v_\theta(\text{context})$, then apply best-first search (or MCTS) over the tactic tree.

**Key systems:**

- **GPT-f** (OpenAI, 2020): First to show LLMs could generate useful tactics for Metamath and Lean.
- **Hypertree Proof Search** (Meta, 2022): Online learning with MCTS in Lean, setting records on the miniF2F benchmark.
- **COPRA** and **LeanDojo**: Open frameworks for training and evaluating tactic proof search.
- **DeepSeek-Prover**: Fine-tuned LLM achieving state-of-the-art on miniF2F and ProofNet.

### Proof by Autoformalization

A complementary approach: **translate informal mathematical proofs into formal proofs automatically** (autoformalization). The pipeline:

1. Express an informal proof in natural language or LaTeX.
2. Use an LLM to translate each step into Lean 4 syntax.
3. Run the Lean verifier to check each step; prompt the LLM to fix failures.

This mirrors the workflow of a human mathematician who sketches a proof informally then formalizes it — and provides a scalable path to creating large training datasets of formal proofs from existing informal mathematical literature.

## Large Language Models as Mathematical Reasoners

Beyond formal proving, LLMs have demonstrated striking informal mathematical capabilities:

### Chain-of-Thought and Self-Consistency

Mathematical reasoning improves substantially when models are prompted to show their work step-by-step (**chain-of-thought prompting**). **Self-consistency** — sampling multiple solutions and majority-voting on the final answer — further boosts accuracy by averaging out reasoning errors.

### Process Reward Models (PRMs) for Math

**PRMs** assign a score to each **step** of a mathematical solution, not just the final answer. A PRM trained on annotated correct and incorrect solution steps provides fine-grained signal for:

- **Training:** Reinforcing correct intermediate steps rather than only rewarding correct final answers.
- **Inference-time search:** Guiding MCTS or beam search to prefer solutions with high per-step confidence.

OpenAI's **Math-Shepherd** and **Let's Verify Step by Step** papers demonstrated that PRMs substantially outperform outcome-based reward models on mathematical benchmarks, especially for harder problems requiring 15+ steps.

### Tool Use and Code Execution

A powerful augmentation is allowing the model to **write and execute code** as an external tool:

1. Express the solution strategy in natural language.
2. Translate computational steps into Python.
3. Execute the code and incorporate results back into the solution.

This removes arithmetic and algebraic manipulation errors (which LLMs are prone to) from the critical path, reserving the model for higher-level reasoning. Systems combining LLMs with Python execution achieve near-perfect accuracy on competition-level arithmetic and algebraic computation.

## Benchmark Summary and Progress

| System | MATH | AIME 2024 | miniF2F (Formal) |
| --- | --- | --- | --- |
| GPT-4 (2023) | 52% | ~10% | ~30% |
| GPT-4o (2024) | 76% | 9/30 | — |
| DeepSeek-Prover | — | — | 64% (pass@1) |
| AlphaProof (2024) | — | IMO 2024: 4/6 | — |
| o1 / o3 (2025) | 90%+ | 27/30 | — |

The trajectory is steep: from 52% on MATH in 2023 to 90%+ in 2025, with olympiad-level formal proofs emerging simultaneously.

## Open Challenges

### The Formalization Bottleneck

Formal proof libraries (Mathlib in Lean 4) contain roughly 200,000 theorems — a tiny fraction of published mathematics. Training models require formal proofs; creating them at scale requires either massive human effort or reliable autoformalization. Closing this loop is a central challenge.

### Mathematical Creativity

Current AI systems excel at finding proofs within the space of known techniques. **Genuinely novel mathematical insights** — recognizing that two apparently unrelated fields are deeply connected, inventing new proof techniques, identifying the right abstraction for an unsolved problem — remain beyond current systems. These require structured exploration of conceptual space that is not captured by pattern matching over existing proofs.

### Long-Horizon Reasoning

Research-level mathematics requires weeks or months of work, maintaining a consistent mental model across many intermediate results. Current models have a finite context window and lack persistent, verifiable memory of intermediate lemmas they have proved. Agentic architectures with formal memory (a running Lean proof state) are an emerging approach.

### Generalization to New Mathematics

Current benchmarks test models on problems whose solutions exist in the training corpus (even if indirectly). Evaluating models on genuinely new, unpublished problems — and verifying that solutions do not leak from training data — is an open methodological challenge.

## Why Mathematical AI Matters

- **Scientific discovery:** Mathematical proofs underlie physics, cryptography, and algorithms. Automated provers accelerate discovery in any field that depends on formal reasoning.
- **Software verification:** Lean 4 is used to formally verify software and hardware. AI-assisted provers lower the cost of verification dramatically.
- **Education:** Detailed step-by-step mathematical explanations with error feedback are a powerful tutoring tool.
- **Foundations of AI safety:** Formal verification of AI system properties (safety invariants, reward model constraints) benefits from the same technology as mathematical theorem proving.

## Summary

AI for mathematics has advanced from solving arithmetic word problems to producing formally verified proofs for olympiad-level competition problems in a span of two years. The convergence of LLM-scale pretraining, process reward models, MCTS-guided proof search, and formal verification frameworks like Lean 4 has created a self-reinforcing capability loop. Key open problems — autoformalization at scale, genuine mathematical creativity, and long-horizon research assistance — define the next frontier, where AI transitions from solving known problems to contributing to the advancement of mathematical knowledge itself.
