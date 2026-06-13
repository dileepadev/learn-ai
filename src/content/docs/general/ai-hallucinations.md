---
title: AI Hallucinations
description: Understanding why large language models hallucinate, how to detect hallucinations, and practical strategies for reducing them.
---

AI hallucinations occur when a large language model generates content that is factually incorrect, fabricated, or not grounded in the provided context — but presents it with apparent confidence. Hallucinations are one of the most critical challenges in deploying LLMs for reliable applications.

## What Is a Hallucination?

A hallucination is any output where the model states something false or unsupported as if it were true. Examples include:

- Citing a paper that doesn't exist.
- Stating incorrect statistics, dates, or names.
- Inventing a legal case, product, or historical event.
- Adding details to a summary that weren't in the source document.
- Generating plausible-sounding but incorrect code or SQL.

## Why Do LLMs Hallucinate?

Hallucinations arise from fundamental aspects of how LLMs are trained:

1. **Probabilistic text completion:** LLMs predict the next most likely token. When they don't "know" the answer, they still generate fluent-sounding text based on statistical patterns — which can be wrong.

2. **Training data gaps:** Models may have encountered little or no reliable information about a specific topic, so they interpolate from related patterns.

3. **Training on noisy data:** The internet contains misinformation, and models may have learned incorrect facts alongside correct ones.

4. **RLHF pressure to be helpful:** Reinforcement Learning from Human Feedback rewards models for sounding confident and helpful, which can conflict with epistemic humility.

5. **No grounding mechanism by default:** Standard LLMs have no built-in way to verify claims against a knowledge source.

## Types of Hallucinations

- **Factual hallucinations:** Incorrect facts presented as truth (wrong dates, people, events).
- **Faithfulness hallucinations:** The response contradicts or goes beyond the provided source text (common in summarization).
- **Grounding failures:** The model answers a question about a document using its own training knowledge instead of the document.
- **Reasoning errors:** The model's chain of reasoning is flawed, leading to a wrong conclusion.

## Detecting Hallucinations

Detection is hard because hallucinations are often fluent and plausible. Approaches include:

- **Human review:** Experts check factual claims. High quality but expensive and slow.
- **Self-consistency:** Ask the model the same question multiple times and flag inconsistent answers.
- **Reference-based evaluation:** Compare model output against a ground truth document.
- **LLM-as-judge:** Use a second LLM (or the same one) to evaluate factual consistency.
- **Retrieval grounding:** Check whether each claim in the response is supported by retrieved documents.
- Tools like **Ragas**, **TruLens**, and **Evidently** provide automated hallucination metrics.

## Reducing Hallucinations

### Retrieval-Augmented Generation (RAG)
Ground the model in authoritative external sources. Ask the model to answer only based on the retrieved context and to state when it cannot find an answer in the sources.

### Better Prompting
- Instruct the model to say "I don't know" when uncertain.
- Ask for citations or evidence alongside claims.
- Decompose complex questions into verifiable sub-questions.

### Temperature and Sampling
Lower temperature makes outputs more deterministic, which can reduce hallucinations on factual tasks but may reduce creativity.

### Fine-Tuning
Fine-tuning on high-quality, domain-specific data can reduce hallucinations in that domain.

### Output Validation
Post-process model outputs with fact-checking tools, structured output parsers, or secondary verification models.

## The Reality: Hallucinations Cannot Be Eliminated

Current LLMs will continue to hallucinate to some degree. The goal is not zero hallucinations but **appropriate reliability** for the use case:
- High-stakes applications (medical, legal, financial) require aggressive grounding and human review.
- Lower-stakes applications (brainstorming, drafting) can tolerate more model autonomy.

Building systems that acknowledge uncertainty, cite sources, and fail gracefully is more realistic and more useful than expecting perfect factual accuracy.
