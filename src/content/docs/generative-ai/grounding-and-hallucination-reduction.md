---
title: "Grounding LLMs: Reducing Hallucinations in Production"
description: "Practical techniques for reducing LLM hallucinations — from retrieval augmentation and citation enforcement to self-consistency checks and uncertainty quantification."
---

Hallucination — generating plausible-sounding but factually incorrect content — is one of the most significant reliability challenges in deploying LLMs. This guide covers the practical techniques that actually reduce hallucinations in production systems.

## Why LLMs Hallucinate

LLMs are trained to produce fluent, coherent text that fits the context. They don't have a separate "fact-checking" module — they generate tokens based on learned statistical patterns. When the model doesn't know something, it often generates something plausible rather than admitting uncertainty.

Key causes:
- **Knowledge gaps**: Information not in training data or outdated.
- **Conflicting training signals**: Contradictory information in training data.
- **Sycophancy**: Models trained with RLHF learn to agree with users, even when users are wrong.
- **Overconfidence**: Models don't reliably know what they don't know.

## Technique 1: Retrieval-Augmented Generation (RAG)

The most effective general-purpose solution. Instead of relying on parametric memory, retrieve relevant documents and include them in the context. The model is instructed to answer only from the provided documents.

This shifts the problem from "does the model know this?" to "is the right document in the retrieved set?" — a much more tractable engineering problem.

**Key implementation details**:
- Instruct the model explicitly: "Answer only using the provided documents. If the answer is not in the documents, say so."
- Include source citations in the prompt format to encourage attribution.
- Use re-ranking to ensure the most relevant chunks are in the context.

## Technique 2: Citation Enforcement

Require the model to cite specific sources for every factual claim. This forces the model to ground its statements in retrieved content and makes hallucinations detectable (a citation that doesn't support the claim).

```
Answer the question using the documents below. For each factual claim, 
include a citation in the format [Doc N]. If you cannot find support 
for a claim in the documents, do not make it.
```

## Technique 3: Self-Consistency and Verification

Generate multiple independent answers to the same question and check for consistency. Inconsistent answers signal uncertainty. For high-stakes claims, use a separate verification step:

1. Generate an answer.
2. Extract the key factual claims.
3. For each claim, ask the model: "Is this claim supported by the provided documents? Quote the supporting text."

## Technique 4: Uncertainty Elicitation

Prompt the model to express its confidence:

```
Answer the question. If you are uncertain about any part of your answer, 
explicitly flag it with [UNCERTAIN] and explain why.
```

Models aren't perfectly calibrated, but this prompting strategy does reduce confident hallucinations in practice.

## Technique 5: Constrained Generation

For structured outputs (extracting entities, filling forms), use constrained decoding or schema enforcement. If the model can only output values from a predefined set, it can't hallucinate values outside that set.

## Technique 6: Smaller, Focused Models

Counterintuitively, a smaller model fine-tuned on your specific domain often hallucinates less than a large general model. The fine-tuned model has stronger priors about what's true in your domain and is less likely to confabulate.

## Monitoring for Hallucinations in Production

- **LLM-as-judge**: Use a separate model to evaluate whether responses are grounded in the provided context.
- **Faithfulness metrics**: RAGAS faithfulness score measures whether claims in the response are supported by retrieved documents.
- **Human sampling**: Regularly sample and manually review a fraction of responses.

No technique eliminates hallucinations entirely. Defense in depth — combining multiple techniques — is the right approach for high-stakes applications.
