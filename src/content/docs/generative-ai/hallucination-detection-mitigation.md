---
title: LLM Hallucination — Detection and Mitigation
description: Understand why large language models hallucinate, the taxonomy of hallucination types, how to detect them with automated and human evaluation, and the most effective mitigation strategies for production AI systems.
---

Hallucination in large language models refers to the generation of content that is **fluent and confident but factually incorrect, fabricated, or ungrounded**. It is one of the most critical practical challenges in deploying LLMs, threatening their usefulness in high-stakes domains like medicine, law, and finance.

## What Is Hallucination?

The term borrows from psychology — perceiving something not present in reality. In LLMs, hallucination takes several forms:

### Taxonomy of Hallucination Types

**Factuality Hallucinations**
- **Fabrication:** Inventing entities, events, or facts that don't exist (e.g., citing non-existent papers)
- **Factual error:** Stating a known fact incorrectly (e.g., wrong date, wrong statistic)
- **Outdated information:** Accurate at training time but since superseded

**Faithfulness Hallucinations** (in grounded tasks)
- **Intrinsic hallucination:** Contradicting information provided in the source context
- **Extrinsic hallucination:** Adding information not supported by — but not contradicting — the source
- **Context drift:** Starting faithful but drifting from the source over a long generation

**Instruction Hallucinations**
- Ignoring constraints in the prompt (e.g., "answer in 3 bullet points" → generates 7)
- Misidentifying what the user actually wants

## Why Do LLMs Hallucinate?

### Training-Time Causes
- **Memorization vs. generalization:** LLMs learn statistical patterns, not truth. Low-frequency facts are underrepresented and more likely to be confabulated
- **Exposure bias:** Teacher-forcing trains on correct prefixes; at inference time, errors compound
- **Conflicting training data:** Different sources contradict each other; the model averages conflicting signals
- **RLHF reward hacking:** Models fine-tuned to sound confident learn to generate confident-sounding outputs even when uncertain

### Inference-Time Causes
- **Decoding strategy:** Greedy decoding amplifies the most probable (but potentially wrong) completion; sampling adds noise
- **Context length limitations:** Long documents exceed the model's effective recall window
- **Out-of-distribution queries:** Questions about rare topics, recent events, or technical niches fall outside training distribution

## Detection Methods

### NLI-Based Faithfulness Checking
Natural Language Inference (NLI) models classify whether a generated claim is **entailed**, **contradicted**, or **neutral** with respect to a source document:

```python
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")
result = nli({
    "text": "The Eiffel Tower is 330 meters tall.",   # claim
    "text_pair": "The Eiffel Tower stands 300m high."  # source
})
# → {'label': 'CONTRADICTION', 'score': 0.94}
```

Tools like **TruLens**, **RAGAS**, and **LangChain's evaluation chains** use NLI-based faithfulness scores for RAG pipeline monitoring.

### Self-Consistency
Generate multiple independent answers to the same question (via temperature sampling) and check agreement:
- High variance across answers → likely hallucination zone
- Strong consensus → higher confidence (but not proof of accuracy)

### Reference-Free Detection
Models trained to detect hallucinations without access to external knowledge:
- **SelfCheckGPT:** Samples multiple outputs and measures internal consistency without external reference
- **HaluEval:** A benchmark and set of trained detectors for various hallucination types
- **Lynx / FACTOID:** Specialized faithfulness detectors fine-tuned to flag RAG hallucinations

### Semantic Entropy
Farquhar et al. (2024) propose measuring the entropy over the **semantic meaning** of multiple sampled generations (rather than surface form). High semantic entropy indicates the model's actual uncertainty about the answer.

### Citation Verification
For models that produce citations to sources:
1. Retrieve the cited source
2. Run NLI to verify the claim is actually supported
3. Check that the citation actually exists (URL, DOI validity)

## Mitigation Strategies

### RAG Grounding
The most effective production mitigation: provide relevant source documents in the context and instruct the model to answer **only from the provided context**.

```
System: Answer based only on the provided documents. 
        If the answer is not in the documents, say "I don't know."
```

This converts the task from free-recall to reading comprehension — a fundamentally less hallucination-prone operation. Faithfulness to the retrieved context can then be monitored with NLI-based checks.

### Uncertainty Expression Training
Fine-tune models to express calibrated uncertainty:
- "I'm not certain, but..." / "According to my training data..."
- Train on datasets where the model is explicitly rewarded for appropriate hedging

### Citation-Based Generation
Require the model to attribute each claim to a source passage. Tools like **Perplexity AI**, **Bing Chat**, and **Google Gemini** do this by default for web-sourced answers. Attributions can be automatically verified.

### Chain-of-Thought Verification
After generating an answer, prompt the model to verify each factual claim in its own output:

```
First draft: [answer]
Now, review your answer for any claims that might be inaccurate.
For each claim, evaluate: "Is this definitely true?"
Corrected answer: [revised answer]
```

### Fact Verification Pipelines
For high-stakes deployments, run generated claims through an external fact-checking layer:
1. **Claim decomposition:** Break complex responses into atomic claims
2. **Evidence retrieval:** Search web or knowledge base for each claim
3. **NLI verification:** Classify each claim against retrieved evidence
4. **Filtering:** Remove or flag unverified claims before returning the response

### Model Training-Level Mitigations
- **RLHF with factuality rewards:** Score model outputs for factual accuracy (via human raters or automated fact-checkers) and reinforce truthful generations
- **Calibration fine-tuning:** Train on datasets where the model learns to output correct confidence scores
- **Knowledge-grounded pre-training:** Pre-train with explicit attribution objectives

## Evaluation Benchmarks

| Benchmark | Task | Focus |
|---|---|---|
| TruthfulQA | 817 questions humans often answer wrong | Imitative falsehoods |
| HaluEval | 35,000 hallucinated samples across tasks | Hallucination classification |
| FactScore | Wikipedia biography generation | Fine-grained factuality |
| RAGAs | End-to-end RAG pipeline evaluation | Faithfulness + relevance |
| FaithDial | Dialogue grounded in documents | Conversational faithfulness |
| FELM | Scientific claim verification | Factuality in scientific writing |

## Hallucination in RAG Systems

RAG does not eliminate hallucination — it changes its profile:

- **Retrieval failure:** If the answer is not in any retrieved chunk, the model may confabulate rather than say "I don't know"
- **Context conflict:** When retrieved documents contradict each other, the model may blend them incorrectly
- **Over-literal extraction:** Pulling a quote out of context and misapplying it

**Monitoring a RAG pipeline** should track:
- **Context recall:** Is the relevant information present in the retrieved context?
- **Faithfulness:** Does the final answer faithfully reflect the retrieved context?
- **Answer relevance:** Does the answer address the question asked?

Frameworks like **RAGAS**, **TruLens**, and **DeepEval** automate these metrics.

## Further Reading

- Ji et al. (2022), *Survey of Hallucination in Natural Language Generation*
- Manakul et al. (2023), *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*
- Farquhar et al. (2024), *Detecting Hallucinations in Large Language Models Using Semantic Entropy*
- Min et al. (2023), *FActScoring: Fine-Grained Atomic Evaluation of Factual Precision*
