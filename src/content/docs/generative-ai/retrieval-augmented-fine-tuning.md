---
title: Retrieval-Augmented Fine-Tuning (RAFT)
description: Learn how RAFT bridges the gap between RAG and fine-tuning by training models to answer questions using retrieved context, producing domain-specialized models that are robust to irrelevant documents.
---

Retrieval-Augmented Fine-Tuning (RAFT) is a supervised fine-tuning technique that teaches language models to answer questions by **reasoning over retrieved documents** — including distractors — producing models that excel at domain-specific RAG tasks. It bridges the gap between standard RAG (retrieval at inference time only) and domain fine-tuning (no retrieval awareness).

## The Problem RAFT Solves

### Standard RAG Limitations

In typical RAG pipelines, a pre-trained or instruction-tuned model receives retrieved chunks alongside the user's query. However, the model was never explicitly trained to:

- Distinguish relevant context from distractor documents
- Cite the correct evidence in its answer
- Remain grounded when retrieval quality is imperfect

### Standard Fine-Tuning Limitations

Fine-tuning on a domain corpus improves factual accuracy but:

- Bakes knowledge into weights (static, can't update)
- Doesn't teach the model to use external retrieved context effectively
- May still hallucinate when knowledge boundaries are exceeded

RAFT combines the strengths of both approaches.

## RAFT Training Data Format

Each training example consists of:

- **Question** $Q$: A domain-specific query
- **Gold document** $D^*$: The document that contains the answer
- **Distractor documents** $D_1, D_2, \ldots, D_k$: Irrelevant documents mixed in
- **Answer** $A^*$: The ground-truth answer, which includes a **chain-of-thought reasoning trace** and explicit citations to evidence from $D^*$

```
Question: What is the maximum file size supported by APFS?
Documents: [D* (contains answer), D1 (distractor), D2 (distractor)]
Answer: <REASONING> The Apple File System documentation states in ##ref[D*]## 
that "APFS supports files up to 8 exabytes in size." 
Therefore, the maximum file size is 8 exabytes. </REASONING>
```

## The Training Procedure

RAFT fine-tunes a model on this supervised dataset, teaching two complementary skills:

1. **Oracle mode (with oracle doc):** ~$p\%$ of examples include $D^*$ in the context — the model learns to extract and cite the answer from the correct document
2. **Closed-book mode (without oracle doc):** ~$(1-p)\%$ of examples include only distractors — the model learns to rely on memorized domain knowledge when retrieval fails

The mixing of oracle and closed-book examples makes the trained model **robust to retrieval failures** in production.

## RAFT vs. Related Approaches

| Approach | Retrieval at Training | Retrieval at Inference | Distractor Robustness |
|---|---|---|---|
| Standard Fine-Tuning | No | No | N/A |
| RAG (no fine-tuning) | No | Yes | Low |
| Fine-tuning on RAG output | Context given | Yes | Low |
| **RAFT** | **Yes (with distractors)** | **Yes** | **High** |
| Self-RAG | No | Yes (model decides) | Moderate |

## Answer Format: Chain-of-Thought with Citations

RAFT specifically trains answers to follow a structured format:

```
<REASONING>
Step 1: Identify the relevant document → ##ref[D*]## mentions "..."
Step 2: Extract the key fact → "..."
Step 3: Compose the answer → Based on this, the answer is...
</REASONING>
<ANSWER>: [final answer]
```

This forces the model to **show its work** and link conclusions to evidence, which significantly reduces hallucination and improves faithfulness scores.

## Generating RAFT Training Data

RAFT training data can be **synthetically generated** from any document corpus using a pipeline:

1. **Chunking:** Split domain documents into passages
2. **Question generation:** Use a strong LLM to generate questions that are answerable from each passage
3. **Answer generation:** Generate chain-of-thought answers that cite the oracle passage
4. **Distractor sampling:** Randomly sample $k$ other passages as distractors per example
5. **Closed-book generation:** Generate answers for a fraction of examples without the oracle document

This pipeline requires no human annotation, making RAFT scalable to large proprietary corpora.

## When to Use RAFT

RAFT is well-suited when:

- You have a **stable domain corpus** (technical documentation, legal documents, medical literature)
- Users ask **precise factual questions** that require grounded answers
- Answer quality and **faithfulness** matter more than creative generation
- Retrieval quality in production is variable or imperfect

It is less suitable for:

- General-purpose assistants that need broad world knowledge
- Dynamic knowledge bases that update continuously
- Creative or open-ended generation tasks

## Domain Applications

- **Enterprise Q&A:** Internal documentation, HR policies, product manuals
- **Medical:** Clinical guidelines, drug interaction databases, EHR summarization
- **Legal:** Contract analysis, case law retrieval, regulatory compliance
- **Code:** API documentation, codebase Q&A
- **Customer Support:** Product FAQ, troubleshooting guides

## Implementation

RAFT is model-agnostic and can be applied to any instruction-tunable LLM:

```python
# Illustrative RAFT fine-tuning setup
from transformers import Trainer, TrainingArguments

# Each example: {"question": ..., "documents": [...], "answer": ...}
# Documents include both oracle + distractors

trainer = Trainer(
    model=base_model,
    args=TrainingArguments(
        output_dir="raft-domain-model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
    ),
    train_dataset=raft_dataset,
    data_collator=raft_collator,
)
trainer.train()
```

## Further Reading

- Zhang et al. (2024), *RAFT: Adapting Language Model to Domain Specific RAG* — original paper from UC Berkeley and Microsoft Research
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Asai et al. (2023), *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
