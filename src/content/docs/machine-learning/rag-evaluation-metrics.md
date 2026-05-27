---
title: "RAG Evaluation Metrics: Measuring Retrieval-Augmented Generation"
description: "Learn how to evaluate RAG systems comprehensively — covering retrieval metrics, generation quality, and end-to-end metrics like faithfulness, answer relevancy, and context precision."
---

Retrieval-Augmented Generation (RAG) has become the standard architecture for building knowledge-intensive AI applications. But measuring whether a RAG system is actually working well requires evaluating multiple components independently and jointly.

## The RAG Evaluation Framework

A complete RAG system has three stages:

1. **Retrieval**: Given a query, retrieve relevant documents from the knowledge base.
2. **Context Assembly**: Combine retrieved documents into a prompt for the LLM.
3. **Generation**: The LLM produces an answer based on the provided context.

Each stage has its own evaluation metrics, and the interactions between stages create additional joint metrics.

## Retrieval Metrics

Evaluate the retrieval component independently using the query and the ground-truth relevant documents.

### Precision and Recall
- **Precision@K**: Of the top K retrieved documents, what fraction are relevant?
- **Recall@K**: Of all relevant documents, what fraction are in the top K?
- **mAP (Mean Average Precision)**: Accounts for the ranking order of retrieved documents.

```python
# Precision@K
def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k

# Recall@K
def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant)
```

### NDCG (Normalized Discounted Cumulative Gain)
NDCG accounts for both relevance and ranking order. Documents at higher positions in the retrieved list contribute more to the score.

### MRR (Mean Reciprocal Rank)
The reciprocal of the position of the first relevant document. Simple but effective for single-answer retrieval.

## Context Quality Metrics

How good is the assembled context for answering the query?

### Context Precision
Does the context contain relevant information in high-ranked positions? Irrelevant documents early in the context can distract the LLM.

### Context Recall
How much of the relevant information from the knowledge base was successfully retrieved?

### Context Relevancy
Is the retrieved content actually relevant to the query? This catches cases where retrieval returns documents that look similar but don't help.

## Generation Metrics

### Faithfulness (Groundedness)
Does the generated answer stick to the provided context, or does it hallucinate information not in the context?

```python
# Faithfulness via claim extraction
def compute_faithfulness(answer: str, context: str) -> float:
    claims = extract_factual_claims(answer)
    supported = sum(1 for c in claims if verify_claim_in_context(c, context))
    return supported / len(claims)
```

### Answer Relevancy
Is the answer directly responsive to the query? A highly relevant answer doesn't just contain true information — it contains the information the user wanted.

### Answer Correctness
For queries with a known correct answer, measure answer accuracy directly. For open-ended questions, use LLM-based scoring: "Given the reference answer and the generated answer, score correctness from 0-1."

### Completeness
Does the answer fully address the query, or does it only partially answer it? This is particularly important for complex questions with multiple subparts.

## End-to-End Frameworks

### RAGAS (Retrieval-Augmented Generation Assessment)
RAGAS provides a standardized set of metrics:
- **Faithfulness**: Groundedness in context.
- **Answer Relevancy**: Response relevance to query.
- **Context Relevancy**: Quality of retrieval.
- **Context Recall**: Coverage of relevant information.

### Trulens
An open-source evaluation framework with:
- **Groundedness**: Like faithfulness, but with multi-level granularity.
- **Context Relevance**: Per-token analysis of context utilization.
- **Question Answering Quality**: Combined QA metrics.

### ARES (Automated RAG Evaluation System)
Uses a lightweight LLM to score RAG systems without human-labeled ground truth. Generates synthetic queries and answers from documents, then evaluates retrieval and generation against those.

## Practical Evaluation Pipeline

### 1. Define Your Metrics
Not all metrics matter equally for your use case. For a customer support chatbot, answer relevancy and faithfulness matter most. For research synthesis, completeness and answer correctness matter most.

### 2. Create an Evaluation Dataset
- **Gold standard queries**: Questions with known relevant documents and correct answers.
- **Diverse query types**: Simple lookup, multi-hop reasoning, summarization, comparison.
- **Edge cases**: Ambiguous queries, queries outside the knowledge base, very long queries.

### 3. Automated + Human Evaluation
- Run automated metrics on every change (CI/CD integration).
- Sample 5–10% of outputs for human review.
- Track metrics over time to detect regressions.

### 4. A/B Testing
Deploy two versions of your RAG system to live traffic and compare business metrics: user satisfaction, task completion rate, escalations to human support.

RAG evaluation is complex because it involves multiple components. A system with perfect retrieval but weak generation still fails; a system with weak retrieval but strong generation can sometimes recover with better prompting. Measuring each component independently and jointly gives you the visibility you need to improve systematically.