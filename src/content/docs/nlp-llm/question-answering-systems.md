---
title: Question Answering Systems
description: Comprehensive guide to modern Question Answering (QA) systems, comparing extractive span prediction (BERT/RoBERTa), generative reading, and open-domain retriever-reader architectures.
---

**Question Answering (QA)** is a cornerstone benchmark in Natural Language Processing (NLP) that evaluates a machine's ability to comprehend unstructured text and provide accurate, factual answers to natural language questions.

Over the past decade, QA has evolved through three distinct technological eras:
1. **Rule-Based & Knowledge Graph QA:** Parsing queries into formal SQL/SPARQL queries over structured databases.
2. **Extractive Reading Comprehension (BERT / SQuAD):** Locating and extracting exact text spans from a provided reference context.
3. **Generative Open-Domain QA (RAG & Modern LLMs):** Retrieving relevant documentation from millions of documents and synthesizing fluent, contextual answers using large language models.

---

## Taxonomy of Question Answering

```
                              Question Answering
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
  Extractive QA (Span Selection)                     Generative / Abstractive QA
  • Answer is a verbatim substring                   • Answer is synthesized in free text
  • Model predicts Start/End token indices           • Handles multi-hop reasoning & summarization
  • Examples: BERT, RoBERTa on SQuAD                 • Examples: T5, LLaMA, GPT-4

                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
  Closed-Domain (Context Provided)                   Open-Domain (No Context Provided)
  • Input: Question + Reference Passage              • Input: Only the Question
  • Pure reading comprehension                       • Must search corpus or query internal weights
```

---

## Extractive QA with Transformer Models (BERT / RoBERTa)

In extractive question answering (popularized by the Stanford Question Answering Dataset, SQuAD), the model is given a context paragraph $C$ and a question $Q$, and must predict the starting token index $i$ and ending token index $j$ of the answer span within $C$:

```
Input: [CLS] Question Tokens [SEP] Context Paragraph Tokens [SEP]
                                         │
                                [ Transformer Layers ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     Start Token Classifier (W_s)                    End Token Classifier (W_e)
                 │                                               │
                 ▼                                               ▼
     Softmax over Context Tokens                    Softmax over Context Tokens
```

### Mathematical Formulation

Let $\mathbf{T}_i \in \mathbb{R}^d$ be the final hidden representation of token $i$ in the context. The model learns two parameter vectors: a start vector $\mathbf{w}_s \in \mathbb{R}^d$ and an end vector $\mathbf{w}_e \in \mathbb{R}^d$.

The probability of token $i$ being the **start of the answer span** is:

$$P_{\text{start}}(i) = \frac{\exp(\mathbf{w}_s^\top \mathbf{T}_i)}{\sum_k \exp(\mathbf{w}_s^\top \mathbf{T}_k)}$$

The probability of token $j$ being the **end of the answer span** is:

$$P_{\text{end}}(j) = \frac{\exp(\mathbf{w}_e^\top \mathbf{T}_j)}{\sum_k \exp(\mathbf{w}_e^\top \mathbf{T}_k)}$$

The score of a candidate answer span from index $i$ to $j$ ($j \ge i$) is defined as $\mathbf{w}_s^\top \mathbf{T}_i + \mathbf{w}_e^\top \mathbf{T}_j$. During inference, the span maximizing this score is selected.

### Handling Unanswerable Questions (SQuAD 2.0)
In real-world scenarios, a provided passage may not contain the answer. In SQuAD 2.0, unanswerable questions are handled by treating the `[CLS]` token as the default null span: if the score of the null span $i=0, j=0$ exceeds the best text span score by a threshold $\tau$, the model outputs *"No answer available in context"*.

---

## Open-Domain Question Answering (ODQA)

In **Open-Domain QA**, the model is not given a reference paragraph. Instead, it must find answers from an entire knowledge base (e.g., all 6 million Wikipedia articles):

```
Question: "When was the James Webb Space Telescope launched?"
                           │
                           ▼
            [ Stage 1: Document Retriever ]
            (Dense Vector Index / BM25 over Wikipedia)
                           │
                           ▼
                  Top-k Passages Retrieved
                           │
                           ▼
            [ Stage 2: Document Reader / LLM ]
            (Extractive Span Parser or Generative Decoder)
                           │
                           ▼
            Output: "December 25, 2021"
```

ODQA frameworks fall into two categories:
1. **Retriever-Reader Systems (e.g., DrQA, REALM):** A retriever fetches candidates and an extractive reader locates the exact entity span.
2. **Retriever-Generator Systems (RAG):** The retrieved passages are formatted directly into the prompt of an instruction-tuned LLM that generates the complete factual response with attribution.

---

## Standard Evaluation Metrics

Extractive QA models are evaluated using two primary metrics against human ground-truth answers:

### 1. Exact Match (EM)
A binary metric (0 or 1) checking whether the predicted character sequence matches the ground-truth string **identically**, after standardizing punctuation, casing, and whitespace:

$$\text{EM} = \begin{cases} 1 & \text{if } \text{normalize}(\hat{y}) == \text{normalize}(y^*) \\ 0 & \text{otherwise} \end{cases}$$

### 2. Token-Level F1 Score
Measures word-level precision and recall between the predicted answer tokens and the ground-truth tokens:

$$\text{Precision} = \frac{|\hat{Y} \cap Y^*|}{|\hat{Y}|}, \quad \text{Recall} = \frac{|\hat{Y} \cap Y^*|}{|Y^*|}$$

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

If multiple ground-truth reference answers are provided by human annotators, the maximum F1 score across all references is taken.

---

## Practical Extractive QA with Hugging Face Transformers

```python
from transformers import pipeline

# Instantiate a specialized extractive QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

context = """
The Transformer model was introduced in June 2017 by Ashish Vaswani and colleagues 
at Google Brain and Google Research in their landmark paper 'Attention Is All You Need'.
Unlike recurrent neural networks, Transformers process all sequence tokens simultaneously,
utilizing self-attention mechanisms to compute dependencies regardless of distance.
"""

question = "Who introduced the Transformer architecture?"

result = qa_pipeline(question=question, context=context)

print(f"Answer: {result['answer']}")
print(f"Confidence Score: {result['score']:.4f}")
print(f"Character Start/End: ({result['start']}, {result['end']})")
```

---

## Key Takeaways

- Extractive QA (BERT) identifies the exact coordinates of facts within text with high speed and zero hallucination risk.
- Generative QA (RAG) handles complex reasoning, multi-paragraph synthesis, and natural conversational responses.
- Open-domain QA decomposes the problem into a two-stage pipeline: large-scale candidate retrieval followed by deep contextual reading.
