---
title: Introduction to RAGAS
description: Get started with RAGAS — the open-source framework for evaluating Retrieval-Augmented Generation (RAG) pipelines using reference-free metrics like faithfulness, answer relevancy, context precision, and context recall.
---

RAGAS (Retrieval Augmented Generation Assessment) is an open-source evaluation framework designed specifically for RAG pipelines. Unlike traditional NLP evaluation metrics that require human-written reference answers, RAGAS uses LLMs to evaluate RAG systems along multiple dimensions — making it possible to assess production-grade pipelines without expensive annotation.

## Why RAG Evaluation is Hard

A RAG pipeline has several moving parts:

1. **Retriever**: finds relevant context from a knowledge base
2. **Generator**: produces an answer conditioned on the retrieved context and the query

Standard metrics like ROUGE or BLEU measure token overlap with a reference answer. They fail for RAG because:

- Good answers may be worded differently from the reference
- The retriever and generator can fail independently
- A fluent answer may be unfaithful to the retrieved context (hallucination)

RAGAS decomposes evaluation into components that can be assessed independently.

## Core RAGAS Metrics

RAGAS evaluates along four primary axes. All use an LLM as judge and require no ground-truth answers (except context recall, which optionally uses them).

### Faithfulness

**Faithfulness** measures whether the generated answer is factually grounded in the retrieved context — i.e., can every claim in the answer be inferred from the context?

$$\text{Faithfulness} = \frac{\text{# claims in answer that are supported by context}}{\text{# total claims in answer}}$$

An LLM extracts individual claims from the answer and then checks each one against the context. A score of 1.0 means fully grounded; lower scores indicate hallucination.

**Example:**

```text
Context: "The Eiffel Tower was completed in 1889."
Answer: "The Eiffel Tower was built in 1889 and is 324 meters tall."
```

The claim "built in 1889" is supported. The claim "324 meters tall" is **not** in the context, so faithfulness = 0.5.

### Answer Relevancy

**Answer relevancy** measures how well the generated answer addresses the question — without evaluating factual correctness.

The approach: use an LLM to generate $n$ hypothetical questions that the given answer would answer, then measure the cosine similarity between those generated questions and the original question:

$$\text{Answer Relevancy} = \frac{1}{n} \sum_{i=1}^n \cos(\text{sim}(q, \hat{q}_i))$$

If the answer is off-topic or evasive (e.g., "I don't know"), the generated questions will diverge from the original, giving a low score.

### Context Precision

**Context precision** measures whether the retrieved context is useful — specifically, whether the top-ranked chunks actually contain information relevant to generating the answer.

$$\text{Context Precision} = \frac{\sum_{k=1}^K P@k \cdot \text{rel}(k)}{|\text{relevant chunks}|}$$

where $P@k$ is precision at rank $k$ and $\text{rel}(k)$ is 1 if chunk $k$ is relevant to the question-answer pair. This is essentially a ranking quality metric: relevant chunks should be retrieved early.

### Context Recall

**Context recall** measures whether the retrieved context contains all the information needed to answer the question, using ground-truth answers as reference (when available):

$$\text{Context Recall} = \frac{\text{# claims in ground-truth answer that can be attributed to context}}{\text{# total claims in ground-truth answer}}$$

If the retriever misses important documents, context recall will be low even if what was retrieved is useful.

## Additional Metrics

RAGAS also provides:

| Metric | Description |
| --- | --- |
| **Context Entity Recall** | Checks named entities in ground truth appear in retrieved context |
| **Noise Sensitivity** | Robustness when irrelevant context is introduced |
| **Answer Correctness** | Factual + semantic similarity to ground truth answer (requires reference) |
| **Answer Similarity** | Semantic similarity to ground truth using embeddings |
| **Summarization Score** | Measures conciseness and coverage for summary tasks |

## Installation and Setup

```bash
pip install ragas
```

RAGAS supports OpenAI, Anthropic, Azure OpenAI, and any LangChain-compatible LLM as the judge model.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure judge LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

judge_llm = LangchainLLMWrapper(llm)
judge_embeddings = LangchainEmbeddingsWrapper(embeddings)
```

## Preparing Your Dataset

RAGAS expects a dataset with the following columns:

| Column | Required | Description |
| --- | --- | --- |
| `user_input` | Yes | The user's question |
| `response` | Yes | The generated answer |
| `retrieved_contexts` | Yes | List of retrieved context strings |
| `reference` | For recall/correctness | Ground-truth answer |

```python
from datasets import Dataset

data = {
    "user_input": [
        "When was the Eiffel Tower completed?",
        "What is the boiling point of water?",
    ],
    "response": [
        "The Eiffel Tower was completed in 1889.",
        "Water boils at 100°C at standard atmospheric pressure.",
    ],
    "retrieved_contexts": [
        ["The Eiffel Tower, completed in 1889, is located in Paris."],
        ["Water transitions from liquid to gas at 100 degrees Celsius (212°F) at sea level."],
    ],
    "reference": [
        "The Eiffel Tower was completed in 1889.",
        "Water boils at 100°C (212°F) at standard atmospheric pressure.",
    ],
}

dataset = Dataset.from_dict(data)
```

## Running Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

print(results)
# Output:
# {'faithfulness': 0.97, 'answer_relevancy': 0.93,
#  'context_precision': 0.88, 'context_recall': 0.91}

df = results.to_pandas()
print(df.head())
```

## Integration with RAG Frameworks

### LangChain Integration

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Build your RAG chain
vectorstore = FAISS.load_local("my_index", OpenAIEmbeddings())
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# Run pipeline and collect outputs for RAGAS
questions = ["What is quantum computing?", "Who invented the internet?"]
ragas_data = {"user_input": [], "response": [], "retrieved_contexts": []}

for q in questions:
    result = rag_chain(q)
    ragas_data["user_input"].append(q)
    ragas_data["response"].append(result["result"])
    ragas_data["retrieved_contexts"].append(
        [doc.page_content for doc in result["source_documents"]]
    )
```

### LlamaIndex Integration

RAGAS provides native LlamaIndex support via the `ragas.integrations.llama_index` module, allowing direct evaluation of LlamaIndex query engines.

## Testset Generation

A unique feature of RAGAS is automated **evaluation dataset generation** from your knowledge base:

```python
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()

generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o"),
    critic_llm=ChatOpenAI(model="gpt-4o"),
    embeddings=OpenAIEmbeddings(),
)

testset = generator.generate_with_langchain_docs(
    docs,
    test_size=50,
    distributions={"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25},
)
```

This generates:

- **Simple questions**: directly answered from one chunk
- **Reasoning questions**: require inference
- **Multi-context questions**: require synthesizing multiple chunks

## Interpreting Results

### Score Ranges and Targets

| Metric | Production Target | Investigate if Below |
| --- | --- | --- |
| Faithfulness | > 0.90 | < 0.80 |
| Answer Relevancy | > 0.85 | < 0.75 |
| Context Precision | > 0.80 | < 0.65 |
| Context Recall | > 0.85 | < 0.70 |

### Diagnosing Problems

Low scores point to specific pipeline components:

- **Low faithfulness** → generator is hallucinating; check prompt constraints, consider smaller context windows
- **Low answer relevancy** → generator is off-topic or evasive; improve system prompt
- **Low context precision** → retriever returns too much noise; tune retrieval top-$k$, chunking, or reranking
- **Low context recall** → retriever misses relevant documents; improve chunking strategy, embedding model, or retrieval method

## Advanced Configuration

### Custom Metrics

RAGAS supports custom metrics via the `MetricWithLLM` base class:

```python
from ragas.metrics.base import MetricWithLLM
from dataclasses import dataclass

@dataclass
class ToxicityMetric(MetricWithLLM):
    name: str = "toxicity"

    async def _ascore(self, row: dict, callbacks) -> float:
        # Custom scoring logic using self.llm
        ...
```

### Using Non-OpenAI Models

RAGAS works with any LangChain-compatible LLM, including local models via Ollama:

```python
from langchain_ollama import ChatOllama

local_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2"))
```

## Summary

RAGAS provides a practical, reference-free evaluation framework for RAG pipelines with four core metrics — faithfulness, answer relevancy, context precision, and context recall — each targeting a specific failure mode. Its automated testset generation, framework integrations, and actionable diagnostic output make it a valuable tool for teams building and iterating on production RAG systems.
