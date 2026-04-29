---
title: Introduction to Haystack
description: Get started with Haystack by deepset — an open-source framework for building production-ready search and retrieval-augmented generation pipelines — covering its pipeline abstraction, document stores, retrievers, generators, custom components, and deployment patterns.
---

**Haystack** is an open-source framework developed by deepset for building **search, question answering, and retrieval-augmented generation (RAG) applications** with large language models. Haystack provides a modular **pipeline abstraction** that lets you connect retrievers, document stores, rerankers, generators, and custom components into end-to-end AI systems that are easy to configure, test, and deploy.

While frameworks like LangChain focus on general LLM orchestration and LlamaIndex specializes in data ingestion and indexing, Haystack's strength is its **production-grade pipeline system** — designed for building robust, maintainable search and Q&A applications that operate reliably at scale.

## Installation

```bash
pip install haystack-ai

# Optional: specific document store backends
pip install "haystack-ai[elasticsearch]"
pip install "haystack-ai[opensearch]"

# Common integrations
pip install "amazon-bedrock-haystack"
pip install "cohere-haystack"
```

Haystack 2.x (the current major version) has a redesigned component-based architecture that is more flexible than the earlier 1.x pipeline system.

## Core Architecture

### Components

Everything in Haystack is a **component** — a Python class decorated with `@component` that has defined inputs and outputs. Components are the atomic building blocks of pipelines:

```python
from haystack import component
from typing import List

@component
class DocumentCleaner:
    """Custom component that cleans document text."""
    
    @component.output_types(documents=List[dict])
    def run(self, documents: List[dict]) -> dict:
        cleaned = []
        for doc in documents:
            cleaned.append({
                **doc,
                "content": doc["content"].strip().lower()
            })
        return {"documents": cleaned}
```

Built-in components cover the full pipeline: document converters, splitters, embedders, retrievers, rerankers, readers, generators, and routers.

### Pipelines

A **Pipeline** connects components by wiring their outputs to inputs:

```python
from haystack import Pipeline

pipeline = Pipeline()
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())

# Wire cleaner's output to embedder's input
pipeline.connect("cleaner.documents", "embedder.documents")

result = pipeline.run({"cleaner": {"documents": raw_docs}})
```

Pipelines are **directed acyclic graphs (DAGs)** — components can branch (one output to multiple inputs) and merge (multiple outputs to one input), enabling complex multi-path architectures.

## Building a RAG Pipeline

### Step 1: Indexing Pipeline

```python
from haystack import Pipeline, Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# Initialize document store
document_store = ChromaDocumentStore(persist_path="./chroma_index")

# Build indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component(
    "splitter",
    DocumentSplitter(split_by="sentence", split_length=5, split_overlap=2)
)
indexing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

# Connect components
indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

# Run indexing
indexing_pipeline.run({"converter": {"sources": ["document.pdf"]}})
print(f"Indexed {document_store.count_documents()} document chunks")
```

### Step 2: RAG Query Pipeline

```python
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

# RAG prompt template
RAG_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so.

Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}

Question: {{ question }}

Answer:
"""

# Build RAG pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
rag_pipeline.add_component(
    "retriever",
    ChromaEmbeddingRetriever(document_store=document_store, top_k=5)
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=RAG_TEMPLATE))
rag_pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))

# Connect components
rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")

# Query
result = rag_pipeline.run({
    "query_embedder": {"text": "What are the main findings?"},
    "prompt_builder": {"question": "What are the main findings?"}
})

print(result["generator"]["replies"][0])
```

## Hybrid Search

Haystack makes it straightforward to combine dense (embedding-based) and sparse (keyword-based) retrieval:

```python
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker

hybrid_pipeline = Pipeline()

# Dual retrieval paths
hybrid_pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=store, top_k=10))
hybrid_pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=store, top_k=10))
hybrid_pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder())

# Merge results from both retrievers (reciprocal rank fusion)
hybrid_pipeline.add_component(
    "joiner",
    DocumentJoiner(join_mode="reciprocal_rank_fusion")
)

# Rerank the merged results
hybrid_pipeline.add_component(
    "reranker",
    TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5)
)

hybrid_pipeline.add_component("prompt_builder", PromptBuilder(template=RAG_TEMPLATE))
hybrid_pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))

# Connect: query goes to BM25 directly, embedding retriever needs embedded query
hybrid_pipeline.connect("query_embedder.embedding", "embedding_retriever.query_embedding")
hybrid_pipeline.connect("bm25_retriever.documents", "joiner.documents")
hybrid_pipeline.connect("embedding_retriever.documents", "joiner.documents")
hybrid_pipeline.connect("joiner.documents", "reranker.documents")
hybrid_pipeline.connect("reranker.documents", "prompt_builder.documents")
hybrid_pipeline.connect("prompt_builder", "generator")

result = hybrid_pipeline.run({
    "bm25_retriever": {"query": query},
    "query_embedder": {"text": query},
    "reranker": {"query": query},
    "prompt_builder": {"question": query}
})
```

## Metadata Filtering

Haystack supports rich metadata filtering at retrieval time — restricting results to documents matching specific conditions:

```python
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

# Documents with metadata
docs = [
    Document(content="Q3 revenue was $5.2B", meta={"year": 2024, "quarter": "Q3", "source": "earnings"}),
    Document(content="Q2 revenue was $4.8B", meta={"year": 2024, "quarter": "Q2", "source": "earnings"}),
    Document(content="2023 annual report highlights", meta={"year": 2023, "quarter": None, "source": "annual"}),
]

# Retrieve only 2024 documents
result = rag_pipeline.run({
    "retriever": {
        "filters": {
            "operator": "AND",
            "conditions": [
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"field": "meta.source", "operator": "==", "value": "earnings"}
            ]
        }
    },
    "prompt_builder": {"question": "What were the quarterly revenues in 2024?"}
})
```

## Pipeline Serialization and Deployment

Pipelines can be serialized to YAML for configuration-driven deployment:

```python
# Save pipeline to YAML
with open("rag_pipeline.yaml", "w") as f:
    pipeline.dump(f)

# Load from YAML
from haystack import Pipeline

with open("rag_pipeline.yaml") as f:
    pipeline = Pipeline.load(f)
```

YAML serialization enables version-controlled pipeline configurations — the pipeline definition is stored alongside code, making deployments reproducible and auditable.

### REST API with Hayhooks

**Hayhooks** wraps Haystack pipelines in a FastAPI service automatically:

```bash
pip install hayhooks
hayhooks run --pipelines-dir ./pipelines
```

Every `.yaml` pipeline file in the directory becomes a REST endpoint at `/pipelines/{pipeline_name}/run` — zero additional code required to expose a pipeline as an API.

## Custom Generator Components

Integrating any LLM provider requires just a component wrapper:

```python
import anthropic
from haystack import component
from typing import List, Optional

@component
class AnthropicGenerator:
    def __init__(self, model: str = "claude-opus-4-5", max_tokens: int = 1024):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
    
    @component.output_types(replies=List[str])
    def run(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        return {"replies": [response.content[0].text]}
```

## Evaluation

Haystack integrates with **RAGAS** for pipeline evaluation:

```python
from haystack_experimental.evaluation.harness import RAGEvaluationHarness
from haystack_experimental.evaluation.metrics import RagasMetric

harness = RAGEvaluationHarness(
    rag_pipeline=rag_pipeline,
    rag_pipeline_inputs={...},
    metrics=[
        RagasMetric.ANSWER_FAITHFULNESS,
        RagasMetric.ANSWER_RELEVANCY,
        RagasMetric.CONTEXT_PRECISION,
    ]
)

results = harness.run(
    inputs=[{"question": q, "ground_truth": a} for q, a in test_set]
)
print(results.to_pandas())
```

## Haystack vs. LangChain vs. LlamaIndex

| Aspect | Haystack | LangChain | LlamaIndex |
|---|---|---|---|
| Primary strength | Production search & RAG | General LLM orchestration | Data ingestion & indexing |
| Pipeline model | Typed DAG (strict) | Flexible chains/graphs | Query engines |
| Serialization | Native YAML | LangChain Hub | JSON/YAML |
| REST serving | Hayhooks (built-in) | LangServe | LlamaServer |
| Evaluation | RAGAS integration | LangSmith | Evaluation modules |
| Learning curve | Moderate | Low | Moderate |

Haystack is the right choice when building a well-defined search or Q&A service that needs to be maintainable, testable, and deployable as a REST API — particularly for enterprise applications where pipeline configuration management and reproducibility matter.
