---
title: Knowledge Graph RAG (GraphRAG)
description: Discover how GraphRAG augments retrieval-augmented generation with structured knowledge graphs, enabling multi-hop reasoning, global dataset summarization, and answers that vector search alone cannot produce.
---

GraphRAG (Graph Retrieval-Augmented Generation) extends standard RAG by grounding language model responses in **structured knowledge graphs** rather than (or in addition to) unstructured vector-searched passages. It excels at questions that require multi-hop reasoning, connecting facts across multiple entities, or understanding global themes across an entire document corpus.

## The Limitations of Standard Vector RAG

Standard RAG works by embedding queries and documents into a shared vector space, retrieving the top-$k$ most semantically similar chunks, and feeding them to an LLM. This works well for **local queries** — those answered by a single passage — but fails for:

- **Multi-hop questions:** "Who founded the company that acquired the startup founded by the CEO of Company X?"
- **Global summarization:** "What are the major themes across this entire research corpus?"
- **Implicit relationships:** Facts spread across many documents that must be combined
- **Structured retrieval:** Queries about specific relational properties (e.g., all suppliers in a given region)

Vector similarity retrieves semantically close passages — it does not traverse relationships.

## Knowledge Graphs: A Brief Overview

A **knowledge graph (KG)** represents information as a set of triples:

$$(\text{subject}, \text{predicate}, \text{object}) \quad \text{e.g.,}\quad (\text{AlphaFold}, \text{developed\_by}, \text{DeepMind})$$

Entities become nodes; relationships become directed edges. KGs enable:
- **Traversal:** Hop from entity A → relationship → entity B → relationship → entity C
- **Inference:** Derive new facts from existing relationships
- **Structured queries:** SPARQL or Cypher queries for precise retrieval

## GraphRAG from Microsoft Research

**Microsoft GraphRAG** (Edge et al., 2024) introduced a two-phase pipeline that builds and queries a knowledge graph extracted from arbitrary text corpora:

### Phase 1: Indexing (Graph Construction)

```
Documents
    ↓
[Text Chunking]
    ↓
[LLM Entity Extraction]   → Entities (nodes)
[LLM Relationship Extraction] → Relationships (edges)
    ↓
[Community Detection]     → Hierarchical clusters of related entities
    ↓
[LLM Community Summarization] → Summary for each community
```

1. **Entity and relationship extraction:** An LLM reads each text chunk and extracts (entity, relationship, entity) triples
2. **Graph construction:** Triples are assembled into a property graph
3. **Community detection:** The Leiden algorithm clusters tightly connected entities into communities at multiple hierarchical levels
4. **Community summarization:** An LLM generates a paragraph-length summary for each community

### Phase 2: Querying

**Local Search:** For specific factual queries, retrieves related entities, their relationships, and their community summaries via a combination of text search and graph traversal.

**Global Search:** For broad thematic queries, issues a map-reduce over all community summaries:
1. **Map:** Ask the LLM to identify relevant points in each community summary
2. **Reduce:** Aggregate and synthesize the extracted points into a final answer

This global search answers queries like *"What are the main risks discussed in this document collection?"* — impossible with vector RAG alone.

## Naive RAG vs. GraphRAG vs. Hybrid

| Approach | Best For | Weakness |
|---|---|---|
| Vector RAG | Local, factual, single-passage queries | Multi-hop, global questions |
| GraphRAG (global) | Thematic, dataset-wide questions | High indexing cost |
| GraphRAG (local) | Entity-centric, relational queries | Misses passages not in graph |
| Hybrid RAG | General purpose | Complexity |

## Other Graph-Enhanced RAG Approaches

### KGRAG: External Knowledge Graph Integration
Rather than building a KG from the corpus, use an existing KG (Wikidata, Freebase, domain-specific graphs) to supplement LLM answers:
1. Extract entities from the user query
2. Retrieve subgraphs around those entities from the KG
3. Linearize the subgraph into text triples
4. Feed to LLM alongside retrieved passages

### HippoRAG
HippoRAG (Guo et al., 2024) models long-term associative memory inspired by the human hippocampus using a **personalized PageRank** over a KG built from passages. Queries navigate the graph using propagating relevance rather than a single top-$k$ retrieval.

### RAPTOR: Recursive Abstractive Processing
RAPTOR builds a **tree of summaries** over document chunks using clustering and recursive LLM summarization — a simpler but related approach to hierarchical knowledge organization.

### LightRAG
LightRAG (Guo et al., 2024) is a simpler, faster GraphRAG alternative that focuses on efficient incremental graph updates, making it practical for dynamic corpora where documents are added frequently.

## Implementation Example

Using Microsoft GraphRAG:

```bash
# Install
pip install graphrag

# Initialize a project
python -m graphrag.index --init --root ./my_project

# Place documents in ./my_project/input/
# Configure settings in ./my_project/settings.yaml

# Build the knowledge graph index
python -m graphrag.index --root ./my_project

# Query (global search)
python -m graphrag.query \
  --root ./my_project \
  --method global \
  "What are the major themes in these documents?"

# Query (local search)
python -m graphrag.query \
  --root ./my_project \
  --method local \
  "Who are the key researchers in this field?"
```

## Cost Considerations

GraphRAG's indexing phase calls an LLM to extract entities and generate community summaries for **every chunk in the corpus** — making indexing significantly more expensive than embedding-based vector indexing:

- A 1M-token corpus may require 5–10M LLM tokens for indexing
- Strategies to reduce cost: use smaller extraction models (e.g., GPT-4o-mini), reduce community levels, sample chunks

Global search also issues many parallel LLM calls over community summaries — cost scales with corpus size.

## Use Cases

- **Enterprise knowledge management:** Connecting information siloed across departments
- **Scientific literature mining:** Multi-hop reasoning over research papers
- **Legal and compliance:** Tracing regulatory relationships across statutes and cases
- **Intelligence analysis:** Connecting entities across intelligence reports
- **Medical knowledge bases:** Drug-gene-disease relationship navigation

## Tradeoffs and When to Choose GraphRAG

**Choose GraphRAG when:**
- Questions require reasoning over relationships between entities
- The corpus has rich, interconnected factual content
- Global thematic analysis is valuable
- The corpus is relatively stable (indexing is expensive to redo)

**Stick with vector RAG when:**
- Questions are locally answerable from single passages
- The corpus updates very frequently
- Latency and cost are primary constraints
- The content is unstructured narrative (not entity-rich)

## Further Reading

- Edge et al. (2024), *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Microsoft Research
- Traag et al. (2019), *From Louvain to Leiden: Guaranteeing Well-Connected Communities*
- Guo et al. (2024), *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*
- Guo et al. (2024), *LightRAG: Simple and Fast Retrieval-Augmented Generation*
