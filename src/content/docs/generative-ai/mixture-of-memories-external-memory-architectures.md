---
title: "Mixture of Memories: External Memory Architectures for LLMs"
description: Learn how external memory systems extend LLMs beyond their fixed context windows — covering vector stores, episodic memory, working memory buffers, and hybrid architectures that give language models persistent, searchable, and updatable knowledge.
---

A standard large language model has a fixed context window — a limited number of tokens it can attend to at once. As conversations grow, documents get longer, and tasks require sustained reasoning across sessions, this window becomes the primary bottleneck. External memory architectures address this by moving knowledge outside the model weights and into queryable stores that can be read and written at inference time.

The phrase **Mixture of Memories** captures the key insight: real intelligent systems don't use a single memory type. They maintain working notes for immediate reasoning, episodic records of past events, semantic indexes of conceptual knowledge, and procedural stores for skills. Effective LLM memory architectures mirror this taxonomy.

## Why Context Window Isn't Enough

Before examining solutions, it's worth being precise about the problem:

**Static weights don't update.** A model trained until December 2024 knows nothing about events in 2025. Fine-tuning is expensive and risks forgetting existing knowledge (catastrophic forgetting). External memory provides a write path that doesn't require retraining.

**Context has a quadratic attention cost.** Standard self-attention scales as $O(n^2)$ in sequence length. A 128K-token context is expensive; 1M tokens is often impractical for latency-sensitive applications. Even with efficient attention variants, there are practical limits.

**Retrieval is selective.** Loading an entire document corpus into context is wasteful. An agent that needs one fact from a million-word knowledge base should retrieve just that fact — not pay attention over all million words.

**Persistence across sessions.** Conversations end. Without external memory, an LLM agent starts each new session with no knowledge of previous interactions. Human-like continuity requires persisted state.

## A Taxonomy of LLM Memory

Drawing from cognitive science and systems design, LLM memory can be organized into four types:

| Memory Type | Analogy | Where Stored | When Updated |
|-------------|---------|--------------|--------------|
| **Working Memory** | Short-term attention | Context window | Every token |
| **Episodic Memory** | Autobiography | Vector store / document DB | After events |
| **Semantic Memory** | General knowledge | Vector store / KV cache | Periodically |
| **Procedural Memory** | Skills/habits | Model weights / fine-tuned adapters | Training |

Most memory architectures for agents combine all four layers.

## Working Memory: The Context Window

The context window itself is the model's working memory. Within it, the model can freely attend to any position — it's a scratchpad for the current reasoning task.

Modern LLMs have pushed context windows to 128K (GPT-4o), 1M (Gemini 1.5 Pro), and 2M (Gemini 1.5 Ultra) tokens. But longer context alone doesn't solve all memory problems:

**The "Lost in the Middle" phenomenon:** LLMs attend less reliably to information in the middle of a long context. Performance peaks at the beginning and end, degrading for documents inserted in the middle of a long prompt.

**Token budget constraints:** Even if 1M tokens is technically supported, it may cost $50+ per call and add 60+ seconds of latency. Applications with high request volumes need cheaper alternatives.

**No persistence:** The context window disappears after the response is generated.

## Episodic Memory: Storing and Retrieving Past Events

Episodic memory stores specific events — conversation turns, documents processed, tool outputs observed — and retrieves relevant episodes when they become useful.

The dominant implementation is **vector store retrieval**:

```
Memory Storage (write path):
  Event happens → Extract text → Embed → Store (vector, metadata, timestamp)

Memory Retrieval (read path):
  Query formed → Embed query → ANN search → Retrieve top-k → Insert into context
```

```python
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

client = OpenAI()
qdrant = QdrantClient(":memory:")

# Create collection
qdrant.create_collection(
    collection_name="episodic_memory",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

def remember(event_text: str, metadata: dict):
    """Store an event in episodic memory."""
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=event_text
    ).data[0].embedding
    
    qdrant.upsert(
        collection_name="episodic_memory",
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": event_text, **metadata}
        )]
    )

def recall(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant memories for a query."""
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    results = qdrant.search(
        collection_name="episodic_memory",
        query_vector=query_embedding,
        limit=top_k
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]
```

### Memory Consolidation

Raw episodic storage accumulates noise — every conversation turn, including trivial exchanges, gets stored. Long-running agents need **consolidation**: periodically summarizing, deduplicating, and organizing memories.

A consolidation pipeline might:
1. Cluster recent memories by semantic similarity
2. Summarize each cluster into a compact representation
3. Replace the cluster with the summary, retaining only the most distinctive individual episodes
4. Flag outdated memories (facts that have been superseded) for removal

```python
async def consolidate_memories(agent_id: str, time_window_hours: int = 24):
    """
    Consolidate recent memories into summaries.
    Called periodically by a background task.
    """
    recent_memories = fetch_recent_memories(agent_id, time_window_hours)
    
    if len(recent_memories) < 10:
        return  # Not enough to consolidate
    
    # Use LLM to summarize and extract key facts
    memory_text = "\n".join([m["text"] for m in recent_memories])
    summary_prompt = f"""
    The following are recent observations by an AI assistant.
    Extract the key facts, decisions made, and important context.
    Discard trivial or redundant information.
    
    Observations:
    {memory_text}
    
    Summary of key information:
    """
    
    summary = await llm_call(summary_prompt)
    
    # Store summary as a high-importance episodic memory
    remember(summary, {"type": "consolidation", "source_count": len(recent_memories)})
    
    # Archive raw memories (or delete low-importance ones)
    archive_memories([m["id"] for m in recent_memories])
```

## Semantic Memory: The Knowledge Base

Semantic memory stores general world knowledge — facts, documents, reference material — independent of specific events. This is the layer most commonly implemented as a **Retrieval-Augmented Generation (RAG)** system.

### Hierarchical Semantic Memory

A flat vector store works for small corpora but degrades at scale. Hierarchical structures improve both retrieval quality and efficiency:

**Document → Section → Chunk hierarchy:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryNode:
    id: str
    text: str
    embedding: list[float]
    level: str  # "document", "section", "chunk"
    parent_id: Optional[str]
    children_ids: list[str]
    summary: Optional[str]  # For document/section nodes

class HierarchicalMemory:
    """
    Multi-level memory: documents → sections → chunks.
    Retrieval first finds relevant sections, then retrieves
    specific chunks within them (reduces false positives).
    """
    def __init__(self, vector_store):
        self.store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Step 1: Find relevant sections (coarse)
        sections = self.store.search(
            query=query,
            filter={"level": "section"},
            limit=10
        )
        
        # Step 2: Retrieve chunks within those sections (fine)
        section_ids = [s.id for s in sections]
        chunks = self.store.search(
            query=query,
            filter={
                "level": "chunk",
                "parent_id": {"$in": section_ids}
            },
            limit=top_k
        )
        
        return [c.payload["text"] for c in chunks]
```

### Temporal Indexing and Freshness

Knowledge has a shelf life. A semantic memory system without temporal awareness will confidently retrieve stale information. Best practices:

- **Timestamp all memories** at ingestion
- **Decay relevance scores** for older memories (recency bias)
- **Explicit invalidation:** When a fact is updated, mark the old version as superseded
- **Source tracking:** Record where each memory came from to support freshness verification

```python
def temporally_weighted_search(
    query: str,
    vector_store,
    recency_weight: float = 0.2,
    top_k: int = 5
) -> list[dict]:
    """
    Combine semantic similarity with recency in the ranking.
    score_final = (1 - w) * semantic_score + w * recency_score
    """
    import time
    now = time.time()
    
    # Get more candidates than needed
    candidates = vector_store.search(query=query, limit=top_k * 3)
    
    for c in candidates:
        age_days = (now - c.payload["timestamp"]) / 86400
        # Exponential decay: score halves every 90 days
        recency_score = 2 ** (-age_days / 90)
        c.final_score = (
            (1 - recency_weight) * c.score + 
            recency_weight * recency_score
        )
    
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates[:top_k]
```

## Working Memory Augmentation: Scratchpads and Memory Buffers

Some tasks require maintaining state across multiple reasoning steps without committing that state to long-term memory. Scratchpad mechanisms provide this.

### ReAct-Style Working Memory

The ReAct framework interleaves reasoning (Thought), action (Act), and observation (Obs) steps. The accumulated Thought-Act-Obs chain functions as explicit working memory:

```
Thought: The user wants a summary of recent climate research. 
         I should search my memory for recent papers on this topic.
Act: search_memory(query="climate change research 2024-2025")
Obs: Found 8 relevant papers. Key themes: carbon capture efficiency, 
     tipping points, methane reduction.

Thought: I now have enough context. I should organize by theme.
Act: write_to_scratchpad(
    "Themes: 1) Carbon capture (3 papers) 2) Tipping points (2 papers) 
     3) Methane (3 papers)"
)
Obs: Scratchpad updated.

Thought: Now I can write a structured summary addressing each theme.
Act: generate_response(...)
```

The scratchpad is part of the context window but explicitly structured so the model can write to it and reference it in subsequent steps.

### External Scratchpad for Long Chains

For very long reasoning chains, the scratchpad itself can be externalized — stored outside the context window and retrieved as needed:

```python
class ExternalScratchpad:
    """
    A write-read scratchpad that lives outside the main context.
    Supports writing notes during reasoning and reading them back.
    """
    def __init__(self):
        self.notes: list[dict] = []
        self.vector_store = InMemoryVectorStore()
    
    def write(self, note: str, key: str = None):
        entry = {"key": key or f"note_{len(self.notes)}", "text": note}
        self.notes.append(entry)
        self.vector_store.add(note, metadata=entry)
    
    def read(self, query: str = None, key: str = None) -> list[str]:
        if key:
            return [n["text"] for n in self.notes if n["key"] == key]
        elif query:
            return self.vector_store.search(query, top_k=3)
        else:
            return [n["text"] for n in self.notes[-5:]]  # Recent notes
    
    def clear(self):
        self.notes = []
        self.vector_store.clear()
```

## Memory-Augmented Architectures

Several architectures integrate memory more deeply than simple retrieval:

### MemGPT / OS-like Memory Management

MemGPT (Packer et al., 2023) treats the LLM as a processor with limited RAM (context window), backed by disk storage (external memory). The model has explicit memory management functions it can call:

- `core_memory_append(name, content)` — write to always-in-context memory
- `core_memory_replace(name, content)` — update a core memory slot
- `archival_memory_insert(content)` — write to external vector store
- `archival_memory_search(query)` — read from external vector store

The model decides autonomously when to update its memory during conversations.

### Retrieval-Augmented Thoughts (RAT)

RAT interleaves retrieval with chain-of-thought reasoning steps. At each reasoning step, the model can issue a retrieval query; the retrieved context is incorporated before the next reasoning step:

```
Reasoning step 1: "I need to understand the mechanism of RNA interference."
Retrieval query: "RNA interference mechanism siRNA Dicer"
Retrieved: [3 relevant paper excerpts]

Reasoning step 2: "Based on the retrieved context, siRNA acts by..."
[Next reasoning step, with option to retrieve again]
```

This enables deep, multi-hop reasoning that would exhaust a fixed context window if all retrieved content were loaded upfront.

### Differentiable Memory (RETRO, Atlas, etc.)

**RETRO (Borgeaud et al., 2022)** is a language model that retrieves from a frozen database of text chunks at each attention layer. The architecture has a special **cross-attention layer** that attends to retrieved neighbors:

$$\text{output} = \text{CrossAttend}\left(\text{hidden states}, \text{retrieved chunks}\right)$$

Unlike post-hoc RAG, RETRO is trained end-to-end with retrieval as an integral part of the forward pass. The model learns to use retrieved context efficiently because the training signal flows through the retrieval operation.

## Design Patterns for Memory Systems

When building production memory systems for LLM agents, several patterns have emerged:

**Pattern 1: Dual Memory (Short-term + Long-term)**
```
Short-term: Last N conversation turns in context
Long-term: Vector store of all historical interactions
Merge: Before each response, retrieve relevant long-term 
       memories and prepend to context
```

**Pattern 2: Hierarchical Priority Queue**
```
Priority 1 (always in context): User profile, agent persona, current task
Priority 2 (retrieved on demand): Domain knowledge, past interactions
Priority 3 (on explicit request): Archived documents, raw data
```

**Pattern 3: Read-Write Separation**
```
Read path: Optimized for low-latency semantic search
Write path: Async processing — embed, cluster, deduplicate, store
Background: Scheduled consolidation and freshness refresh
```

**Pattern 4: Memory Provenance Tracking**
```
Every memory entry has:
  - source (URL, document ID, session ID)
  - confidence score
  - timestamp
  - version (supports invalidation)
  - access count (for LRU eviction)
```

## Evaluating Memory Systems

Measuring memory system quality is non-trivial. Key metrics:

- **Recall@k:** What fraction of relevant memories are in the top-k results?
- **Precision@k:** What fraction of top-k results are actually relevant?
- **Memory faithfulness:** Are retrieved memories accurately represented in the final response?
- **Update consistency:** After a fact is updated, does the system stop returning the old version?
- **Latency:** P50/P95/P99 retrieval latency under load
- **Staleness rate:** What fraction of retrieved memories are outdated?

## Practical Recommendations

For production agentic systems:

1. **Start with simple RAG** — a well-chunked vector store with good embeddings handles 80% of memory needs
2. **Add episodic memory** when user personalization and session continuity matter
3. **Implement consolidation** once the memory store exceeds ~10,000 entries to prevent retrieval noise accumulation
4. **Use a hybrid retrieval** strategy: dense retrieval (vectors) + sparse retrieval (BM25) with reciprocal rank fusion for better coverage
5. **Don't over-retrieve** — injecting too much retrieved context degrades LLM performance; 3–5 highly relevant chunks beats 20 mediocre ones
6. **Track memory health** — stale, conflicting, or redundant memories silently degrade agent behavior

External memory is one of the most impactful levers for improving long-horizon LLM agent behavior. A well-designed memory architecture transforms a stateless language model into a persistent, learning assistant that improves over time.
