---
title: "RAG Architectures: Building Production-Ready Retrieval Systems"
description: "Design and implement production RAG systems — from basic retrieval to advanced techniques like hybrid search, reranking, query rewriting, and knowledge graph integration."
---

Retrieval-Augmented Generation (RAG) has become the standard architecture for knowledge-intensive AI applications. This guide covers the complete landscape of RAG system design, from basic implementations to production-ready systems.

## The RAG Pipeline

A RAG system consists of several stages:

```python
class RAGPipeline:
    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever      # Vector search
        self.reranker = reranker        # Cross-encoder scoring
        self.generator = generator      # LLM for generation
    
    def __call__(self, query: str) -> str:
        # Stage 1: Query understanding
        query = self.rewrite_query(query)
        
        # Stage 2: Retrieval
        docs = self.retriever.retrieve(query, top_k=100)
        
        # Stage 3: Reranking
        ranked = self.reranker.rerank(query, docs, top_k=10)
        
        # Stage 4: Context assembly
        context = self.assemble_context(ranked)
        
        # Stage 5: Generation
        response = self.generator.generate(context, query)
        
        return response
```

## Document Processing

### Text Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def chunk(self, documents):
        """Chunk documents into pieces."""
        chunks = self.splitter.create_documents(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
        return chunks

# Semantic-aware chunking
class SemanticChunker:
    def chunk(self, document, embedding_model):
        """Chunk based on semantic similarity."""
        sentences = split_into_sentences(document.page_content)
        embeddings = embedding_model.encode(sentences)
        
        # Find natural breaks based on embedding similarity
        breakpoints = find_semantic_breaks(embeddings)
        
        chunks = []
        for i, start in enumerate(breakpoints):
            end = breakpoints[i + 1] if i + 1 < len(breakpoints) else len(sentences)
            chunk = " ".join(sentences[start:end])
            chunks.append({
                "content": chunk,
                "start_sentence": start,
                "end_sentence": end,
            })
        
        return chunks
```

### Metadata Extraction

```python
class MetadataExtractor:
    def extract(self, document):
        """Extract structured metadata from document."""
        return {
            "title": extract_title(document),
            "headers": extract_headers(document),
            "page_numbers": extract_pages(document),
            "section_titles": extract_sections(document),
            "keywords": extract_keywords(document),
            "entities": extract_entities(document),
            "language": detect_language(document),
            "document_type": classify_document(document),
        }
```

## Retrieval Strategies

### Dense Retrieval

```python
class DenseRetriever:
    def __init__(self, embedding_model, vector_store, index_name):
        self.embedding_model = embedding_model
        self.vector_store = vector_store(index_name)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search
        results = self.vector_store.similarity_search(
            query_embedding,
            k=top_k,
            include_metadata=True,
        )
        
        return results
```

### Hybrid Search

```python
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, fusion_method="rrf"):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.fusion_method = fusion_method
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        # Dense retrieval
        dense_results = self.dense.retrieve(query, top_k * 2)
        
        # Sparse retrieval
        sparse_results = self.sparse.retrieve(query, top_k * 2)
        
        # Fuse results using Reciprocal Rank Fusion
        fused = self.rrf_fuse(dense_results, sparse_results, top_k)
        
        return fused
    
    def rrf_fuse(self, dense, sparse, k=60):
        """Reciprocal Rank Fusion."""
        scores = {}
        
        for rank, doc in enumerate(dense):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
        
        for rank, doc in enumerate(sparse):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
        
        sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
        return [get_doc_by_id(did) for did, _ in sorted_docs[:k]]
```

### Query Expansion

```python
class QueryExpander:
    def __init__(self, llm):
        self.llm = llm
    
    def expand(self, query: str) -> List[str]:
        """Generate query variations."""
        prompt = f"""Generate 3 variations of this search query
        to improve retrieval. Each variation should capture
        a different aspect or rephrase the query.
        
        Original: {query}
        
        Variations:
        1."""
        
        response = self.llm.generate(prompt)
        variations = parse_variations(response)
        return [query] + variations
    
    def expand_with_context(self, query: str, context: str) -> List[str]:
        """Expand query considering conversation context."""
        prompt = f"""Rewrite this query to incorporate context from the
        conversation history.
        
        Conversation: {context}
        Current query: {query}
        
        Rewritten query:"""
        
        rewritten = self.llm.generate(prompt)
        return [query, rewritten]
```

## Reranking

```python
class CrossEncoderReranker:
    def __init__(self, cross_encoder_model):
        self.model = cross_encoder_model
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 10):
        """Score query-document pairs and return top results."""
        # Prepare pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Batch scoring
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in ranked[:top_k]]
```

## Knowledge Graph RAG

```python
class KnowledgeGraphRAG:
    def __init__(self, kg_store, vector_store, llm):
        self.kg = kg_store
        self.vector = vector_store
        self.llm = llm
    
    def retrieve(self, query: str) -> str:
        # Step 1: Extract entities from query
        entities = self.extract_entities(query)
        
        # Step 2: Query knowledge graph
        kg_context = self.kg.query(entities)
        
        # Step 3: Retrieve relevant documents
        docs = self.vector.similarity_search(query, top_k=10)
        
        # Step 4: Fuse contexts
        fused_context = self.fuse_contexts(kg_context, docs)
        
        # Step 5: Generate
        response = self.llm.generate(
            self.create_prompt(query, fused_context)
        )
        
        return response
    
    def extract_entities(self, query: str) -> List[Entity]:
        """Use LLM to extract entities from query."""
        prompt = f"""Extract named entities from this query.
        Return as JSON list with type and name.
        
        Query: {query}
        
        Entities:"""
        
        return self.llm.generate_json(prompt)
    
    def fuse_contexts(self, kg_context, docs):
        """Combine knowledge graph and document context."""
        return {
            "knowledge_graph": kg_context,
            "documents": [d.page_content for d in docs],
            "citation_map": self.create_citations(kg_context, docs),
        }
```

## Context Assembly

```python
class ContextAssembler:
    def __init__(self, max_tokens=6000, separator="\n\n"):
        self.max_tokens = max_tokens
        self.separator = separator
    
    def assemble(self, documents: List[Document], query: str) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        current_length = 0
        
        # Sort by relevance
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        
        for doc in sorted_docs:
            doc_text = self.format_document(doc, query)
            doc_tokens = count_tokens(doc_text)
            
            if current_length + doc_tokens > self.max_tokens:
                # Truncate or skip
                if current_length == 0:
                    doc_text = truncate_to_token_limit(doc_text, self.max_tokens)
                else:
                    break
            
            context_parts.append(doc_text)
            current_length += doc_tokens
        
        return self.separator.join(context_parts)
    
    def format_document(self, doc: Document, query: str) -> str:
        """Format document with citation."""
        return f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
```

## Production Considerations

### Caching

```python
class RAGCache:
    def __init__(self, vector_store, redis_client):
        self.vector = vector_store
        self.redis = redis_client
    
    def get_or_query(self, query: str, retriever, ttl=3600):
        """Check cache before retrieval."""
        cache_key = hash_query(query)
        
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        results = retriever.retrieve(query)
        self.redis.setex(cache_key, ttl, json.dumps(results))
        
        return results
```

### Monitoring

```python
class RAGMonitor:
    def __init__(self, metrics_client):
        self.client = metrics_client
    
    def record_retrieval(self, query: str, num_results: int, latency: float):
        self.client.gauge("rag.retrieval.latency", latency)
        self.client.gauge("rag.retrieval.results_count", num_results)
    
    def record_generation(self, prompt: str, response: str, latency: float):
        self.client.gauge("rag.generation.latency", latency)
        self.client.gauge("rag.generation.response_length", len(response))
    
    def record_retrieval_quality(self, query: str, relevance_scores: List[float]):
        self.client.histogram(
            "rag.retrieval.relevance_score",
            relevance_scores
        )
```

### Fallbacks

```python
class RAGWithFallback:
    def __init__(self, primary_rag, fallback_rag):
        self.primary = primary_rag
        self.fallback = fallback_rag
    
    def generate(self, query: str) -> str:
        try:
            return self.primary.generate(query)
        except VectorStoreError:
            return self.fallback.generate(query)
        except Exception as e:
            log_error(e)
            return self.primary.generate_simplified(query)
```

Building production RAG systems requires attention to retrieval quality, context construction, latency, and reliability. The techniques here provide a foundation for building systems that work reliably at scale.