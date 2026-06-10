---
title: Graph Neural Networks for Knowledge Graphs
description: Reasoning over structured knowledge — embedding entities and relations, knowledge graph completion, and link prediction.
---

**Knowledge graphs** represent facts as triples: (subject, relation, object). Example: (Einstein, discoveredTheory, Relativity). They capture structured knowledge at scale — Google's Knowledge Graph has billions of entities, Wikidata has millions.

**Knowledge graph embeddings** learn vector representations of entities and relations such that true facts score higher than false ones. This enables reasoning, knowledge graph completion (predicting missing facts), and information retrieval.

## Knowledge Graphs

### Representation

A knowledge graph $\mathcal{G} = (E, R, T)$ consists of:
- **Entities** $E$: Objects (Einstein, Newton, Relativity).
- **Relations** $R$: Relationships (discoveredTheory, birthPlace, influenced).
- **Triples** $T \subseteq E \times R \times E$: Facts (Einstein, birthPlace, Ulm).

Heterogeneous knowledge graphs have multiple entity and relation types.

### Applications

- **Link prediction**: Predict missing edges. Who did Einstein influence? Who influenced Darwin?
- **Entity alignment**: Match entities across knowledge graphs (same person in DBpedia and Wikidata).
- **Recommendation**: Recommend products by reasoning over user-product-feature knowledge graphs.
- **Question answering**: Answer queries by retrieving and reasoning over knowledge graph paths.

## Embedding Methods

### TransE

**TransE** models relations as translations in embedding space:

For a true triple $(h, r, t)$, embeddings should satisfy:

$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

Train by minimizing:

$$\mathcal{L} = \sum_{(h,r,t) \in T} \sum_{(h',r,t') \in T'} [\gamma + d(\mathbf{h} + \mathbf{r}, \mathbf{t}) - d(\mathbf{h'} + \mathbf{r}, \mathbf{t'})]_+$$

where $T'$ are corrupted triples (negative samples), $d$ is distance (e.g., L2), and $[\cdot]_+$ is the hinge loss.

**Simplicity**: Easy to train; efficient.

**Limitation**: Can't model symmetric relations (if $r$ is symmetric, all entities should be equivalent), compositional relations (path reasoning), or 1-N relations (one head, many tails).

### RotatE

Model relations as rotations in complex space:

$$\mathbf{t} = \mathbf{r} \odot \mathbf{h}$$

where $\odot$ is element-wise multiplication in complex space. Relations represent rotations; composing rotations models relation composition (e.g., parent of parent = ancestor).

**Advantages**:
- Can represent symmetric ($r = 1$), antisymmetric ($r = -1$), inverse ($r = 1/r$), and compositional relations.
- Theoretically motivated by rotation groups.

### Graph Convolutional Networks for KG Embedding

**CompGCN** combines GCN-style message passing with knowledge graph structure:

1. **Initialize**: Entity embeddings and relation embeddings from embedding methods.
2. **Message passing**: Aggregate information from neighbors using relation-specific transformations.
3. **Scoring**: Predict likelihood of triples using aggregated embeddings.

Combines structural (graph) and relational (embedding) information.

## Link Prediction

### Task

Given $(h, r, ?)$, predict the missing tail. Or $(?, r, t)$, predict the missing head.

**Evaluation**:
- **Hits@10**: Fraction of correct answers ranked in top-10.
- **MRR (Mean Reciprocal Rank)**: Average rank of correct answer.

### Candidate Ranking

For each triple, score candidates:

$$\text{score}(h, r, t) = -d(\mathbf{h} + \mathbf{r}, \mathbf{t})$$

For link prediction $(h, r, ?)$, rank all entities $t$ by score; return top-k.

### Negative Sampling

Training is expensive: for each positive triple, compare against all entities. Efficient approaches:

**Corrupt and score**: Replace tail with random entity; score should be low.

**Adversarial sampling**: Sample hard negatives (entities with similar embeddings to correct answer) — more informative than random negatives.

## Knowledge Graph Completion

Complete a partial knowledge graph by predicting missing triples.

### Why It's Hard

- **Sparsity**: Real knowledge graphs have millions of entities but billions of possible triples; most are unobserved.
- **Ambiguity**: Multiple plausible completions (Einstein influenced many).
- **Temporal dynamics**: Facts change (person's employment, location).

### Evaluation

**Held-out test set**: Remove 10% of triples; train on 90%; predict held-out triples.

**Filtered metrics**: When ranking candidates, exclude triples already in the knowledge graph (wouldn't be errors).

## Temporal Knowledge Graphs

Knowledge graphs evolve: facts become true/false over time.

### Temporal Reasoning

Extend embeddings to include time:

$$\text{score}(h, r, t, \tau) = -d(\mathbf{h}(\tau) + \mathbf{r}(\tau), \mathbf{t}(\tau))$$

Embeddings vary with time; can model how entities and relations evolve.

### Applications

- **Event prediction**: Predict future events (stock price changes, political conflicts).
- **Temporal consistency**: Ensure reasoned facts respect temporal constraints.

## Multimodal Knowledge Graphs

Incorporate images, descriptions, or other modalities alongside triples.

### Multimodal Embeddings

Learn embeddings that align structured triples with images/text:

$$\mathcal{L} = d(\text{TextEmbedding}(h), \text{ImageEmbedding}(h))$$

Align modalities in a shared embedding space.

## Reasoning and Path-Based Methods

Rather than single-hop prediction, reason over paths.

### Multi-Hop Reasoning

Query: (Einstein, ?, Darwin) — what path connects them?

Paths: Einstein → influencedBy ← Newton → influencedBy ← Descartes → ...

Find paths through knowledge graph that answer queries.

### Reinforcement Learning for Path Finding

Treat path finding as RL:
- **State**: Current entity.
- **Action**: Choose a relation to follow.
- **Reward**: Reach target entity.

Train an agent to find paths; enables complex reasoning (not just link prediction).

## Challenges

### Cold Start Problem

Newly added entities have few triples. Embeddings are unreliable; need transfer from similar entities or multimodal information.

### Out-of-Distribution Entities

Entities not in training graph are unobserved; can't compute embeddings. Transfer learning or zero-shot approaches (from descriptions) help.

### Temporal Knowledge Graphs

Most methods ignore time; extending to temporal graphs is complex.

### Evaluation Limitations

Filtered metrics only count predicted triples as errors if they're already in the graph. But absence of an edge doesn't mean falsehood (incomplete knowledge).

## Applications

### Recommendation Systems

Knowledge graphs encode user-product-feature relations. Graph embedding + link prediction recommends products.

### Question Answering

Answer queries like "Who influenced Darwin?" by finding paths in the knowledge graph.

### Scientific Discovery

Predict new drug-disease interactions, protein interactions, or scientific relationships by embedding biological knowledge graphs.

### Search and Information Retrieval

Enrich search results with knowledge graph facts and related entities.

## Current Trends

- **Hybrid methods**: Combine GNNs with embedding methods.
- **Neuro-symbolic approaches**: Integrate neural embeddings with logic rules for interpretable reasoning.
- **Scaling to billions of entities**: Efficient methods for web-scale knowledge graphs.
- **Multimodal and temporal**: Modern knowledge graphs are multimodal and evolve over time.

Knowledge graph embeddings have become indispensable for structured reasoning, enabling AI systems to leverage vast repositories of organized knowledge for prediction and inference.
