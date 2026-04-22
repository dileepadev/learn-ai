---
title: Graph Machine Learning
description: A comprehensive overview of machine learning on graph-structured data — covering node classification, link prediction, graph classification, and message-passing frameworks, with applications from social networks to molecular biology.
---

**Graph machine learning** is the application of machine learning to data that is naturally represented as **graphs** — sets of nodes connected by edges. Graphs are a universal data structure: social networks, knowledge graphs, molecular structures, citation networks, road maps, supply chains, and biological interaction networks all have a natural graph representation. Specialized ML methods for graphs exploit this relational structure to learn representations that capture not just node attributes but also the patterns of connection.

This field is broader than **graph neural networks** (GNNs) — it encompasses the full spectrum of tasks, methods, and applications, including classical graph-theoretic feature engineering, spectral methods, message-passing GNNs, and scalable inference on billion-node graphs.

## Graph Fundamentals

A graph $G = (V, E)$ consists of:

- **Nodes** (vertices) $V$: Entities — users, atoms, documents, or transactions.
- **Edges** $E \subseteq V \times V$: Relationships — friendships, bonds, citations, or money transfers.
- **Node features** $X \in \mathbb{R}^{|V| \times d}$: Attribute vectors for each node (age, element type, word embedding).
- **Edge features**: Optional attributes on edges (relationship type, bond order, transaction amount).

Graphs can be:

- **Directed or undirected**: Edges may or may not have directionality.
- **Weighted or unweighted**: Edges may carry numerical weights.
- **Homogeneous or heterogeneous**: All nodes/edges may share one type, or multiple node/edge types may coexist (knowledge graphs).
- **Bipartite**: Two disjoint node sets with edges only between sets (users and items in recommendation).
- **Temporal**: Edges and nodes appear and disappear over time.

## Graph ML Tasks

Graph ML problems are organized by the **level of prediction**:

### Node-Level Tasks

**Node classification**: Predict the label (class) of each node based on its features and neighborhood structure.

- Community membership in social networks.
- Protein function prediction in protein interaction networks.
- Document category prediction in citation graphs.

**Node regression**: Predict a continuous property for each node (e.g., influence score, energy).

**Node clustering**: Identify natural groupings of nodes without labels.

### Edge-Level Tasks

**Link prediction**: Predict whether an edge exists between two nodes — useful for recommendation systems (will this user buy this item?), knowledge graph completion (is entity A related to entity B?), and drug-target interaction prediction.

**Edge classification**: Classify the type or property of an existing edge.

**Relation prediction**: In knowledge graphs, predict the type of relationship between two entities.

### Graph-Level Tasks

**Graph classification**: Classify an entire graph into a category — used for molecule property prediction (is this compound toxic?), program analysis (does this code contain a bug?), and social network analysis.

**Graph regression**: Predict a scalar property of a graph — drug potency, material conductivity, reaction yield.

**Graph generation**: Generate new graphs with desired properties — drug molecule design, material structure synthesis.

## Classical Graph Features

Before deep learning, graph ML relied on **hand-crafted structural features**:

### Node-Level Features

- **Degree**: Number of connections. High-degree nodes are hubs.
- **Clustering coefficient**: Fraction of a node's neighbors that are also connected to each other — measures local density.
- **PageRank**: Probability of reaching a node via a random walk — measures global importance.
- **Betweenness centrality**: How often a node lies on the shortest path between two other nodes.
- **Eigenvector centrality**: Recursive notion of importance — a node is important if it is connected to important nodes.

### Graph-Level Features

- **Graph diameter**: Longest shortest path.
- **Spectral features**: Eigenvalues of the graph Laplacian.
- **Graphlet count**: Frequency of small subgraph patterns (triangles, 4-cycles, stars).
- **Weisfeiler-Lehman (WL) subtree features**: Iteratively aggregate and hash neighborhood labels to produce a structural fingerprint. The WL test is also the theoretical foundation for analyzing the expressive power of GNNs.

## Message-Passing Graph Neural Networks

The dominant GNN paradigm is **message passing** (Gilmer et al., 2017): nodes iteratively aggregate information from their neighbors to update their representations.

At each layer $l$, the update for node $v$ is:

$$h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

Different GNN architectures differ in how they implement AGGREGATE and UPDATE.

### Graph Convolutional Network (GCN)

**GCN** (Kipf & Welling, 2017):

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

where $\hat{A} = A + I$ (adjacency matrix with self-loops) and $\hat{D}$ is the degree matrix. This is a symmetric normalized aggregation: each node's new representation is the mean of its neighbors' representations (plus itself), weighted by degree.

### Graph Attention Network (GAT)

**GAT** (Veličković et al., 2018) uses learned attention weights to aggregate neighbors — not all neighbors are equally important:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(l)} W^{(l)} h_u^{(l)}\right)$$

where $\alpha_{vu}$ is the attention weight between nodes $v$ and $u$, computed by a small attention network applied to the concatenation of their features.

Multi-head attention (as in the original transformer) is used to stabilize learning and capture different types of relationships.

### GraphSAGE

**GraphSAGE** (Hamilton et al., 2017) enables **inductive** learning — applying a trained GNN to new, unseen graphs. Rather than learning embeddings for specific nodes, GraphSAGE learns an aggregation function:

1. Sample a fixed-size neighborhood for each node.
2. Aggregate sampled neighbors' features.
3. Concatenate with the node's own features and apply a linear transformation.

The sampling step enables mini-batch training on large graphs where processing the full neighborhood is infeasible.

### Graph Isomorphism Network (GIN)

**GIN** (Xu et al., 2019) is theoretically motivated: it is the most expressive GNN architecture achievable with the message-passing framework (as powerful as the WL graph isomorphism test). The aggregation is:

$$h_v^{(l+1)} = \text{MLP}^{(l)}\left((1 + \epsilon^{(l)}) h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} h_u^{(l)}\right)$$

The sum aggregation (rather than mean or max) preserves the multiplicity of node features, which is key to distinguishing non-isomorphic graphs.

## Scalability Challenges

A fundamental challenge for GNNs is the **neighborhood explosion problem**: computing exact $K$-hop neighborhoods for a node requires materializing an exponentially growing number of nodes as $K$ increases. For a node with average degree $d$ and $K$ layers, the $K$-hop neighborhood has up to $d^K$ nodes.

**Sampling-based methods** address this:

- **GraphSAGE neighborhood sampling**: Sample a fixed number of neighbors at each hop.
- **FastGCN** (importance sampling): Sample nodes from all layers with importance weights derived from the graph structure.
- **LADIES** (Layer-Dependent Importance Sampling): Samples nodes layer by layer with importance weights that account for the previous layer's sampled nodes.

**Cluster-based methods**:

- **ClusterGCN**: Partition the graph into clusters using METIS or random partitioning. Mini-batches consist of one or more clusters. Avoids neighborhood explosion by processing subgraphs.
- **GraphSAINT**: Samples entire subgraphs (node samplers, edge samplers, random walk samplers) and trains on the induced subgraph — enabling unbiased estimation of the full-graph GNN.

## Graph Transformers

**Graph transformers** apply the transformer architecture to graphs, addressing the key limitation of message-passing GNNs: information can only travel one hop per layer, requiring many layers to capture long-range dependencies.

In graph transformers, every node attends to every other node (or a sampled subset), with graph structure encoded in the attention bias or positional encoding:

- **Graphormer** (Microsoft): Adds structural encodings based on node degree and shortest-path distance to the transformer's attention mechanism.
- **GPS (General, Powerful, Scalable)**: Combines local message passing with global transformer attention at each layer — capturing both local graph structure and long-range dependencies efficiently.

## Knowledge Graphs and Embeddings

**Knowledge graphs** represent factual knowledge as a set of (entity, relation, entity) triples. Knowledge graph completion — predicting missing triples — is a major application.

**Knowledge graph embedding methods** learn embeddings for entities and relations such that true triples score higher than false ones:

- **TransE**: Models relations as translations — for a true triple $(h, r, t)$, the entity embeddings should satisfy $h + r \approx t$.
- **RotatE**: Models relations as rotations in complex space, enabling representation of symmetric, antisymmetric, invertible, and compositional relations.
- **DistMult / ComplEx**: Bilinear scoring functions for entity-relation-entity triples.

**GNN-based KG completion** (CompGCN, R-GCN) uses message passing to incorporate graph structure into entity embeddings, outperforming pure embedding methods.

## Graph ML for Molecules and Drug Discovery

Molecular graphs — where nodes are atoms and edges are bonds — are one of the most impactful application domains for graph ML:

- **Property prediction**: Predicting molecular properties (toxicity, solubility, bioactivity) from graph structure, enabling virtual screening of compound libraries.
- **Reaction prediction**: Predicting the products of chemical reactions from reactant graphs.
- **Drug-target interaction**: Predicting whether a compound binds to a target protein.
- **De novo drug design**: Generating novel molecular graphs with desired property profiles.

**MPNN (Message Passing Neural Networks)** (Gilmer et al., 2017) was specifically designed for quantum chemistry prediction and set new records on the QM9 benchmark — predicting 12 molecular properties from atomic graph structure.

## Evaluation and Benchmarks

| Benchmark | Task | Domain |
|-----------|------|--------|
| **Open Graph Benchmark (OGB)** | Node, link, graph classification | Multiple domains |
| **Cora / CiteSeer / PubMed** | Node classification | Citation networks |
| **TUDatasets** | Graph classification | Biochemistry, social |
| **QM9** | Graph regression (12 properties) | Quantum chemistry |
| **ZINC** | Graph regression (penalized logP) | Drug discovery |
| **ogbl-collab** | Link prediction | Collaboration networks |

**OGB** (Hu et al., 2020) provides large-scale, realistic benchmarks with standardized train/val/test splits to prevent data leakage — addressing reproducibility issues in earlier graph ML research.

## Practical Considerations

**Node features matter more than structure** in many practical applications. Adding good node features (e.g., pretrained language embeddings for text nodes) often provides a larger improvement than switching GNN architectures.

**Depth vs. over-smoothing**: Deep GNNs suffer from **over-smoothing** — after many message-passing layers, all node representations converge to the same value, losing the discrimination between nodes. In practice, 2–4 layers is optimal for most tasks. Residual connections, jumping knowledge networks, and normalization layers mitigate this.

**Heterogeneous graphs**: Many real-world graphs have multiple node and edge types. HAN (Heterogeneous Attention Network), HetGNN, and HGT are specialized architectures for heterogeneous graphs.

**Dynamic graphs**: When the graph evolves over time, temporal GNNs (TGAT, TGN) extend message passing to incorporate temporal information from the sequence of interactions.

Graph machine learning is one of the most active areas of ML research, driven by high-impact applications in drug discovery, materials science, fraud detection, and recommendation — domains where relational structure is essential and graph ML provides capabilities unavailable to standard neural architectures.
