---
title: Graph Foundation Models
description: Explore graph foundation models (GFMs) — large pretrained models designed to generalize across diverse graph-structured domains. Covers pretraining objectives (masked node/edge prediction, contrastive learning, graph-level self-supervision), architecture choices (Graph Transformers, GPS, Graphormer), cross-domain transfer challenges, and leading systems including OFA, PRODIGY, and UniGraph.
---

**Graph foundation models (GFMs)** are large pretrained neural networks designed to serve as general-purpose encoders and reasoners over graph-structured data — analogous to what large language models (LLMs) are for text, or vision foundation models for images. The goal is to train a single model on diverse graph datasets (molecular graphs, social networks, knowledge graphs, code dependency graphs, citation networks) and then fine-tune or prompt-adapt it for downstream graph tasks without training from scratch.

Graph data is ubiquitous: atoms and bonds form molecular graphs, proteins fold into contact graphs, web pages are linked graphs, and programs are abstract syntax trees. Yet unlike natural language (which can be tokenized uniformly) or images (which are fixed grids), graphs vary enormously in node/edge semantics, degree distributions, sizes, and attribute types — making generalist pretraining substantially more challenging.

## Why Graphs Are Harder to Generalize Across Domains

A single graph Foundation model must overcome several structural obstacles:

- **Heterogeneous feature spaces**: a molecular graph node is an atom with electronic properties; a social graph node is a user with text attributes; a knowledge graph node is an entity with an embedding. There is no shared token vocabulary.
- **Variable topology**: graphs range from trees (parse trees) to dense cliques (protein contact maps) to sparse power-law networks (web graphs), each requiring different inductive biases.
- **Task diversity**: node classification, link prediction, graph classification, graph regression, subgraph matching, and graph generation require fundamentally different output heads.
- **Evaluation transfer gap**: a model pretrained on citation networks may not capture the geometric reasoning needed for molecular property prediction.

## Pretraining Objectives

### Masked Node/Edge Prediction (MNEP)

Following the BERT masked language modeling paradigm, MNEP randomly masks a fraction (typically 15-30%) of node features or edge attributes and trains the model to reconstruct them from context:

$$\mathcal{L}_\text{MNEP} = -\sum_{v \in \mathcal{V}_\text{masked}} \log p_\theta(\mathbf{x}_v \mid \mathcal{G} \setminus \mathbf{x}_v)$$

This forces the model to aggregate multi-hop neighborhood information and learn structural context. Used in **GraphMAE** and **GraphBERT** variants.

### Graph Contrastive Learning

Graph contrastive pretraining learns representations that are invariant to semantic-preserving augmentations and discriminative across different graphs. Two views $\mathcal{G}^+_1, \mathcal{G}^+_2$ of the same graph are constructed (via node dropout, edge perturbation, or subgraph sampling) and trained with InfoNCE:

$$\mathcal{L}_\text{CL} = -\log \frac{\exp(\text{sim}(z_1, z_2)/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_1, z_k)/\tau)}$$

Used in **GraphCL**, **MVGRL**, and **SimGRACE**. Key challenge: choosing augmentations that preserve semantic meaning (atom dropping may change molecular properties; social tie removal may not).

### Graph-Level Self-Supervision

For cross-graph pretraining, auxiliary tasks operating at the graph level improve transferability:

- **Graph-level contrastive**: treat graphs from the same domain as positive pairs, cross-domain as negatives.
- **Context prediction**: predict properties of a graph's $k$-hop neighborhood from its center subgraph (used in **Hu et al., 2020** molecular pretraining).
- **Motif prediction**: predict the presence of chemical motifs (rings, functional groups) or social structural motifs (triangles, stars) as weak labels.

## Architecture: Graph Transformers

Standard message-passing GNNs (GCN, GAT, GraphSAGE) have limited expressivity and struggle with long-range dependencies. Graph Transformers apply full self-attention over all node pairs, modulated by structural biases:

$$\mathbf{h}_v^{(l+1)} = \text{Attn}\left( \mathbf{q}_v, \{ \mathbf{k}_u \}_{u \in \mathcal{V}}, \{ \mathbf{v}_u \}_{u \in \mathcal{V}} \right) + \mathbf{b}_{d(v,u)}$$

where $\mathbf{b}_{d(v,u)}$ is a learned bias encoding the shortest-path distance between $v$ and $u$. This allows the model to directly attend to distant nodes while respecting graph topology.

### Graphormer

**Graphormer** (Ying et al., 2021) introduced spatial encoding and edge encoding into the Transformer attention:

- **Spatial encoding**: bias attention scores by $\phi(d(v,u))$ where $d$ is shortest-path distance, learned as a scalar lookup.
- **Edge encoding**: incorporate edge features along the shortest path between node pairs as additive attention biases.
- **Centrality encoding**: use in-degree and out-degree as additional node embeddings.

Graphormer achieved state-of-the-art on the OGB-LSC quantum chemistry challenge (PCQM4Mv2).

### GPS (General, Powerful, Scalable)

**GPS** (Rampášek et al., 2022) combines local MPNN layers (for structural inductive bias) with global Transformer attention (for long-range signals) in a hybrid architecture:

$$\mathbf{h}^{(l+1)} = \text{MPNN}^{(l)}(\mathbf{h}^{(l)}) + \text{MHA}^{(l)}(\mathbf{h}^{(l)})$$

This is more scalable than full-graph Transformer ($O(n^2)$) while retaining global context. GPS++ extended this for large molecular property prediction tasks.

## Cross-Domain Transfer Approaches

### Feature Alignment

When node features differ across domains, a shared **feature projector** maps domain-specific features into a common embedding space:

$$\tilde{\mathbf{x}}_v = W_d \cdot \mathbf{x}_v + b_d$$

where $W_d$ is a domain-specific linear projection. The backbone GFM then operates in the shared embedding space.

### Text-Attributed Graphs (TAG)

Many real-world graphs have **textual node attributes**: paper titles/abstracts in citation networks, product descriptions in e-commerce graphs, function names in code graphs. LLM encoders (BERT, LLaMA) can embed these text features into a shared semantic space, providing natural cross-domain alignment.

**GraphGPT** and **LLaGA** leverage this: encode node text with an LLM, then apply a graph encoder on top of the LLM node embeddings. The LLM vocabulary acts as a universal node feature space.

### In-Context Learning on Graphs

**PRODIGY** (Huang et al., 2023) introduces in-context learning for graphs: given a "prompt graph" containing labeled examples and a "query node," the model predicts the query label without fine-tuning — analogous to few-shot ICL in LLMs. Graphs and their labels are encoded into a unified in-context graph, and the model is pretrained to exploit this structure.

### One For All (OFA)

**OFA** (Liu et al., 2023) proposes a unified framework for training a single GNN on multiple graph classification, node classification, and link prediction tasks simultaneously. Key contributions:

- **Node-aligned feature space**: all node features are encoded through a shared text encoder using natural language descriptions.
- **NOI (Nodes of Interest) subgraph**: standardize multi-task learning by extracting a task-specific subgraph and labeling a subset of its nodes/edges as the prediction target.
- Demonstrated zero-shot transfer to unseen graph datasets.

### UniGraph

**UniGraph** (He et al., 2024) scales GFM pretraining to 38 diverse graph datasets spanning molecular, social, citation, and e-commerce domains. It uses a shared Transformer backbone with domain-aware positional encodings and achieves positive transfer across all domains, outperforming single-domain baselines on most tasks when fine-tuned.

## Limitations and Open Problems

- **Expressivity ceiling**: the Weisfeiler-Lehman (WL) graph isomorphism test bounds the expressive power of MPNN-based GFMs. Distinguishing non-isomorphic graphs requires either higher-order WL tests or structural identifiers — both computationally expensive.
- **Scalability**: full-graph Transformer attention is $O(n^2)$ in node count. Most large real-world graphs (billions of nodes) require approximations (e.g., sampled subgraphs, sparse attention) that sacrifice global context.
- **Negative transfer**: a model trained across heterogeneous graph domains can suffer negative transfer if domain distributions are too dissimilar. Task-specific adapter layers partially mitigate this.
- **Evaluation benchmarks**: unlike language (GLUE, SuperGLUE, BIG-Bench) and vision (ImageNet, COCO), there is no universally accepted GFM benchmark spanning diverse graph types and tasks. OGB and LRGB are popular but domain-limited.

## Summary

Graph foundation models adapt the large-scale pretraining paradigm to graph-structured data. Key pretraining objectives include masked node prediction, graph contrastive learning, and motif self-supervision. Architectures like Graphormer and GPS extend Transformers to graphs with structural positional encodings. Cross-domain generalization is achieved via text-attributed node features, feature alignment projectors, and in-context learning (PRODIGY, OFA, UniGraph). While promising, GFMs still face challenges in expressivity, scalability to billion-node graphs, and negative transfer — making them an active area of research at the intersection of graph learning, self-supervised learning, and large-scale pretraining.
