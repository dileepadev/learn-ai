---
title: Graph Neural Networks - Learning on Graph-Structured Data
description: Understanding graph neural networks and their applications to network data.
---

Many real-world problems involve graph-structured data: social networks, molecules, recommendation systems, knowledge bases. Graph Neural Networks (GNNs) extend neural networks to learn from graphs. This post explores GNNs and their applications.

## What is Graph-Structured Data?

Data represented as networks of connected nodes.

**Components:**
- **Nodes:** Entities (people, molecules, web pages)
- **Edges:** Relationships (friendships, bonds, links)
- **Features:** Properties of nodes/edges

**Examples:**
```
Social Network: Nodes=users, Edges=friendships
Molecules: Nodes=atoms, Edges=chemical bonds
Web: Nodes=pages, Edges=links
Recommendation: Nodes=users/items, Edges=interactions
```

## Why Not Regular Neural Networks?

Regular neural networks assume fixed-size, ordered inputs.

**Graph Problems:**
- Variable number of neighbors
- Unordered neighbor sets
- Rich relational information
- Varying graph sizes

**Example:**
```
User A has 5 friends
User B has 500 friends

Different input sizes → Regular NN fails
Graph structure → GNN handles naturally
```

## Graph Representations

### Adjacency Matrix

Matrix representation of connections.

```
Users: A, B, C, D
       A  B  C  D
A    [0, 1, 1, 0]  (A connected to B, C)
B    [1, 0, 1, 0]  (B connected to A, C)
C    [1, 1, 0, 1]  (C connected to A, B, D)
D    [0, 0, 1, 0]  (D connected to C)
```

**Pros:** Simple, standard linear algebra

**Cons:** 
- Sparse graphs → Wasteful storage
- Unscalable for large graphs
- Permutation sensitive

### Edge List

List of connections.

```
Edge List:
(A, B)
(A, C)
(B, C)
(C, D)
```

**Pros:** Efficient for sparse graphs

**Cons:** Less convenient for computation

## Message Passing Framework

Core concept behind most GNNs.

### Basic Idea

Each node learns from its neighbors.

```
Node X aggregates information from neighbors
Updates representation
Repeats for multiple layers
Result: Node representations capture local structure
```

### Message Passing Steps

**1. Neighbor Aggregation**

Gather information from neighbors.

```
Neighbors of Node X: A, B, C
Messages: Features of A, B, C
```

**2. Aggregation Function**

Combine neighbor information.

```
Options:
- Sum: Add all neighbor features
- Mean: Average neighbor features
- Max: Maximum across neighbors
- Learned: Neural network combination
```

**3. Update Function**

Update node representation.

```
New_representation(X) = Update_function(Old_representation(X), Aggregated_neighbors)
```

**4. Repeat**

Multiple layers for multi-hop influence.

```
Layer 1: Direct neighbors
Layer 2: Neighbors' neighbors
Layer 3: 3-hop neighbors
Deeper layers → Larger receptive field
```

## Types of Graph Neural Networks

### Graph Convolutional Networks (GCN)

Convolution on graphs.

**Approach:**
```
Each node: Convolve with neighbor features
Like image convolution but on graph
```

**Formula:**
```
H_{l+1} = σ(D^{-1/2} A D^{-1/2} H_l W_l)

Where:
A = Adjacency matrix
D = Degree matrix (how many neighbors)
H_l = Node representations at layer l
W_l = Learnable weights
σ = Activation function
```

**Benefit:** Simple, efficient, effective

### GraphSAGE (Graph SAmple and aggreGatE)

Sampling-based approach for scalability.

**Key Idea:**
```
Instead of using all neighbors (expensive)
Sample random subset
Aggregate from sample
```

**Process:**
```
For each node:
    Sample K random neighbors
    Aggregate features
    Learn from sample
```

**Advantage:**
- Scales to large graphs
- Efficient mini-batch training
- Works on unseen nodes

### Graph Attention Networks (GAT)

Use attention to weight neighbors.

**Approach:**
```
Not all neighbors equally important
Learn attention weights
Important neighbors: Higher weight
Less important: Lower weight
```

**Attention Mechanism:**
```
For each node's neighbor:
    Calculate attention score
    Based on node and neighbor features
Normalize across neighbors
Aggregate with attention weights
```

**Benefit:**
- Learns which neighbors matter
- Interpretable (see attention weights)
- Flexible

### Recurrent GNNs

Use recurrence for sequential updates.

```
Repeatedly apply message passing
Until convergence or fixed iterations
Node representations stabilize
```

## Graph-Level Tasks

### Node Classification

Classify individual nodes.

```
Task: Predict node label
Example: Social network → Predict user interests
```

**Training:**
- Some nodes labeled
- Predict labels for unlabeled
- Use message passing to propagate information

### Link Prediction

Predict missing edges.

```
Task: Will these nodes connect?
Example: Recommendation → Predict user-item interaction
```

**Approach:**
```
Get node embeddings
Score potential edges
High score → Likely edge
```

### Graph Classification

Classify entire graphs.

```
Task: Classify graph as whole
Example: Molecule classification → Predict properties
```

**Process:**
1. Node-level features from GNN
2. Graph-level aggregation (pooling)
3. Classification

### Community Detection

Find clusters/communities.

```
Task: Group similar nodes
Example: Social network → Find communities
```

**Approach:**
- Node embeddings capture similarity
- Clustering on embeddings
- Reveals community structure

## Graph Pooling

Aggregate nodes into higher-level representation.

### Global Pooling

Combine all node features.

```
Options:
- Sum: Add all node embeddings
- Mean: Average node embeddings
- Max: Maximum across nodes
```

### Hierarchical Pooling

Create coarse graph.

```
Assign nodes to clusters
Create super-nodes
Connect super-nodes
Repeat at higher levels
```

**Benefit:**
- Captures multi-scale structure
- Efficient computation
- Hierarchical understanding

## Applications

### Social Networks

**Tasks:**
- Friend recommendation
- Community detection
- Content recommendation
- Influence estimation

**Benefit:**
- Leverage friend relationships
- Network effects matter
- GNNs naturally capture structure

### Molecular Property Prediction

**Input:** Molecular graph (atoms=nodes, bonds=edges)

**Task:** Predict properties (toxicity, efficacy, etc.)

**Advantage:**
- Atoms and bonds encode structure
- Molecular graphs standardized
- GNNs learn chemical patterns
- Drug discovery acceleration

### Recommendation Systems

**Graph:**
- Users and items as nodes
- Interactions as edges
- Social relationships

**Task:**
- Recommend items to users
- Leverage both collaborative and content-based signals

**Benefit:**
- Combine user-item interactions with social graph
- Better recommendations than single-modality

### Knowledge Graphs

**Graph:**
- Entities as nodes
- Relations as edges

**Tasks:**
- Link prediction (infer missing relations)
- Entity classification
- Relation extraction

**Application:**
- Question answering
- Search enhancement
- Information extraction

### Protein Structure Prediction

**Graph:** Protein contact graph or interaction graph

**Task:** Predict 3D structure

**Benefit:**
- Captures spatial relationships
- Physics-informed
- Breakthrough results (AlphaFold)

### Traffic Flow Prediction

**Graph:**
- Road intersections as nodes
- Roads as edges
- Traffic sensor data

**Task:** Predict traffic flow

**Benefit:**
- Spatial dependencies (neighboring roads affect each other)
- Temporal + spatial modeling
- Better traffic prediction

## Challenges in GNNs

### Over-Smoothing

Repeated aggregation causes representations to converge.

```
Layer 1: Different node representations
Layer 2: Some nodes more similar
Layer 3: Nodes very similar
Layer K: All nodes nearly identical
Problem: Information loss from aggregation
```

**Solutions:**
- Skip connections (bypass layers)
- Residual networks on graphs
- Deeper architectures carefully designed

### Scalability

Large graphs don't fit in memory.

```
Social network: Billions of nodes, trillions of edges
Full computation infeasible
```

**Solutions:**
- GraphSAGE sampling
- Mini-batch training
- Distributed training
- Graph partitioning

### Heterogeneous Graphs

Multiple node types and edge types.

```
E-commerce: Users, products, sellers (different node types)
Purchases, ratings, follows (different edge types)
```

**Solution:** Heterogeneous GNNs handle multiple types

### Dynamic Graphs

Graphs change over time.

```
Social network: New friendships form
Recommendation: New interactions occur
```

**Approach:**
- Temporal GNNs
- Continuous updates
- Learn temporal patterns

## GNN Architectures

### Simple GCN

```
Input Features
    ↓
GCN Layer (aggregate from neighbors)
    ↓
ReLU Activation
    ↓
GCN Layer
    ↓
Output (predictions)
```

### Complex Architecture

```
Input Features
    ↓
Multiple GNN Layers with residual connections
    ↓
Graph Pooling (coarsen graph)
    ↓
Multiple GNN Layers
    ↓
Graph-level Readout
    ↓
Output
```

## Implementation Frameworks

### PyTorch Geometric

Most popular GNN library for PyTorch.

```python
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(10, 16)
        self.conv2 = GCNConv(16, 2)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

### DGL (Deep Graph Library)

Framework-agnostic GNN library.

```python
import dgl
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feat, h_feat)
        self.conv2 = dglnn.GraphConv(h_feat, out_feat)
```

### Spektral

For TensorFlow/Keras.

### Jraph

Google's JAX-based graph neural networks.

## Conclusion

Graph Neural Networks extend deep learning to graph-structured data. Message passing enables nodes to learn from neighbors, capturing local and global graph structure. Different architectures—GCNs, GraphSAGE, GATs—provide various tradeoffs. Applications span social networks, molecules, recommendations, and knowledge graphs. While challenges like over-smoothing and scalability exist, GNNs have proven remarkably effective. As graph-structured data becomes increasingly prevalent and GNN techniques improve, they'll play growing roles in AI systems. From molecular discovery to social understanding to recommendation engines, GNNs unlock value from relational data.
