---
title: "Tensor Network Methods for Machine Learning"
description: Explore how tensor networks — Matrix Product States, MERA, PEPS, and Tucker decompositions — are applied to machine learning for model compression, quantum-inspired algorithms, and representing high-dimensional distributions that are intractable for standard neural networks.
---

A neural network with $n$ parameters is a point in an $n$-dimensional space. Training is the search for a point in this space that minimizes a loss function. For modern networks, $n$ ranges from millions to trillions — a daunting high-dimensional space.

Tensor networks offer a fundamentally different perspective. Originating in quantum physics for simulating many-body quantum systems, they represent high-dimensional objects not as lists of parameters but as networks of interconnected smaller tensors. The structure of the network encodes inductive biases about the problem; the parameters are the tensor entries.

Applied to machine learning, tensor networks provide tools for model compression, probabilistic generative modeling, and handling structured high-dimensional data — with formal guarantees that are rare in deep learning.

## What Is a Tensor Network?

A **tensor** is a multidimensional array. A scalar is a 0-tensor, a vector is a 1-tensor, a matrix is a 2-tensor, and higher-order tensors have more indices. An order-$n$ tensor $T_{i_1 i_2 \cdots i_n}$ has $d^n$ entries for $d$-dimensional indices.

A **tensor network** is a set of tensors connected by shared contracted indices — indices that are summed over. The structure of connections defines the network topology, which encodes the computational and representational properties of the model.

**Example — matrix multiplication as tensor contraction:**

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

Here $j$ is a contracted (shared) index between $A$ and $B$. The result $C$ has free indices $i$ and $k$. This is a simple two-tensor network.

In tensor networks for ML, tensors are arranged in specific topological patterns — chains, trees, hierarchies — each with different expressiveness and computational properties.

## Matrix Product States (MPS) — Tensor Trains

The simplest non-trivial tensor network is the **Matrix Product State (MPS)**, also called a **Tensor Train** in the ML literature.

A vector in $\mathbb{R}^{d^n}$ (representing, for example, all $d^n$ possible assignments to $n$ variables) is represented as:

$$v_{i_1 i_2 \cdots i_n} = \sum_{\alpha_1, \ldots, \alpha_{n-1}} A^{[1]}_{i_1, \alpha_1} A^{[2]}_{\alpha_1, i_2, \alpha_2} \cdots A^{[n]}_{\alpha_{n-1}, i_n}$$

where each $A^{[k]}$ is a small 3-tensor (left bond, physical index, right bond), and the $\alpha$ indices (bond indices) are summed over.

The key parameter is the **bond dimension** $\chi$: the maximum size of the bond indices. A full tensor has $d^n$ parameters; the MPS representation has only $O(n \cdot d \cdot \chi^2)$ parameters. This is an exponential compression — but only represents tensors with limited **entanglement** (correlations between distant parts of the tensor).

### Why This Matters for ML

MPS imposes an inductive bias: **sequential locality**. Variables at the extremes of the chain interact only through the bond, which is a bottleneck. This is appropriate for:
- **Sequential data** (time series, sequences) where short-range correlations dominate
- **Images scanned linearly** — though 2D locality is better captured by 2D networks
- **Probability distributions** that factor approximately locally

The bond dimension $\chi$ is a hyperparameter that controls the expressiveness vs. efficiency tradeoff. Small $\chi$ = fast but limited; large $\chi$ = slow but expressive.

### MPS in Practice

```python
import numpy as np

class MPS:
    def __init__(self, n_sites, d, bond_dim):
        """
        n_sites: number of physical sites (variables)
        d: physical dimension per site
        bond_dim: max bond dimension chi
        """
        self.tensors = []
        
        # First tensor: shape (d, chi)
        self.tensors.append(np.random.randn(d, bond_dim))
        
        # Middle tensors: shape (chi, d, chi)
        for _ in range(n_sites - 2):
            self.tensors.append(np.random.randn(bond_dim, d, bond_dim))
        
        # Last tensor: shape (chi, d)
        self.tensors.append(np.random.randn(bond_dim, d))
    
    def contract(self):
        """Contract the MPS to get the full tensor."""
        result = self.tensors[0]
        for tensor in self.tensors[1:]:
            result = np.tensordot(result, tensor, axes=1)
        return result
    
    def n_parameters(self):
        return sum(t.size for t in self.tensors)
```

## Tree Tensor Networks (TTN) and MERA

### Tree Tensor Networks

TTN arranges tensors in a binary tree. Each leaf node corresponds to a physical variable; internal nodes are tensors that contract their children and produce a coarser representation. The root tensor captures global correlations.

Properties:
- More effective than MPS for capturing correlations between distant parts of the input
- The tree structure means no cycles — efficient contraction is guaranteed
- Appropriate for data with hierarchical structure (document → paragraph → sentence → word)

### MERA (Multi-scale Entanglement Renormalization Ansatz)

MERA is a more complex network originally developed for simulating quantum critical systems. It adds **disentangler** tensors at each level of the hierarchy — tensors that reduce correlations between neighboring regions before coarse-graining.

In ML terms, MERA can be understood as a deep hierarchical model where:
- Each level of the hierarchy applies a learned transformation (the disentanglers)
- Then coarse-grains the representation by contracting pairs of tensors

MERA has a natural interpretation as a multi-scale feature extractor, analogous to a deep convolutional network but with a rigorous entanglement-based theory of its representational capacity.

## Tensor Networks for Model Compression

Neural network weight matrices can be replaced by tensor network decompositions, dramatically reducing parameter counts:

### Tensor Train Decomposition of Weight Matrices

A weight matrix $W \in \mathbb{R}^{m \times n}$ can be reshaped into a high-order tensor (e.g., $\mathbb{R}^{d_1 \times \cdots \times d_p \times e_1 \times \cdots \times e_q}$ where $d_1 \cdots d_p = m$ and $e_1 \cdots e_q = n$) and then approximated as a tensor train.

**Compression ratio:** For a fully connected layer with $m = n = 512$ ($m \times n = 262{,}144$ parameters), a TT decomposition with bond dimension $\chi = 4$ and $p = q = 4$ sub-dimensions of size 4 each requires only $4 \times 4^2 \times 4 \times 4 = 1{,}024$ parameters — a **256× compression** with modest accuracy loss.

**TT-Linear layer in PyTorch:**

```python
import torch
import torch.nn as nn

class TTLinear(nn.Module):
    """
    Tensor-Train factorized linear layer.
    Replaces a (prod(input_dims), prod(output_dims)) matrix
    with a tensor train.
    """
    def __init__(self, input_dims, output_dims, bond_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        n = len(input_dims)
        self.cores = nn.ParameterList()
        
        for i in range(n):
            chi_left = bond_dims[i]
            chi_right = bond_dims[i + 1]
            d_in = input_dims[i]
            d_out = output_dims[i]
            core = nn.Parameter(
                torch.randn(chi_left, d_in, d_out, chi_right) 
                * 0.01
            )
            self.cores.append(core)
    
    def forward(self, x):
        # Reshape input and contract through TT cores
        # (implementation details omitted for brevity)
        pass
```

**Applications:**
- Compressing embedding matrices in NLP models (huge savings for large vocabulary)
- Replacing dense layers in CNNs and transformers
- Compressing multi-head attention weight matrices

### Tucker Decomposition

Tucker decomposition factorizes a tensor as:

$$T_{i_1 \cdots i_n} = \sum_{j_1 \cdots j_n} G_{j_1 \cdots j_n} U^{(1)}_{i_1 j_1} \cdots U^{(n)}_{i_n j_n}$$

where $G$ is a core tensor and $U^{(k)}$ are factor matrices. For weight tensors in convolutional networks (4-tensor: out_channels × in_channels × height × width), Tucker decomposition reduces parameters significantly while maintaining spatial structure.

Higher-order SVD (HOSVD) computes Tucker decompositions similarly to computing matrix SVD.

## Tensor Networks as Generative Models

### Born Machines

A tensor network can parameterize a probability distribution by interpreting squared tensor entries as probabilities. For a binary vector $\mathbf{x} = (x_1, \ldots, x_n) \in \{0,1\}^n$:

$$p(\mathbf{x}) = |MPS(x_1, \ldots, x_n)|^2 / Z$$

where $MPS(\mathbf{x})$ is the scalar obtained by contracting the MPS with the input $\mathbf{x}$, and $Z$ normalizes to a probability distribution. This is called a **Born machine** by analogy with quantum mechanics.

**Advantages:**
- Exact computation of marginals and conditionals (via efficient contraction)
- No mode collapse issues that plague GANs
- Formal guarantees on representational power via bond dimension theory

**Limitations:**
- Exact sampling and normalization scale with $\chi^n$ for arbitrary topologies; practical implementations restrict to 1D or tree structures
- Cannot match the sample quality of diffusion models or GANs for images

### Quantum Circuit-Inspired Generative Models

The connection between tensor networks and quantum circuits — both mathematically described by tensor contractions — has inspired a class of quantum-inspired ML models that run on classical hardware:

- **Quantum-inspired sampling algorithms** for combinatorial problems
- **Quantum kernel methods** that use tensor network feature maps to compute inner products in exponentially large feature spaces efficiently

These approaches are particularly active in the context of quantum-classical hybrid computing, where near-term quantum devices have limited depth and coherence time.

## Tensor Networks and Transformers

Recent work has drawn connections between attention mechanisms and tensor networks:

**Self-attention as a tensor contraction:** The attention operation $\text{softmax}(QK^T / \sqrt{d}) V$ can be interpreted as a special case of tensor network contraction. This framing suggests new architectures that replace attention with different contraction patterns.

**MPS language models:** MPS representations of probability distributions over token sequences provide exact conditional distributions and tractable sampling. While they underperform large neural LMs, they offer interpretability and exact inference guarantees valuable for structured prediction tasks.

**Tensor decompositions for embedding compression:** Token embeddings in large LMs can be factorized using Tucker or TT decompositions, reducing memory and enabling larger vocabulary coverage with the same parameter budget.

## Practical Tradeoffs and When to Use Tensor Networks

| Application | Recommended TN type | Key benefit |
|-------------|-------------------|-------------|
| Sequential model compression | MPS/Tensor Train | Large parameter reduction, maintains locality |
| Conv filter compression | Tucker | Spatial structure preserved |
| Hierarchical data modeling | TTN | Multi-scale correlations |
| Probability estimation (discrete) | Born machines (MPS) | Exact inference, no sampling approximation |
| Embedding table compression | TT/Tucker | Very large compression ratios |
| Interpretability research | MPS | Bond dimension = information bottleneck |

Tensor networks are most valuable when:
1. **Strong structured compression is needed** and the structure matches the tensor network topology
2. **Exact probabilistic inference** is required (tensor networks support exact marginalization)
3. **Quantum-classical hybrid algorithms** are a target deployment platform

They are less appropriate when:
- Maximum raw performance on standard benchmarks is the goal (deep neural networks remain superior)
- Data has no clear spatial/sequential/hierarchical structure
- The tensor network topology significantly mismatches the data's correlation structure

The intersection of tensor network theory and machine learning is an active frontier, driven partly by theoretical interest in understanding deep learning through the lens of established physics, and partly by practical demand for extreme model compression in resource-constrained settings. As quantum hardware matures, the connection between tensor networks and quantum computation may become practically significant for a wider range of ML tasks.
