---
title: Probabilistic Graphical Models
description: Master probabilistic graphical models (PGMs) — the mathematical framework for representing and reasoning under uncertainty using graphs. Learn Bayesian networks, Markov random fields, factor graphs, exact and approximate inference algorithms (variable elimination, belief propagation), and applications to sequence labeling, image segmentation, and joint probability modeling.
---

**Probabilistic graphical models** (PGMs) combine probability theory with graph theory to compactly represent joint distributions over many variables. The graph structure encodes **conditional independence** assumptions: edges indicate direct probabilistic relationships, and missing edges encode independence. This structure enables efficient inference and learning algorithms that would be intractable on raw joint distributions.

PGMs underpin a wide range of systems: spam filters, speech recognition (Hidden Markov Models), medical diagnosis systems, image segmentation (Markov random fields), and natural language processing (Conditional Random Fields). Modern deep learning has largely replaced standalone PGMs, but the ideas — particularly conditional independence, belief propagation, and variational inference — remain foundational and are embedded in many deep probabilistic models.

## Two Families of Graphical Models

The fundamental distinction in PGMs is between **directed** and **undirected** graphs:

| Property | Bayesian Network (DAG) | Markov Random Field (undirected) |
| --- | --- | --- |
| Graph type | Directed Acyclic Graph | Undirected graph |
| Local function | Conditional probability $P(X_i \mid \text{Pa}(X_i))$ | Clique potential $\psi_c(X_c)$ |
| Normalization | Automatic (chain rule factorization) | Requires partition function $Z$ |
| Independence | d-separation | Markov properties |
| Typical use | Causal modeling, generative models | Discriminative models, spatial models |

## Bayesian Networks

A **Bayesian network** (BN) is a directed acyclic graph where each node $X_i$ stores a conditional probability table (CPT) $P(X_i \mid \text{Pa}(X_i))$. The joint distribution factorizes as the product of these local CPTs:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Pa}(X_i))$$

### D-Separation

D-separation is the graphical criterion for reading off conditional independence from a Bayesian network. Given sets of nodes $A$, $B$, $C$: $A \perp B \mid C$ (A is independent of B given C) if and only if $C$ **d-separates** $A$ from $B$.

Three cases block information flow along a path (making nodes independent):

- **Causal chain** $X \rightarrow Z \rightarrow Y$: blocked if $Z$ observed
- **Common cause** $X \leftarrow Z \rightarrow Y$: blocked if $Z$ observed
- **Common effect (v-structure)** $X \rightarrow Z \leftarrow Y$: blocked if $Z$ *unobserved* — but becomes *unblocked* when $Z$ or any descendant is observed (**explaining away**)

The **explaining away** phenomenon is a key feature of Bayesian reasoning: learning that one cause of an event is true makes other causes less likely. If both a fire alarm and a neighbor's test both cause an alert, observing one explanation reduces the probability of the other.

## Variable Elimination for Exact Inference

**Variable elimination** (VE) computes marginals and conditional probabilities by systematically eliminating variables in a chosen order:

```python
import numpy as np
from itertools import product
from typing import Optional

class Factor:
    """
    A factor: a function over a subset of variables.
    
    Represents probability tables: P(X | Pa(X)), joint tables,
    or intermediate products during variable elimination.
    """
    
    def __init__(self, variables: list[str], table: np.ndarray):
        """
        variables: ordered list of variable names this factor covers
        table: probability array, shape = (|dom(var_1)|, |dom(var_2)|, ...)
        """
        self.variables = variables
        self.table = table

    def marginalize(self, variable: str) -> "Factor":
        """Sum out a variable: reduce factor by summing over all values of 'variable'."""
        axis = self.variables.index(variable)
        new_table = self.table.sum(axis=axis)
        new_variables = [v for v in self.variables if v != variable]
        return Factor(new_variables, new_table)

    def multiply(self, other: "Factor") -> "Factor":
        """Pointwise product of two factors (broadcasting over shared variables)."""
        # Find all variables in the product factor
        all_vars = self.variables + [v for v in other.variables if v not in self.variables]
        
        # Compute product by iterating over all variable assignments
        domain_sizes = {v: 2 for v in all_vars}   # binary variables for simplicity
        
        result_shape = tuple(domain_sizes[v] for v in all_vars)
        result = np.zeros(result_shape)
        
        for assignment in product(*[range(domain_sizes[v]) for v in all_vars]):
            val_map = dict(zip(all_vars, assignment))
            
            idx_self = tuple(val_map[v] for v in self.variables)
            idx_other = tuple(val_map[v] for v in other.variables)
            
            result[assignment] = self.table[idx_self] * other.table[idx_other]
        
        return Factor(all_vars, result)

    def reduce(self, variable: str, value: int) -> "Factor":
        """Observe that 'variable = value' — slice and remove from factor."""
        axis = self.variables.index(variable)
        new_table = np.take(self.table, value, axis=axis)
        new_variables = [v for v in self.variables if v != variable]
        return Factor(new_variables, new_table)

    def normalize(self) -> "Factor":
        """Normalize table to sum to 1 (convert to probability distribution)."""
        total = self.table.sum()
        return Factor(self.variables, self.table / total if total > 0 else self.table)


def variable_elimination(
    factors: list[Factor],
    query_variable: str,
    observed: Optional[dict[str, int]] = None,
    elimination_order: Optional[list[str]] = None
) -> Factor:
    """
    Variable elimination: compute P(query | observed) exactly.
    
    Algorithm:
    1. Condition on observed variables (reduce all factors)
    2. For each variable to eliminate (in chosen order):
       a. Multiply all factors involving that variable
       b. Marginalize (sum out) the variable
    3. Multiply remaining factors and normalize
    
    Complexity: exponential in the **treewidth** of the variable graph —
    for graphs with small treewidth (trees, chains), this is polynomial.
    """
    # Step 1: Condition on evidence
    if observed:
        factors = [
            f.reduce(var, val) if var in f.variables else f
            for f in factors
            for var, val in observed.items()
            if var in f.variables
        ]
        # Flatten: each factor was potentially replaced multiple times above
        # In practice, apply observations one at a time:
        conditioned_factors = factors[:]
        for var, val in (observed or {}).items():
            conditioned_factors = [
                f.reduce(var, val) if var in f.variables else f
                for f in conditioned_factors
            ]
        factors = conditioned_factors
    
    # Step 2: Determine variables to eliminate (all except query)
    all_vars = set()
    for f in factors:
        all_vars.update(f.variables)
    eliminate = [v for v in all_vars if v != query_variable]
    
    if elimination_order:
        eliminate = [v for v in elimination_order if v in eliminate]
    
    # Step 3: Eliminate each variable
    for var in eliminate:
        # Gather all factors involving this variable
        relevant = [f for f in factors if var in f.variables]
        others = [f for f in factors if var not in f.variables]
        
        if not relevant:
            continue
        
        # Multiply all relevant factors together
        product_factor = relevant[0]
        for f in relevant[1:]:
            product_factor = product_factor.multiply(f)
        
        # Sum out the variable
        marginalized = product_factor.marginalize(var)
        factors = others + [marginalized]
    
    # Step 4: Multiply remaining factors and normalize
    result = factors[0]
    for f in factors[1:]:
        result = result.multiply(f)
    
    return result.normalize()
```

## Belief Propagation on Trees

For tree-structured graphs (no loops), the **sum-product algorithm** (belief propagation) computes all marginals exactly and efficiently in a single forward-backward pass:

```python
def sum_product_chain(
    factors: list[np.ndarray],     # factors[i] = P(X_i | X_{i-1}), shape (d, d)
    prior: np.ndarray              # P(X_0), shape (d,)
) -> list[np.ndarray]:
    """
    Sum-product belief propagation for a Markov chain.
    
    Equivalent to the forward-backward algorithm in Hidden Markov Models.
    Computes exact marginals P(X_i) for all i simultaneously.
    
    factors[i]: transition/emission table connecting X_i to X_{i+1}
    Returns: list of normalized marginal distributions P(X_i)
    
    Forward pass: alpha[i](x_i) = Σ_{x_{i-1}} alpha[i-1](x_{i-1}) × factor[i](x_i|x_{i-1})
    Backward pass: beta[i](x_i) = Σ_{x_{i+1}} beta[i+1](x_{i+1}) × factor[i+1](x_{i+1}|x_i)
    Marginal: P(X_i) ∝ alpha[i](x_i) × beta[i](x_i)
    """
    n = len(factors) + 1
    d = len(prior)
    
    # Forward pass
    alpha = [None] * n
    alpha[0] = prior.copy()
    
    for i in range(len(factors)):
        # alpha[i+1](x) = Σ_y alpha[i](y) × factor[i](y→x)
        alpha[i + 1] = factors[i].T @ alpha[i]
        alpha[i + 1] /= alpha[i + 1].sum()   # numerical stability
    
    # Backward pass
    beta = [None] * n
    beta[-1] = np.ones(d) / d
    
    for i in range(len(factors) - 1, -1, -1):
        # beta[i](y) = Σ_x factor[i](y→x) × beta[i+1](x)
        beta[i] = factors[i] @ beta[i + 1]
        beta[i] /= beta[i].sum()
    
    # Marginals: P(X_i) ∝ alpha[i] × beta[i]
    marginals = []
    for i in range(n):
        marginal = alpha[i] * beta[i]
        marginals.append(marginal / marginal.sum())
    
    return marginals
```

## Conditional Random Fields for Sequence Labeling

**Conditional Random Fields** (CRFs, Lafferty et al., 2001) are undirected graphical models used for sequence labeling tasks (NER, POS tagging, chunking). Unlike a Naive Bayes or HMM that models $P(X, Y)$, a CRF models $P(Y \mid X)$ directly — discriminatively conditioning on the entire input sequence:

$$P(Y \mid X) = \frac{1}{Z(X)} \exp\left(\sum_t \sum_k \lambda_k f_k(y_{t-1}, y_t, X, t)\right)$$

where $f_k$ are feature functions and $\lambda_k$ are learned weights. The partition function $Z(X) = \sum_Y \exp(\cdots)$ is computed exactly using the forward algorithm.

Linear-chain CRF training and inference is efficiently implemented in the `torchcrf` library, and CRF layers are often added on top of BiLSTM or BERT encoders:

```python
import torch
import torch.nn as nn
from torchcrf import CRF   # pip install pytorch-crf

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF: the workhorse of sequence labeling before BERT transformers.
    Still used as the final layer in many BERT-based NER models because:
    - Enforces valid label transitions (e.g., I-PER cannot follow B-LOC)
    - Performs Viterbi decoding to find globally optimal label sequence
    - Often improves F1 by 1-2% over argmax per-token decoding
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, num_labels: int,
                 num_lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None):
        """
        Training: compute negative log-likelihood (CRF loss)
        Inference: Viterbi decode to get best label sequence
        """
        embeddings = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(self.dropout(lstm_out))  # (B, T, num_labels)
        
        mask = attention_mask.bool()
        
        if labels is not None:
            # CRF loss: negative of log-likelihood
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
            return -log_likelihood
        else:
            # Viterbi decoding
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
```

## Applications

**Medical diagnosis**: Bayesian networks explicitly encode expert medical knowledge (symptoms → disease → test results) and can reason under partial observations ("patient has fever and cough but no blood test results yet").

**Image segmentation with MRF**: The classical MRF energy minimization approach models pixel labels as nodes, with unary terms from a classifier and pairwise terms that encourage neighboring pixels to share labels. Minimized via graph cuts (max-flow/min-cut) or belief propagation.

**Hidden Markov Models**: A special case of Bayesian network with a latent chain (states) generating observations. Forward-backward (belief propagation) gives marginals; Viterbi gives the most likely state sequence. Foundational to classical speech recognition.

PGMs provide a principled vocabulary — conditional independence, factorization, message passing — that remains directly relevant in modern deep learning through variational autoencoders, diffusion models, normalizing flows, and Bayesian deep learning.
