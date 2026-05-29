---
title: "Fully Connected (Dense) Layers"
description: "Understanding dense layers, initialization, regularization, and efficient implementations."
date: "2026-06-06"
tags: ["deep-learning", "neural-networks", "fully-connected"]
---

Fully connected layers connect every input to every output. While less common in modern architectures than in the past, they remain essential for classification heads and certain architectures.

## Basic Dense Layer

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# PyTorch built-in
linear = nn.Linear(512, 256)
output = linear(input)  # input: (batch, 512), output: (batch, 256)
```

## Weight Initialization for Dense Layers

```python
# Xavier initialization (for tanh/sigmoid)
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# He initialization (for ReLU)
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.zeros_(layer.bias)

# For very deep networks
nn.init.orthogonal_(layer.weight)
```

## Regularization

```python
# L2 regularization (weight decay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Or manually
l2_loss = 0
for param in model.parameters():
    l2_loss += (param ** 2).sum()
loss = main_loss + weight_decay * l2_loss

# Dropout for regularization
dropout = nn.Dropout(p=0.5)
output = dropout(linear(input))
```

## Sparsifying Dense Layers

```python
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.9):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.sparsity = sparsity
    
    def forward(self, x):
        mask = (torch.rand_like(self.weight) > self.sparsity)
        weight = self.weight * mask.float()
        return F.linear(x, weight, self.bias)
```

## Efficient Large Matrix Multiplication

```python
# Sparse matrix multiplication
from scipy.sparse import csr_matrix

def sparse_linear(x, weight, bias=None):
    """Efficient sparse-dense multiplication."""
    sparse_weight = csr_matrix(weight.detach().numpy())
    
    if bias is not None:
        return x @ sparse_weight.T + bias
    return x @ sparse_weight.T


# Low-rank approximation for compression
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.u = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.v = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # weight ≈ u @ v
        weight = self.u @ self.v
        return F.linear(x, weight, self.bias)
```

## Practical Considerations

- Dense layers have many parameters: O(in_features × out_features)
- Use batch normalization after dense layers for training stability
- For very wide layers, consider weight initialization carefully
- In modern architectures, dense layers are often at the end (classification)

```python
# Typical classification head
class ClassifierHead(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)
```

Dense layers are simple but powerful — use them where full connectivity is meaningful.