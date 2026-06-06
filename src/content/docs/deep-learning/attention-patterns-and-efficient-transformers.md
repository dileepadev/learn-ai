---
title: "Efficient Attention Patterns: Sparse, Linear, and Hierarchical"
description: "Understanding efficient attention mechanisms that reduce quadratic complexity in transformers — sparse attention, linear attention, and hierarchical approaches."
date: "2026-06-06"
tags: ["deep-learning", "transformers", "attention", "efficiency"]
---

Standard self-attention has $O(N^2)$ complexity with sequence length $N$, making it a bottleneck for long sequences. This guide covers the main approaches to efficient attention that reduce this cost.

## The Attention Complexity Problem

For a sequence of length $N$ with embedding dimension $d$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

The $QK^T$ matrix multiplication costs $O(N^2)$ time and memory. For sequences of 10K+ tokens, this becomes prohibitive.

```python
import torch
import torch.nn.functional as F

def attention_flops(seq_len: int, d_model: int, heads: int = 8) -> int:
    """Calculate FLOPs for multi-head attention."""
    head_dim = d_model // heads
    
    # Q, K, V projections: 3 * N * d_model * d_model
    qkv_flops = 3 * seq_len * d_model * d_model
    
    # Scaled dot-product: N * N * head_dim
    attention_flops = seq_len * seq_len * head_dim
    
    # Output projection: N * d_model * d_model
    output_flops = seq_len * d_model * d_model
    
    return qkv_flops + attention_flops + output_flops


# For seq_len=4096, d_model=4096:
# Attention alone: 4096 * 4096 * 512 ≈ 8.6 billion FLOPs
# This dominates total compute for long sequences
```

## Sparse Attention

Sparse attention patterns restrict attention to a subset of positions:

```python
class SparseAttentionPattern:
    """Different sparse attention patterns."""
    
    @staticmethod
    def sliding_window(sequence_length: int, window_size: int = 512):
        """Each token attends to previous window_size tokens."""
        # For each position i, attend to [i-window_size, i)
        mask = torch.zeros(sequence_length, sequence_length)
        for i in range(sequence_length):
            start = max(0, i - window_size)
            mask[i, start:i+1] = 1
        return mask
    
    @staticmethod
    def dilated_attention(sequence_length: int, window_size: int = 256, dilation: int = 2):
        """Dilated sliding windows for broader context."""
        mask = torch.zeros(sequence_length, sequence_length)
        for i in range(sequence_length):
            for d in range(dilation):
                pos = i - (d + 1) * window_size
                if pos >= 0:
                    mask[i, pos] = 1
        return mask
    
    @staticmethod
    def global_tokens(sequence_length: int, num_global: int = 16):
        """Special global tokens attend to all positions."""
        mask = torch.zeros(sequence_length, sequence_length)
        mask[:num_global, :] = 1  # Global tokens attend to all
        mask[:, :num_global] = 1  # All tokens attend to globals
        return mask


class LongformerAttention(nn.Module):
    """Longformer attention: sliding window + global tokens."""
    def __init__(self, d_model: int, num_heads: int, window_size: int = 512, 
                 num_global_tokens: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.num_global = num_global_tokens
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, attention_mask=None):
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        
        # Compute attention scores
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) / (self.head_dim ** 0.5)
        
        # Create sliding window mask
        mask = torch.ones(N, N, device=x.device)
        
        # Sliding window attention
        mask = torch.zeros(N, N, device=x.device)
        for i in range(N):
            start = max(0, i - self.window_size)
            mask[i, start:i+1] = 1
        
        # Global tokens attend to all
        mask[:self.num_global, :] = 1
        mask[:, :self.num_global] = 1
        
        attn = attn.masked_fill(mask == 0, -1e9)
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        
        return self.proj(out.view(B, N, D))


# Complexity: O(N * window_size) instead of O(N^2)
# For window_size=512, N=16384: 8x reduction
```

## Linear Attention

Linear attention replaces softmax with a kernel approximation:

$$\text{Attention} = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) (\phi(K)^T \mathbf{1})}$$

If $\phi(x)$ is a low-rank approximation, this can be computed in $O(N)$ time.

```python
class LinearAttention(nn.Module):
    """
    Linear attention with feature map approximation.
    
    Uses the identity: softmax(x^T y) = lim_{lambda->inf} exp(lambda * x^T y)
    
    Approximated as: φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))
    """
    def __init__(self, d_model: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        # Feature map parameters
        self.gate = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        
        # Feature map (ReLU + L2 normalization trick)
        def phi(x):
            return F.relu(x)
        
        q = self.q(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim)
        
        # Apply feature map
        q = phi(q)
        k = phi(k)
        
        # Compute kernel products (linear in N)
        k_expanded = k.unsqueeze(3)  # B, N, H, D, 1
        v_expanded = v.unsqueeze(2)  # B, N, H, 1, D
        
        # kv: B, H, D, D
        kv = torch.einsum('bnhd,bnhm->bhdm', k, v)
        
        # qkv: B, N, H, D
        qkv = torch.einsum('bnhd,bhdm->bnhm', q, kv)
        
        # Normalizer: q @ (k @ 1) = q * sum(k)
        # B, N, H, 1
        k_sum = torch.einsum('bnhd->bhn', k)
        normalizer = torch.einsum('bnhm,bhn->bnhm', q, k_sum.unsqueeze(-1)) + self.eps
        
        out = qkv / normalizer
        out = out.view(B, N, D)
        
        return self.proj(out)


class PerformerAttention(nn.Module):
    """Fast Attention via Positive Orthogonal Random Features (FAVOR)."""
    def __init__(self, d_model: int, num_heads: int, num_features: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_features = num_features
        
        # Random projection matrix
        self.register_buffer('m', torch.randn(num_features, self.head_dim))
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        
        q = self.q(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim)
        
        # Positive random features approximation
        def prf(x):
            # Project to random features space
            # φ(x) = exp(-||x||²/2) * (m @ x.T).T
            x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2
            return torch.exp(x_norm_sq) * torch.einsum('nm,bnd->bmn', self.m, x)
        
        q_prime = prf(q)
        k_prime = prf(k)
        
        # Compute attention using random features (linear in N)
        # Deno: k_prime @ v
        kv = torch.einsum('bnd,bndm->bm', k_prime, v.unsqueeze(1))
        
        # Nume: q_prime @ (k' @ v)
        qkv = torch.einsum('bnd,bm->bmn', q_prime, kv)
        
        # Normalizer: q' @ k'
        normalizer = torch.einsum('bnd,bnd->bn', q_prime, k_prime).unsqueeze(-1)
        
        out = qkv / normalizer
        
        return self.proj(out.view(B, N, D))
```

## Hierarchical Attention

Hierarchical approaches process sequences in chunks:

```python
class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention: local attention within chunks + global attention between chunks.
    
    Reduces complexity from O(N^2) to O(N * chunk_size + num_chunks^2).
    """
    def __init__(self, d_model: int, num_heads: int, chunk_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // num_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Pad to chunk size
        pad_len = (self.chunk_size - N % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Reshape into chunks: (B, num_chunks, chunk_size, D)
        num_chunks = N // self.chunk_size
        x = x.view(B, num_chunks, self.chunk_size, D)
        
        q = self.q(x).view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        k = self.k(x).view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        v = self.v(x).view(B, num_chunks, self.chunk_size, self.num_heads, self.head_dim)
        
        # 1. Local attention within each chunk: O(num_chunks * chunk_size^2)
        local_attn = torch.einsum('bnhkd,bnhmd->bnhkm', q, k) / self.head_dim ** 0.5
        local_attn = F.softmax(local_attn, dim=-1)
        local_out = torch.einsum('bnhkm,bnhmd->bnhkd', local_attn, v)
        
        # 2. Global attention between chunks: O(num_chunks^2 * chunk_size)
        # Pool each chunk to a single representation
        chunk_k = k.mean(dim=2)  # (B, num_chunks, num_heads, head_dim)
        chunk_v = v.mean(dim=2)
        
        global_attn = torch.einsum('bnhkd,bmhmd->bnhkm', q.mean(dim=2), chunk_k) / self.head_dim ** 0.5
        global_attn = F.softmax(global_attn, dim=2)
        global_out = torch.einsum('bnhm,bmhd->bnhd', global_attn, chunk_v)
        
        # 3. Combine local and global
        # Global contribution to each position in chunk
        global_out = global_out.unsqueeze(2).expand(-1, -1, self.chunk_size, -1, -1)
        
        out = local_out + global_out
        out = out.reshape(B, num_chunks * self.chunk_size, D)[:, :N]
        
        return self.proj(out)
```

## Flash Attention

Flash Attention computes attention with optimal memory access patterns using GPU tiles:

```python
class FlashAttentionExample:
    """
    Flash Attention algorithm (simplified).
    
    Key insights:
    1. Tiling: Process attention in blocks that fit in SRAM
    2. Recomputation: Store softmax stats, recompute activations during backward
    """
    
    def flash_attention(self, q, k, v, block_size=64):
        """
        Compute attention with tiling for better memory efficiency.
        
        Instead of storing N x N attention matrix, compute in blocks.
        """
        B, H, N, D = q.shape
        output = torch.zeros(B, H, N, D, device=q.device)
        
        # Tile sizes
        Br = block_size  # Rows per tile
        Bc = block_size  # Columns per tile
        
        # Store softmax statistics for backward
        l = torch.zeros(B, H, N, device=q.device)
        m = torch.full((B, H, N), -float('inf'), device=q.device)
        
        for i in range(0, N, Bc):
            kj = k[:, :, i:i+Bc]  # Block of keys
            vj = v[:, :, i:i+Bc]  # Block of values
            
            for j in range(0, N, Br):
                qi = q[:, :, j:j+Br]  # Block of queries
                
                # Compute attention for this block
                attn = torch.einsum('bhd,bmhd->bhm', qi, kj) / D ** 0.5
                
                # Block-wise softmax
                mask = torch.ones_like(attn)
                
                # Update max values
                m_ij = torch.max(attn, dim=-1, keepdim=True).values
                m_new = torch.maximum(m[:, :, j:j+Br], m_ij)
                
                # Update sum of exp
                attn = attn - m_ij  # Subtract max for numerical stability
                p = torch.exp(attn) * mask
                
                l_ij = p.sum(dim=-1, keepdim=True)
                l[:, :, j:j+Br] = torch.exp(m[:, :, j:j+Br] - m_new) * l[:, :, j:j+Br] + l_ij
                m[:, :, j:j+Br] = m_new
                
                # Compute output
                out_ij = torch.einsum('bhm,bmhd->bhd', p, vj)
                output[:, :, j:j+Br] = out_ij
        
        # Normalize
        output = output / l.unsqueeze(-1)
        
        return output


# PyTorch's flash attention (requires compatible hardware)
from torch.nn.functional import scaled_dot_product_attention

def use_flash_attention(q, k, v, attn_mask=None):
    """Flash attention when available."""
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=0.0
        )
    else:
        # Fallback to standard attention
        attn = torch.einsum('bhtd,bhsd->bhts', q, k) / (q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        return torch.einsum('bhts,bhsd->bhtd', attn, v)


# Flash attention reduces memory from O(N^2) to O(N)
# Enables attention on sequences of 100K+ tokens
```

## Comparison of Efficient Attention Methods

| Method | Complexity | Quality | Use Case |
| --- | --- | --- | --- |
| Full Attention | $O(N^2)$ | Best | Short sequences (< 4K) |
| Sparse (Sliding) | $O(N \cdot w)$ | Good | Documents, code |
| Sparse (Block) | $O(N \cdot \sqrt{N})$ | Good | Medium sequences |
| Linear (FAVOR) | $O(N \cdot d)$ | Moderate | Very long sequences |
| Hierarchical | $O(N \cdot c + (N/c)^2)$ | Good | Multi-scale tasks |
| Flash | $O(N \cdot d)$ | Best | GPU-optimized |

```python
class EfficientTransformerLayer(nn.Module):
    """Transformer layer with choice of efficient attention."""
    def __init__(self, d_model: int, num_heads: int, 
                 attention_type: str = 'flash', **kwargs):
        super().__init__()
        
        self.attention_type = attention_type
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x, seq_len=None):
        # Compute Q, K, V
        q = self.q(x).view(*x.shape[:2], -1)
        k = self.k(x).view(*x.shape[:2], -1)
        v = self.v(x).view(*x.shape[:2], -1)
        
        # Apply attention based on type
        if self.attention_type == 'flash':
            out = use_flash_attention(q, k, v)
        elif self.attention_type == 'linear':
            out = LinearAttention(d_model=x.size(-1), num_heads=8)(x)
        elif self.attention_type == 'sparse':
            out = LongformerAttention(d_model=x.size(-1), num_heads=8)(x)
        else:
            # Default: full attention
            attn = torch.einsum('bnd,bmd->bnm', q, k) / (q.size(-1) ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('bnm,bmd->bnd', attn, v)
        
        # Residual and FFN
        x = x + out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        
        return x
```

## Practical Recommendations

- **Short sequences (< 4K tokens)**: Use standard attention
- **Medium sequences (4K-32K)**: Flash attention or sparse window attention
- **Long sequences (32K+)**: Hierarchical attention or linear attention
- **Hardware-aware**: Flash attention is best when available
- **Flash Attention 2/3**: Use for maximum efficiency on modern GPUs

Efficient attention enables transformers to process much longer sequences, opening up new applications in document understanding, code analysis, and long-context language models.