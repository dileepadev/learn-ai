---
title: "Embedding Layers in Deep Learning"
description: "Understanding embedding layers for words, items, and categorical features — initialization, pooling, and contextual embeddings."
date: "2026-06-06"
tags: ["deep-learning", "embeddings", "nlp"]
---

Embedding layers map discrete tokens to dense vector representations. They are foundational in NLP, recommendation systems, and any domain with categorical features.

## Basic Embedding Layer

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx)
    
    def forward(self, x):
        return self.embeddings(x)


# PyTorch built-in
embedding = nn.Embedding(vocab_size=30000, embedding_dim=512, padding_idx=0)

# Input: (batch, seq_len) of token indices
# Output: (batch, seq_len, embed_dim)
```

## Positional Embeddings

Transformers need positional information since attention is permutation-invariant.

### Sinusoidal Positional Embeddings

```python
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]


# Used in original Transformer, BERT, etc.
```

### Learned Positional Embeddings

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x, position_ids=None):
        if position_ids is None:
            batch_size, seq_len = x.shape[:2]
            position_ids = torch.arange(seq_len, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        return self.position_embeddings(position_ids)


# Used in RoBERTa, some GPT implementations
```

### Relative Positional Bias

```python
class RelativePositionalBias(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_pos_emb = nn.Embedding(2 * max_len + 1, d_model)
    
    def forward(self, seq_len):
        # Generate relative positions
        pos = torch.arange(seq_len)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.max_len, self.max_len)
        rel_pos = rel_pos + self.max_len  # Shift to positive indices
        
        return self.rel_pos_emb(rel_pos)


# Used in T5, DeBERTa
```

## Embedding Pruning

For large vocabularies, prune rare embeddings:

```python
class PrunedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, prune_ratio=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.prune_ratio = prune_ratio
        self.mask = None
    
    def prune(self):
        # Prune smallest magnitude embeddings
        magnitudes = self.weight.abs().sum(dim=1)
        threshold = torch.quantile(magnitudes, self.prune_ratio)
        self.mask = magnitudes >= threshold
        self.weight.data[~self.mask] = 0
    
    def forward(self, x):
        if self.training and self.mask is None:
            return F.embedding(x, self.weight)
        return F.embedding(x, self.weight * self.mask.unsqueeze(1))
```

## Handling Rare Tokens

```python
class AdaptiveTokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, min_freq=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Main embedding table
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Subword fallback for rare tokens
        self.char_embed = nn.Embedding(256, embed_dim)
        self.char_conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
    
    def forward(self, x):
        # Try token embedding first
        output = self.token_embed(x)
        
        # For <unk> tokens, use character-based
        unk_mask = (x == 1)  # Assuming 1 is <unk>
        if unk_mask.any():
            char_emb = self.char_conv(self.char_embed(x[unk_mask]).transpose(1, 2))
            char_emb = char_emb.transpose(1, 2).mean(dim=1)
            output[unk_mask] = char_emb
        
        return output
```

## Embedding Regularization

```python
class EmbeddingWithNorm(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_norm=1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_norm = max_norm
    
    def forward(self, x):
        return self.embedding(x)
    
    def normalize_embeddings(self):
        with torch.no_grad():
            self.embedding.weight.div_(
                self.embedding.weight.norm(dim=1, keepdim=True).clamp(min=self.max_norm)
            )
```

## Practical Tips

- Use smaller embeddings for very large vocabularies
- Consider byte-pair encoding (BPE) for subword tokenization
- Scale embeddings by sqrt(d_model) before adding positional encoding
- Normalize embeddings for similarity-based tasks