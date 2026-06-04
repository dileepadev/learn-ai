---
title: "The Transformer Architecture: Architecture and Applications"
description: "Understand the transformer architecture that powers modern AI — from the original attention is all you need paper to BERT, GPT, and beyond."
---

The transformer architecture has become the foundation of modern deep learning. Originally designed for machine translation, transformers now power language models, vision systems, and multimodal AI. This guide covers the architecture, variations, and applications.

## The Transformer Architecture

The transformer uses self-attention to process sequences without recurrence or convolution:

```python
import torch
import torch.nn as nn
import math

class TransformerConfig:
    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=0.1,
        max_seq_len=2048,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
```

## Input Embedding

```python
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, input_ids):
        # Token embeddings: (batch, seq_len) -> (batch, seq_len, d_model)
        token_embeds = self.token_embedding(input_ids)
        
        # Scale by sqrt(d_model)
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Add position embeddings
        position_embeds = self.position_embedding[:, :input_ids.size(1), :]
        
        # Combine and dropout
        return self.dropout(token_embeds + position_embeds)
```

## Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

## Scaled Dot-Product Attention

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Q, K, V: (batch, n_heads, seq_len, d_k)
        mask: (batch, 1, seq_len, seq_len) or (batch, seq_len, seq_len)
        """
        # Compute attention scores
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
```

## Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        query = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention = self.attention(query, key, value, mask)
        
        # Concatenate heads and project
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.w_o(x)
        
        return x
```

## Position-Wise Feed-Forward Network

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
```

## Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture (more stable than post-norm)
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention with causal mask
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with encoder
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(self, config, encoder_only=False):
        super().__init__()
        self.encoder_only = encoder_only
        
        self.embedding = InputEmbedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Decoder (if not encoder-only)
        if not encoder_only:
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            self.decoder_embedding = InputEmbedding(config.vocab_size, config.d_model)
            self.decoder_pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        self.classifier = nn.Linear(config.d_model, config.vocab_size)
    
    def generate_mask(self, size):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, src, tgt=None):
        # Encode source
        src_emb = self.pos_encoder(self.embedding(src))
        
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        
        if self.encoder_only:
            # For encoder-only models (BERT, etc.)
            return self.classifier(src_emb)
        
        # Decode
        tgt_emb = self.decoder_pos_encoder(self.decoder_embedding(tgt))
        tgt_mask = self.generate_mask(tgt.size(1))
        
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, tgt_mask=tgt_mask)
        
        return self.classifier(tgt_emb)
```

## Transformer Variants

### BERT: Bidirectional Encoder Representations

```python
class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config, encoder_only=True)
        self.pooler = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()
        self.next_sentence_predictor = nn.Linear(config.d_model, 2)
        self.masked_lm_predictor = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        transformer_output = self.transformer(input_ids)
        
        # Use [CLS] token for classification
        pooled = self.activation(self.pooler(transformer_output[:, 0]))
        
        # Next sentence prediction
        next_sentence_logits = self.next_sentence_predictor(pooled)
        
        # MLM (predict masked tokens)
        mlm_logits = self.masked_lm_predictor(transformer_output)
        
        return {
            "mlm_logits": mlm_logits,
            "nsp_logits": next_sentence_logits
        }
```

### GPT: Generative Pre-Training

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Transformer(config, encoder_only=True)
        # Remove causal mask from decoder (built into generation)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        transformer_output = self.transformer(input_ids)
        return self.lm_head(transformer_output)
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Get predictions
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Greedy decoding (can use top-k/top-p sampling)
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return input_ids
```

## Efficient Attention Variants

### Grouped Query Attention

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = d_model // n_heads
        
        # Fewer KV heads than Q heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_groups * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)
```

### Sliding Window Attention

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=512, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.attention = ScaledDotProductAttention(dropout)
        # Implement causal masking within window
    
    def forward(self, query, key, value, mask=None):
        # Only attend to tokens within window_size
        pass
```

The transformer architecture has transformed deep learning. Understanding its components — attention, feed-forward networks, layer normalization — provides the foundation for working with any transformer-based model, from BERT to GPT to Vision Transformers.