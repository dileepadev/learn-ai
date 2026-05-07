---
title: Diffusion Language Models
description: A comprehensive guide to diffusion models for text generation, covering masked diffusion, absorbing state diffusion, score-based text diffusion, and how these models overcome autoregressive limitations for controllable and parallel text generation.
---

# Diffusion Language Models

Autoregressive language models generate text left-to-right, one token at a time. **Diffusion language models** take a fundamentally different approach: they learn to reverse a noising process that progressively corrupts text, enabling **parallel decoding**, **bidirectional context**, and fine-grained **controllable generation** that is structurally difficult for autoregressive models. As of 2025–2026, diffusion LMs are approaching autoregressive quality while unlocking qualitatively new capabilities.

## The Core Idea

In continuous diffusion (DDPM, score matching), a forward process adds Gaussian noise to data and the model learns to denoise. For discrete text, noise is applied by **masking** or **replacing** tokens:

$$q(x_t | x_0) = \text{Categorical}(\alpha_t x_0 + (1-\alpha_t) \mathbf{m})$$

where $\mathbf{m}$ is the mask token and $\alpha_t \in [0,1]$ is a noise schedule — at $t=T$, all tokens are masked; at $t=0$, the original text is recovered.

## Masked Diffusion Language Models (MDLM)

**MDLM** (Sahoo et al., 2024) frames text diffusion as a **masked absorbing state** process — the cleanest formulation for discrete sequences:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class MaskedDiffusionLM(nn.Module):
    """Masked diffusion language model with BERT backbone."""

    MASK_TOKEN_ID = 103   # [MASK] in BERT tokenizer

    def __init__(self, vocab_size: int, hidden_size: int = 768, n_layers: int = 12):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=12,
            intermediate_size=hidden_size * 4,
        )
        self.transformer = BertModel(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply masking noise at time t. t in [0, 1]."""
        # Cosine schedule: alpha_t = cos(pi/2 * t)^2
        alpha_t = torch.cos(torch.pi / 2 * t) ** 2   # (B,)
        mask_prob = 1.0 - alpha_t                      # higher t → more masking
        mask = torch.bernoulli(
            mask_prob.unsqueeze(1).expand_as(x0).float()
        ).bool()
        xt = x0.clone()
        xt[mask] = self.MASK_TOKEN_ID
        return xt

    def denoise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict x0 from xt at noise level t."""
        # Concatenate time embedding via special token or positional encoding
        h = self.transformer(xt).last_hidden_state       # (B, L, D)
        return self.lm_head(h)                           # (B, L, vocab_size)

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B, L = x0.shape
        t = torch.rand(B, device=x0.device)
        xt = self.forward_process(x0, t)
        logits = self.denoise(xt, t)                     # (B, L, V)
        # Only compute loss on masked positions
        mask = xt == self.MASK_TOKEN_ID
        return F.cross_entropy(
            logits[mask],
            x0[mask],
        )
```

### Sampling from MDLM

```python
@torch.no_grad()
def mdlm_sample(model, seq_len: int, n_steps: int = 100, device="cuda") -> torch.Tensor:
    vocab_size = model.lm_head.out_features
    # Start fully masked
    x = torch.full((1, seq_len), model.MASK_TOKEN_ID, dtype=torch.long, device=device)

    timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for i in range(n_steps):
        t = timesteps[i].expand(1)
        t_next = timesteps[i + 1].expand(1)

        logits = model.denoise(x, t)                    # predict x0
        probs = logits.softmax(dim=-1)                  # (1, L, V)

        # Remask: fraction of tokens to keep unmasked at t_next
        alpha_t = (torch.cos(torch.pi / 2 * t) ** 2).item()
        alpha_next = (torch.cos(torch.pi / 2 * t_next) ** 2).item()

        # Sample x0 for currently masked positions
        flat = probs.view(-1, vocab_size)
        x0_pred = flat.multinomial(1).view(1, seq_len)

        # Determine which positions to reveal at this step
        still_masked = (x == model.MASK_TOKEN_ID)
        remask_prob = (1 - alpha_next) / (1 - alpha_t + 1e-8)
        reveal = still_masked & (torch.rand_like(x.float()) > remask_prob)

        x[reveal] = x0_pred[reveal]

    return x
```

## SEDD: Score Entropy Discrete Diffusion

**SEDD** (Lou et al., 2023) generalizes score matching to discrete spaces using a **score entropy** objective that works with uniform, absorbing, or learned noise kernels:

$$\mathcal{L}_{\text{SEDD}} = \mathbb{E}_{t, x_t}\left[\sum_{y \neq x_t} R_t(x_t, y) \left(s_\theta(x_t, t)_y - \frac{q(x_t | y)}{q(x_t | x_0)}\right)^2\right]$$

where $s_\theta(x_t, t)$ is the **discrete score** — a vector over vocabulary indicating which tokens would have been more likely at the previous step.

```python
class SEDDModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.score_head = nn.Linear(d_model, vocab_size)   # discrete score

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(xt)
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1)
        h = self.transformer(h + t_emb)
        return self.score_head(h)   # (B, L, V) discrete score
```

## Advantages Over Autoregressive Models

| Property | Autoregressive | Diffusion LM |
|---|---|---|
| Generation direction | Left-to-right only | Bidirectional, global |
| Decoding parallelism | Sequential (O(L) steps) | Parallel (O(T) steps, T ≪ L) |
| Constrained generation | Hard (requires special decoding) | Natural (fix some tokens, diffuse rest) |
| Infilling / editing | Awkward | Native |
| Token-level control | Limited | Fine-grained via masking |
| Perplexity (current SOTA) | Better | Closing gap |

## Controlled Generation

Diffusion LMs naturally support **constrained generation**: fix certain tokens (beginning, end, keywords, format) and diffuse the rest.

```python
def constrained_sample(model, prompt_ids, suffix_ids, span_len, n_steps=100):
    """Fill a span between a fixed prefix and suffix."""
    seq = torch.cat([
        prompt_ids,
        torch.full((span_len,), model.MASK_TOKEN_ID),
        suffix_ids,
    ]).unsqueeze(0)

    # Mask indicating which positions are free to change
    free_mask = torch.zeros_like(seq, dtype=torch.bool)
    start = prompt_ids.size(0)
    free_mask[0, start:start + span_len] = True

    timesteps = torch.linspace(1.0, 0.0, n_steps + 1)
    for i in range(n_steps):
        t = timesteps[i].unsqueeze(0)
        t_next = timesteps[i + 1].unsqueeze(0)
        logits = model.denoise(seq, t)
        x0_pred = logits.softmax(-1).view(-1, logits.size(-1)).multinomial(1).view_as(seq)

        alpha_next = (torch.cos(torch.pi / 2 * t_next) ** 2).item()
        alpha_t = (torch.cos(torch.pi / 2 * t) ** 2).item()
        remask_prob = (1 - alpha_next) / (1 - alpha_t + 1e-8)
        reveal = free_mask & (seq == model.MASK_TOKEN_ID) & (torch.rand_like(seq.float()) > remask_prob)
        seq[reveal] = x0_pred[reveal]

    return seq
```

## Notable Models and Results

| Model | Approach | Params | Notes |
|---|---|---|---|
| MDLM (2024) | Absorbing-state masked diffusion | 110M–3B | Best discrete diffusion perplexity |
| SEDD (2023) | Score entropy, uniform noise | 110M | Principled score-based framework |
| LLaDA (2025) | Large Language Diffusion w/ Masking | 8B | First 8B diffusion LM, competitive with Llama 3 8B |
| MD4 (2024) | Masked diffusion + Flow matching | 400M | Improved sampling efficiency |
| Plaid (2023) | Continuous embedding diffusion | 400M | Operates in embedding space |

## LLaDA: Scaling Diffusion LMs

LLaDA (2025) demonstrated that a masked diffusion model at 8B parameters achieves **comparable benchmark performance** to Llama 3 8B on standard NLP tasks, while natively supporting:

- Bidirectional text infilling without fine-tuning
- Length-controllable generation
- Better calibration on constrained formats (code, JSON)

## Current Limitations

- **Perplexity gap**: top autoregressive models still outperform on unconditional generation quality
- **Sampling speed**: $T = 100$–1000 steps needed for quality; recent work reduces to $T = 10$–50 with distillation
- **Tokenization sensitivity**: absorbing-state diffusion is sensitive to tokenizer granularity
- **Evaluation metrics**: standard perplexity is well-defined only for autoregressive models; diffusion models need alternative metrics

## Summary

Diffusion language models represent a fundamentally different paradigm for text generation — replacing sequential token prediction with iterative denoising over the full sequence. MDLM, SEDD, and LLaDA demonstrate that this approach can scale to competitive performance while enabling controllable generation capabilities that autoregressive models cannot match natively. As sampling efficiency improves through distillation and better noise schedules, diffusion LMs are emerging as a serious alternative architecture for next-generation language models.
