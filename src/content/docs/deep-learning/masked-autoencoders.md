---
title: Masked Autoencoders (MAE)
description: Learn how Masked Autoencoders achieve state-of-the-art self-supervised visual representation learning by masking a high fraction of image patches and training a Vision Transformer encoder-decoder to reconstruct the missing pixels — producing transferable representations with minimal labeled data.
---

**Masked Autoencoders** (He et al., Meta AI, 2021) demonstrated that an embarrassingly simple idea — hide most of an image and predict the missing parts — produces visual representations that rival supervised ImageNet pretraining. The approach scales cleanly with model size, uses far less compute than contrastive methods, and transfers well to diverse downstream tasks.

MAE was directly inspired by the success of masked language modeling in NLP (BERT's masked token prediction), but adapting it to images required rethinking the architecture and the masking ratio.

## The Core Insight: High Masking Ratio

In NLP, BERT masks 15% of tokens — enough to create a difficult prediction task because language is highly semantic and context-dependent. Images are fundamentally different: adjacent pixels are heavily correlated, so predicting a masked patch from its immediate neighbors is trivially easy.

MAE's key finding is that **masking 75% of image patches** creates a sufficiently challenging reconstruction task that forces the encoder to develop holistic, semantic representations rather than relying on local texture interpolation.

This high masking ratio has a crucial practical side-effect: **it dramatically reduces the encoder's computational cost** during pretraining. The encoder processes only the 25% of visible patches; the decoder handles reconstruction. An asymmetric design where the encoder is large and expensive (ViT-Large) and the decoder is small and cheap (a few Transformer blocks) makes MAE highly efficient.

## Architecture

```
Input image (224×224)
    ↓
Patch partition (16×16 patches → 196 tokens)
    ↓
Random mask (drop 75% of tokens → 49 visible tokens)
    ↓
[ENCODER — large ViT]
  Position embed → Transformer blocks → encoded visible tokens (49)
    ↓
[DECODER — lightweight Transformer]
  Insert learnable mask tokens (147) + encoded visible tokens (49)
  + full position embeddings
  → Reconstruct pixel values for all 196 patches
    ↓
MSE loss on masked patches only (in pixel space)
```

The **encoder never sees mask tokens** — it only processes the visible subset. This is what makes MAE efficient: for 75% masking, the encoder processes 3–4× fewer tokens than a standard ViT on the full image.

The **decoder** is only used during pretraining. At inference time it is discarded; the encoder alone produces the representation.

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from functools import partial

# ---- Patch Embedding ----
class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        # Convolution with kernel=stride=patch_size acts as patch splitting + linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, embed_dim)
        x = self.proj(x)                   # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)   # (B, N, embed_dim)
        return x


# ---- Random Masking ----
def random_masking(x, mask_ratio=0.75):
    """
    Randomly mask patches by shuffling and keeping the first (1 - mask_ratio) fraction.

    Returns:
        x_visible: (B, N_visible, D) — unmasked tokens
        mask: (B, N) — binary mask (1 = masked)
        ids_restore: (B, N) — indices to unshuffle tokens in the decoder
    """
    B, N, D = x.shape
    n_keep = int(N * (1 - mask_ratio))

    # Random noise for shuffling
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = noise.argsort(dim=1)          # ascending: small noise = kept
    ids_restore = ids_shuffle.argsort(dim=1)    # inverse permutation

    ids_keep = ids_shuffle[:, :n_keep]
    x_visible = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = mask.gather(1, ids_restore)   # 1 = masked, 0 = visible

    return x_visible, mask, ids_restore


# ---- Transformer Block (simplified) ----
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ---- MAE Model ----
class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16,
        encoder_dim=768, encoder_depth=12, encoder_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        mask_ratio=0.75
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=encoder_dim)
        num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio

        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.enc_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_dim), requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_dim, encoder_heads) for _ in range(encoder_depth)
        ])
        self.enc_norm = nn.LayerNorm(encoder_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.dec_pred = nn.Linear(decoder_dim, patch_size**2 * 3)  # pixel values

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Sinusoidal positional embeddings would be initialized here

    def encode(self, x):
        tokens = self.patch_embed(x)
        tokens = tokens + self.enc_pos_embed[:, 1:, :]  # add positional embedding

        tokens, mask, ids_restore = random_masking(tokens, self.mask_ratio)

        # Prepend cls token
        cls = self.cls_token + self.enc_pos_embed[:, :1, :]
        cls = cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.enc_norm(tokens)

        return tokens, mask, ids_restore

    def decode(self, encoded, ids_restore):
        tokens = self.decoder_embed(encoded)

        # Append mask tokens and unshuffle to original order
        B, N_visible_plus_cls, D = tokens.shape
        N = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N + 1 - N_visible_plus_cls, -1)

        # Remove cls, unshuffle, re-add cls
        x_ = torch.cat([tokens[:, 1:, :], mask_tokens], dim=1)
        x_ = x_.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        tokens = torch.cat([tokens[:, :1, :], x_], dim=1)

        tokens = tokens + self.dec_pos_embed
        for block in self.decoder_blocks:
            tokens = block(tokens)
        tokens = self.dec_norm(tokens)
        tokens = self.dec_pred(tokens[:, 1:, :])  # remove cls, predict patches
        return tokens

    def forward(self, imgs):
        encoded, mask, ids_restore = self.encode(imgs)
        pred = self.decode(encoded, ids_restore)
        return pred, mask


# ---- Loss (MSE on masked patches only) ----
def mae_loss(pred, imgs, mask, patch_size=16, norm_pix_loss=True):
    """
    Compute MSE loss on masked patches.
    If norm_pix_loss=True, normalize each patch to zero mean, unit variance
    before computing loss (empirically improves representation quality).
    """
    # Patchify target: (B, N, patch_size^2 * 3)
    B, C, H, W = imgs.shape
    h = w = H // patch_size
    target = imgs.reshape(B, C, h, patch_size, w, patch_size)
    target = target.permute(0, 2, 4, 1, 3, 5)     # (B, h, w, C, p, p)
    target = target.reshape(B, h * w, -1)           # (B, N, p*p*3)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)               # per-patch MSE: (B, N)
    loss = (loss * mask).sum() / mask.sum()  # mean over masked patches
    return loss
```

## Pretraining and Fine-tuning

During pretraining, MAE trains entirely in a self-supervised manner with no labels:

```python
# Pretraining loop (simplified)
model = MaskedAutoencoder(encoder_dim=1024, encoder_depth=24, encoder_heads=16)  # ViT-Large
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

for imgs, _ in dataloader:  # labels ignored during pretraining
    pred, mask = model(imgs)
    loss = mae_loss(pred, imgs, mask, norm_pix_loss=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

After pretraining, **fine-tuning** discards the decoder and attaches a linear head or fine-tunes the full encoder:

```python
import torchvision.models as tvm

class MAEFineTuned(nn.Module):
    def __init__(self, pretrained_encoder, num_classes=1000):
        super().__init__()
        self.encoder = pretrained_encoder  # from MAE pretraining
        self.head = nn.Linear(1024, num_classes)  # for ViT-Large

    def forward(self, x):
        encoded, _, _ = self.encoder.encode(x)
        cls_token = encoded[:, 0]  # use cls token as global representation
        return self.head(cls_token)
```

## Results and Comparison

MAE-pretrained ViT-Large achieves **87.8% top-1 accuracy on ImageNet** with supervised fine-tuning — outperforming supervised ViT-Large trained from scratch (86.6%) and matching or exceeding contrastive methods like MoCo v3 and DINO.

| Method | Backbone | ImageNet Top-1 | Pretraining Cost |
|---|---|---|---|
| Supervised | ViT-Large | 86.6% | High (labeled) |
| MoCo v3 | ViT-Large | 84.1% | High |
| DINO v2 | ViT-Large | 86.3% | Very High |
| **MAE** | **ViT-Large** | **87.8%** | **Moderate** |
| **MAE** | **ViT-Huge** | **87.8%** | **High** |

## Extensions and Variants

**VideoMAE** extends MAE to video by masking spatiotemporal tubes, achieving strong video understanding with even higher masking ratios (90–95%) due to temporal redundancy.

**Point-MAE** applies the masked reconstruction idea to 3D point clouds, masking groups of points and reconstructing them with a lightweight decoder.

**AudioMAE** applies patch masking to mel-spectrogram representations of audio, achieving state-of-the-art audio classification.

**MAE-style pretraining has become the dominant self-supervised approach for ViT-based models** — largely replacing contrastive learning due to its simplicity, scalability, and strong fine-tuning performance across vision, video, audio, and 3D modalities.
