---
title: Image Tokenization and VQ-VAE
description: Learn how image tokenization converts continuous pixels into discrete visual tokens — enabling language-model-style generation over images — covering VQ-VAE's codebook learning and commitment loss, VQ-GAN's perceptual adversarial training, FSQ, DALL-E tokenization, Magvit-v2, and the role of image tokens in multimodal LLMs.
---

Language models generate text by predicting discrete tokens from a finite vocabulary. To apply the same recipe to images, we need to convert continuous pixel grids into compact sequences of discrete tokens. **Image tokenization** — transforming images into sequences of visual codes drawn from a learned codebook — is what makes language-model-style image generation possible. It is the foundation of DALL-E, GPT-4o's image understanding, and modern multimodal LLMs.

## Why Tokenize Images?

Pixels are poor tokens for autoregressive generation:

- A 256×256 RGB image has 196,608 continuous values — too long for an autoregressive sequence model
- Raw pixel prediction doesn't leverage the semantic structure of images — nearby tokens are nearly identical

Image tokenization compresses each image into a sequence of a few hundred to a few thousand discrete codes, where each code represents a learned visual patch concept. An 256×256 image might become 256 tokens from a codebook of 8,192 entries — a compression ratio of 768× while preserving perceptual quality.

## VQ-VAE: Vector Quantized Variational Autoencoder

VQ-VAE (van den Oord et al., 2017) replaces the continuous Gaussian latent space of a standard VAE with a **discrete codebook** $\mathcal{E} = \{e_1, \ldots, e_K\} \subset \mathbb{R}^d$.

### Architecture

```text
Image x → Encoder z_e → Quantize → z_q → Decoder → x̂

Quantization: z_q = e_k where k = argmin_j ||z_e - e_j||₂
```

The encoder maps each spatial location to a continuous vector $z_e(\mathbf{x})$. Quantization replaces it with the nearest codebook entry:

$$\hat{z} = e_k, \quad k = \arg\min_j \|z_e(\mathbf{x}) - e_j\|_2$$

### The Commitment Loss

Quantization is non-differentiable (the argmin has no gradient). VQ-VAE uses a **straight-through estimator**: during the backward pass, gradients flow through the quantization step as if it were an identity, directly updating the encoder.

The training objective:

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{reconstruction}} + \underbrace{\|\mathrm{sg}[z_e] - e\|_2^2}_{\text{codebook update}} + \underbrace{\beta\|z_e - \mathrm{sg}[e]\|_2^2}_{\text{commitment loss}}$$

where $\mathrm{sg}[\cdot]$ is the stop-gradient operator. The commitment loss (weighted by $\beta \approx 0.25$) prevents the encoder from growing arbitrarily and ensures encoded vectors stay close to their codebook entries.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 8192, embedding_dim: int = 256, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # z_e: (B, C, H, W) → flatten spatial → (B*H*W, C)
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)

        # Nearest-neighbor lookup
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dists.argmin(dim=1)
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        # Losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q, z_e.detach())

        # Straight-through gradient estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.view(B, H, W), codebook_loss + commitment_loss


class VQVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 256, num_codes: int = 8192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, latent_channels, 3, 1, 1),
        )
        self.quantizer = VectorQuantizer(num_codes, latent_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, in_channels, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, indices, vq_loss

    def encode_to_tokens(self, x):
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices  # (B, H/4, W/4) discrete token map
```

## VQ-GAN: Perceptual + Adversarial Training

VQ-VAE reconstructions tend to be blurry because MSE loss averages over pixel uncertainty. **VQ-GAN** (Esser et al., 2021) adds:

- **Perceptual loss**: feature-level MSE using a pretrained VGG/LPIPS network
- **Adversarial loss**: a patch discriminator (PatchGAN) encourages sharp, realistic reconstructions

$$\mathcal{L}_{\mathrm{VQGAN}} = \mathcal{L}_{\mathrm{recon}} + \lambda \cdot \mathcal{L}_{\mathrm{percept}} + \mathcal{L}_{\mathrm{adv}} + \mathcal{L}_{\mathrm{VQ}}$$

The discriminator weight $\lambda$ is adapted per-sample using the ratio of reconstruction gradient to adversarial gradient norms. VQ-GAN produces sharp, high-fidelity tokenizations at compression ratios of 4–16× in spatial dimensions (16–256× in token count).

## DALL-E Tokenization (dVAE)

DALL-E 1 (Ramesh et al., 2021) used a **discrete VAE** (dVAE) with Gumbel-softmax relaxation to train a differentiable codebook without straight-through estimators. Each 256×256 image was mapped to a 32×32 grid of tokens from a 8,192-entry codebook. The autoregressive Transformer then modeled the joint distribution of text tokens and image tokens.

## FSQ: Finite Scalar Quantization

FSQ (Mentzer et al., 2023) eliminates the VQ training instabilities (codebook collapse, commitment loss tuning) by rounding each channel of the encoder output to a finite set of integers:

$$z_{q,c} = \text{round}(\tanh(z_{e,c}) \cdot L_c) \in \{-L_c, \ldots, L_c\}$$

The codebook size is implicitly $\prod_c (2L_c + 1)$. Each code is simply a tuple of quantized channel values — no explicit codebook to train, no commitment loss. FSQ matches VQ-GAN quality with simpler training.

## Magvit-v2 and High-Fidelity Video Tokens

**Magvit-v2** (Yu et al., 2024) extended image tokenization to video using a 3D causal convolutional encoder with lookup-free quantization (LFQ) — a variant of FSQ applied to video spatiotemporal patches. A 256×256 video at 10fps is tokenized to ~5% of its original pixel count, enabling autoregressive generation of coherent video with a language-model backbone.

## Role in Multimodal LLMs

Image tokens from a VQ-VAE or VQ-GAN tokenizer are treated identically to text tokens by a language model:

```python
# Multimodal sequence: [TEXT TOKENS] + [IMAGE TOKENS]
text_tokens = tokenizer.encode("A photo of a ")   # e.g., [100, 200, 300]
image_tokens = vq_vae.encode_to_tokens(image)      # e.g., 256 visual codes (flattened)
image_flat = image_tokens.view(image_tokens.size(0), -1)

# Offset image tokens to avoid collision with text vocabulary
OFFSET = text_vocab_size
combined = torch.cat([
    torch.tensor(text_tokens),
    image_flat[0] + OFFSET,
], dim=0)
# Autoregressive model predicts text and image tokens in one sequence
logits = lm(combined.unsqueeze(0))
```

GPT-4o, Chameleon, and Show-o use variants of this approach — a unified vocabulary of text and image tokens processed by a single Transformer.

## Comparison of Tokenization Methods

| Method | Codebook Size | Training Stability | Reconstruction Quality | Video Support |
| --- | --- | --- | --- | --- |
| VQ-VAE | 512–8192 | Moderate (collapse risk) | Blurry (MSE) | 3D extension |
| VQ-GAN | 8192–16384 | Moderate | Sharp (perceptual) | VGAN-3D |
| dVAE (DALL-E) | 8192 | Good (Gumbel relaxation) | Good | No |
| FSQ | Implicit (product) | High | Sharp | Yes |
| Magvit-v2 | 262144 | High (LFQ) | Excellent | Yes |

## Summary

Image tokenization bridges continuous visual data and discrete sequence models:

- **VQ-VAE** introduced the codebook-quantization paradigm with straight-through gradients and commitment loss — making discrete image representation learnable end-to-end
- **VQ-GAN** added perceptual and adversarial losses to achieve sharp reconstructions, enabling high-quality autoregressive generation
- **FSQ** eliminates codebook training instabilities by rounding quantization, matching VQ-GAN quality with simpler training dynamics
- **Magvit-v2** extends spatial tokenization to spatiotemporal video tokens using 3D convolutions and lookup-free quantization
- In multimodal LLMs, image tokens and text tokens share a unified vocabulary — enabling a single Transformer to understand and generate both modalities
