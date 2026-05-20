---
title: Neural Image Compression
description: Explore learned image compression — covering the hyperprior model, entropy coding with neural networks, quantization and rate-distortion optimization, vector quantization approaches, and how neural codecs surpass JPEG and WebP at low bitrates.
---

Traditional image codecs (JPEG, WebP, HEVC/BPG) rely on hand-crafted transforms — DCT, wavelet, or prediction filters — followed by entropy coding. **Neural image compression** replaces these components with learned transforms trained end-to-end to minimize a rate-distortion objective. At low bitrates, neural codecs now consistently outperform all classical standards by significant margins, and the gap continues to grow as architectures improve.

## The Rate-Distortion Framework

Image compression seeks the best tradeoff between file size (rate $R$, in bits per pixel) and reconstruction quality (distortion $D$):

$$\mathcal{L} = R + \lambda \cdot D$$

where $\lambda$ controls the operating point on the rate-distortion curve. Minimizing $\mathcal{L}$ simultaneously reduces bits and reconstruction error.

- **Rate**: $R = \mathbb{E}[-\log_2 p(y)]$, the expected number of bits to encode latent $y$
- **Distortion**: $D = d(x, \hat{x})$, typically MSE (correlated with PSNR) or a perceptual metric like MS-SSIM or LPIPS

The challenge: quantization (converting continuous latents to discrete codes) is non-differentiable. Neural compression requires differentiable approximations to enable end-to-end training.

## The Basic Autoencoder Framework

The foundational neural compression architecture (Ballé et al., 2017) is an asymmetric autoencoder:

$$\text{Encoder: } y = g_a(x; \phi) \qquad \text{Quantization: } \hat{y} = \lfloor y \rceil \qquad \text{Decoder: } \hat{x} = g_s(\hat{y}; \theta)$$

where $g_a$ is the analysis transform (encoder) and $g_s$ is the synthesis transform (decoder).

### Differentiable Quantization

Hard rounding $\lfloor y \rceil$ has zero gradient almost everywhere. Two common approximations:

1. **Additive uniform noise**: replace $\lfloor y \rceil$ with $y + \mathcal{U}(-\frac{1}{2}, \frac{1}{2})$ during training — the distribution of noisy values approximates the distribution of rounded values
1. **Straight-through estimator (STE)**: pass gradients through the rounding operation as if it were the identity

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Identity gradient

def quantize(y, training=True):
    if training:
        # Additive noise approximation
        return y + torch.zeros_like(y).uniform_(-0.5, 0.5)
    else:
        return RoundSTE.apply(y)
```

### Entropy Model

The rate term requires a learned probability model $p(\hat{y})$ for entropy coding. A fully factorized model (Ballé et al., 2017) assumes independent latents with learned marginal distributions:

$$p(\hat{y}) = \prod_i p_i(\hat{y}_i)$$

Each marginal $p_i$ is a learnable parametric density (mixture of logistics or learned CDF). During compression, arithmetic coding uses these probabilities to encode $\hat{y}$ near its entropy bound.

## The Hyperprior Model

The factorized model ignores spatial correlations in $\hat{y}$ — nearby latents tend to have similar magnitudes. The **hyperprior model** (Ballé et al., 2018) introduces a second level of latent variables $z$ that captures these correlations:

$$p(\hat{y} \mid \hat{z}) = \prod_i \mathcal{N}(\mu_i(\hat{z}), \sigma_i(\hat{z}))$$

The hyperencoder $h_a$ compresses $y$ to $z$; the hyperdecoder $h_s$ decodes $\hat{z}$ into per-element means and scales for $\hat{y}$:

```text
x → g_a → y → quantize → ŷ → g_s → x̂
              ↓
           h_a → z → quantize → ẑ → h_s → (μ, σ) → entropy model for ŷ
```

The total rate is:

$$R = \underbrace{\mathbb{E}[-\log_2 p(\hat{y}|\hat{z})]}_{\text{main latent rate}} + \underbrace{\mathbb{E}[-\log_2 p(\hat{z})]}_{\text{hyperlatent rate}}$$

The hyperlatent $\hat{z}$ is small (typically $\frac{1}{4}$ the resolution of $\hat{y}$), so its rate overhead is modest.

```python
import torch
import torch.nn as nn
from compressai.models import ScaleHyperprior
from compressai.losses import RateDistortionLoss

# Pre-built hyperprior model
model = ScaleHyperprior(N=128, M=192)

criterion = RateDistortionLoss(lmbda=0.01)  # Lambda controls bitrate

# Training step
x = batch["image"]  # (B, 3, H, W), normalized to [0, 1]
out = model(x)      # Contains "x_hat", "likelihoods"

loss = criterion(out, x)
# loss["loss"]: combined rate-distortion loss
# loss["bpp_loss"]: bits per pixel
# loss["mse_loss"]: reconstruction MSE
```

## Attention and Context Models

### Channel-Conditional Entropy Model

Rather than predicting each latent independently, **channel-conditional** models predict the distribution of channel $c$ conditioned on previously decoded channels $1, \ldots, c-1$:

$$p(\hat{y}^{(c)} \mid \hat{y}^{(1)}, \ldots, \hat{y}^{(c-1)}, \hat{z})$$

This captures cross-channel correlations at no spatial decoding cost, as channels can be decoded sequentially.

### Spatial Context (Checkerboard)

Spatial context models condition each latent on its spatial neighbors, enabling more accurate probability estimates but requiring sequential decoding. The **checkerboard model** (He et al., 2021) divides the latent grid into two sets (like a chess board):

1. **Anchor set** (black squares): decoded in parallel using only the hyperprior
1. **Non-anchor set** (white squares): decoded in parallel using hyperprior + neighboring anchors

This provides 2-pass parallel decoding instead of fully sequential, dramatically reducing decoding time while capturing spatial correlations.

### Transformer-Based Compression

Vision transformers replace convolutional encoders/decoders, enabling attention over long-range dependencies. **Swin-based compression** (Zhu et al., 2022) uses shifted-window attention:

- Captures non-local correlations that DCT and convolutions miss
- Window-partitioned attention for computational tractability
- Achieves state-of-the-art rate-distortion performance across all quality levels

## Vector Quantization

Rather than scalar quantization of continuous latents, **VQ-based codecs** use a learned codebook:

$$\hat{y} = \arg\min_{e_k \in \mathcal{E}} \|y - e_k\|_2$$

The closest codebook entry replaces the continuous latent. The codebook $\mathcal{E} = \{e_1, \ldots, e_K\}$ is learned alongside encoder/decoder.

VQ enables **extreme compression**: with $K = 1024$ codebook entries and $d$-dimensional vectors, each latent code needs only $\log_2(1024) = 10$ bits. This is the foundation of neural codecs like VQ-VAE-2 and the tokenizers used in image generation models (DALL-E, VQGAN).

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # z: (B, C, H, W)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])

        # Distances to codebook entries
        d = (z_flat**2).sum(1, keepdim=True) \
            - 2 * z_flat @ self.embedding.weight.T \
            + (self.embedding.weight**2).sum(1)

        encoding_indices = d.argmin(1)
        quantized = self.embedding(encoding_indices).view(z.permute(0, 2, 3, 1).shape)
        quantized = quantized.permute(0, 3, 1, 2)

        # Commitment + codebook loss
        loss = nn.functional.mse_loss(quantized.detach(), z) \
             + self.commitment_cost * nn.functional.mse_loss(quantized, z.detach())

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        return quantized_st, loss, encoding_indices
```

## Perceptual and Generative Compression

MSE-optimized codecs tend to produce blurry reconstructions at very low bitrates — they average over uncertainty. **Perceptual codecs** replace or augment the distortion term:

$$D = \alpha \cdot \text{MSE} + \beta \cdot \mathcal{L}_{\text{perceptual}} + \gamma \cdot \mathcal{L}_{\text{GAN}}$$

- $\mathcal{L}_{\text{perceptual}}$: distance in VGG feature space (captures texture and structure better than pixel-level MSE)
- $\mathcal{L}_{\text{GAN}}$: adversarial loss from a discriminator (generates plausible high-frequency detail)

**Generative compression** (Mentzer et al., 2020; HiFiC) at very low bitrates produces photorealistic reconstructions by synthesizing plausible detail consistent with the compressed bits — sacrificing some fidelity for perceptual quality. This is measured by **FID** and **LPIPS** rather than PSNR.

## Codec Comparison

| Codec | Standard/Neural | Release | Typical use |
| --- | --- | --- | --- |
| JPEG | Classical (DCT) | 1992 | Web images, legacy |
| WebP | Classical (VP8) | 2010 | Web images |
| HEVC/BPG | Classical (hybrid) | 2013 | Video frames, HDR |
| BPG | Classical (HEVC intra) | 2014 | High-quality stills |
| Ballé 2018 | Neural (hyperprior) | 2018 | Research baseline |
| Cheng2020 | Neural (attention+AR) | 2020 | Strong neural baseline |
| VTM (VVC intra) | Classical | 2020 | State-of-the-art classical |
| Zou2022 | Neural (Swin-based) | 2022 | SotA neural |

Neural codecs consistently outperform JPEG by 30–50% in BD-Rate (bits saved at matched quality) and match or beat VTM at low-to-medium quality levels.

## Evaluation Metrics

### PSNR

$$\text{PSNR} = 10 \log_{10} \frac{255^2}{\text{MSE}}$$

Higher is better; measured in dB. Correlates with MSE-optimized codecs but poorly reflects perceptual quality.

### MS-SSIM

Multiscale structural similarity measures luminance, contrast, and structure at multiple resolutions. More correlated with human perception than PSNR, especially at low bitrates.

### BD-Rate

Bjøntegaard Delta Rate measures the average bitrate saving (or overhead) of one codec vs another across a range of quality levels. Negative BD-Rate means codec A saves bits compared to codec B at matched quality.

### FID / LPIPS

For generative/perceptual codecs, Fréchet Inception Distance and Learned Perceptual Image Patch Similarity better capture how natural the output looks to human observers.

## Summary

Neural image compression has matured from research curiosity to a competitive alternative to classical codecs:

- **Autoencoder + entropy model**: the foundation — learned transforms with differentiable quantization and arithmetic coding
- **Hyperprior**: two-level latent hierarchy capturing spatial statistics of the main latent
- **Context models**: checkerboard and channel-conditional schemes reduce rate by modeling correlations within the latent grid
- **Transformer encoders**: attention over long-range dependencies further improves rate-distortion curves
- **Vector quantization**: discrete codebooks enabling extreme compression and compatibility with generative models
- **Perceptual/generative codecs**: sacrifice pixel fidelity for human-perceptual quality at very low bitrates

As inference speed improves (GPU decoders, faster entropy coding), neural codecs are moving from research benchmarks toward deployment in streaming, storage, and real-time video applications.
