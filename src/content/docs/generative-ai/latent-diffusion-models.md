---
title: Latent Diffusion Models
description: Deep-dive into Latent Diffusion Models (LDMs) — the architecture behind Stable Diffusion. Covers the three-component design (VAE, UNet, text encoder), the forward and reverse diffusion processes in latent space, classifier-free guidance, and how SDXL, SD3, and FLUX extend the original LDM framework.
---

**Latent Diffusion Models (LDMs)** solve a fundamental efficiency problem with diffusion models: running the iterative denoising process directly in pixel space is extraordinarily expensive — a 512×512 image contains 786,432 pixels, and each denoising step requires a full network forward pass. LDMs compress images into a compact **latent space** using a Variational Autoencoder, then run diffusion entirely in that lower-dimensional space. The result is a model that is 4–8× cheaper to train and run while producing equivalent or superior image quality.

## Three-Component Architecture

An LDM consists of three largely independent components that can be trained separately and combined modularly:

```
Text prompt → [Text Encoder] → text embeddings ─────────────────────────────┐
                                                                               ↓
Image → [VAE Encoder] → latent z → [Noisy latent z_t] → [UNet Denoiser] → [VAE Decoder] → Generated image
                                    (forward process)     (uses cross-attn    
                                                          with text embeddings)
```

### Component 1: The Variational Autoencoder

The VAE is trained separately (not end-to-end with the diffusion UNet). It learns to compress images into a latent space that is approximately 8× smaller in each spatial dimension (a 512×512×3 image becomes a 64×64×4 latent):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LDMEncoder(nn.Module):
    """
    VAE encoder for Latent Diffusion Models.
    Compresses spatial image to latent space: (B, 3, H, W) → (B, 8, H/8, W/8)
    The 8 channels encode mean and log-variance (4 each) of the latent distribution.
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4,
                 channel_mult: tuple = (1, 2, 4, 4), num_res_blocks: int = 2):
        super().__init__()
        base_channels = 128
        channels = [base_channels * m for m in channel_mult]
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for out_ch in channels:
            block = nn.Sequential(
                *[ResBlock(in_ch if i == 0 else out_ch, out_ch)
                  for i in range(num_res_blocks)],
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # 2× downsample
            )
            self.down_blocks.append(block)
            in_ch = out_ch
        
        # Bottleneck with attention
        self.mid = nn.Sequential(
            ResBlock(in_ch, in_ch),
            SelfAttention(in_ch),
            ResBlock(in_ch, in_ch)
        )
        
        # Output: 2 * latent_channels (mean + log_var)
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv2d(in_ch, 2 * latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        mean, log_var = h.chunk(2, dim=1)
        return mean, log_var


def reparameterize(mean: torch.Tensor, log_var: torch.Tensor,
                   scale_factor: float = 0.18215) -> torch.Tensor:
    """
    Reparameterization trick: sample z ~ N(mean, exp(log_var/2)).
    Scale factor 0.18215 normalizes latent variance to ~1 (empirical).
    This normalization is critical: diffusion training assumes unit-variance noise.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mean + eps * std
    return z * scale_factor   # scale into diffusion-friendly range
```

### Component 2: The UNet Denoiser

The UNet operates entirely in latent space. Its key modification from a standard UNet: **cross-attention layers** that condition on text embeddings at every resolution level:

```python
class CrossAttention(nn.Module):
    """
    Cross-attention for conditioning the UNet denoiser on text embeddings.
    Query comes from image features; key/value from text encoder output.
    """
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8,
                 head_dim: int = 64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (B, HW, query_dim) — flattened spatial features
        context: (B, seq_len, context_dim) — text encoder output
        """
        B, N, _ = x.shape
        h = self.num_heads
        
        q = self.to_q(x).reshape(B, N, h, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, h, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention over text tokens
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    Transformer block within the UNet: self-attn + cross-attn + FFN.
    Applied at each spatial resolution, with cross-attention conditioning on text.
    """
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, dim, num_heads)  # self-attn: context=x
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        
        # Self-attention on spatial features
        x_flat = x_flat + self.self_attn(self.norm1(x_flat), self.norm1(x_flat))
        # Cross-attention: attend to text embeddings
        x_flat = x_flat + self.cross_attn(self.norm2(x_flat), context)
        # Feed-forward
        x_flat = x_flat + self.ff(self.norm3(x_flat))
        
        return x_flat.transpose(1, 2).reshape(B, C, H, W)
```

### Component 3: The Text Encoder

Stable Diffusion 1.x uses **CLIP ViT-L/14** — a contrastive vision-language model whose text encoder produces 768-dimensional token embeddings. The full 77-token sequence is passed as context to all cross-attention layers. SD 2.x switched to OpenCLIP, and SDXL uses two encoders concatenated (OpenCLIP ViT-bigG + CLIP ViT-L).

## The Diffusion Process in Latent Space

### Forward Process (Adding Noise)

Given a latent $z_0 = \text{Enc}(x)$, the forward process gradually adds Gaussian noise over $T$ timesteps:

$$q(z_t | z_0) = \mathcal{N}(z_t;\; \sqrt{\bar{\alpha}_t}\, z_0,\; (1 - \bar{\alpha}_t)\mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ is the cumulative noise schedule. This allows sampling $z_t$ directly without iterating through all intermediate steps:

$$z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### Denoising Training Objective

The UNet $\epsilon_\theta$ is trained to predict the noise $\epsilon$ that was added at each timestep, conditioned on the text embedding $c$:

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t, c}\!\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

```python
def ldm_training_step(
    encoder, unet, text_encoder,
    images: torch.Tensor,       # (B, 3, H, W) normalized to [-1, 1]
    text_tokens: torch.Tensor,  # (B, 77) tokenized prompts
    noise_scheduler,
    optimizer
) -> torch.Tensor:
    """One LDM training step."""
    
    # 1. Encode images to latent space
    with torch.no_grad():
        mean, log_var = encoder(images)
        z0 = reparameterize(mean, log_var)   # (B, 4, H/8, W/8)
        
        # Encode text
        text_embeddings = text_encoder(text_tokens)  # (B, 77, 768)
    
    # 2. Sample random timesteps and add noise
    B = z0.shape[0]
    t = torch.randint(0, noise_scheduler.num_timesteps, (B,), device=z0.device)
    noise = torch.randn_like(z0)
    z_t = noise_scheduler.add_noise(z0, noise, t)   # (B, 4, H/8, W/8)
    
    # 3. Predict noise with UNet (conditioned on text)
    noise_pred = unet(z_t, t, context=text_embeddings)
    
    # 4. MSE loss between predicted and actual noise
    loss = F.mse_loss(noise_pred, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss
```

## Classifier-Free Guidance (CFG)

Classifier-free guidance (Ho & Salimans, 2022) dramatically improves prompt adherence without requiring a separate classifier. During training, 10–20% of text conditions are dropped (replaced with null embeddings). At inference, two UNet forward passes are run:

$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \varnothing) + \gamma \cdot \bigl(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing)\bigr)$$

where $\gamma$ is the guidance scale (typically 7–12 for Stable Diffusion). Higher $\gamma$ produces images that more closely match the prompt but with less diversity; $\gamma = 1$ is equivalent to no guidance.

```python
@torch.no_grad()
def ldm_inference(
    unet, vae_decoder, text_encoder,
    prompt: str,
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512
) -> torch.Tensor:
    """Standard Stable Diffusion inference with CFG."""
    
    # Encode both positive and negative (empty/negative) prompts
    text_emb = text_encoder(tokenize(prompt))         # (1, 77, 768)
    uncond_emb = text_encoder(tokenize(negative_prompt))  # (1, 77, 768)
    
    # Batch both for single UNet call
    embeddings = torch.cat([uncond_emb, text_emb])    # (2, 77, 768)
    
    # Start from pure noise in latent space
    latents = torch.randn(1, 4, height // 8, width // 8)
    
    # DDIM/PNDM scheduler denoising
    for t in noise_scheduler.timesteps:  # e.g., 50 steps from T=1000 to 0
        # Duplicate latents for classifier-free guidance
        latent_input = torch.cat([latents, latents])   # (2, 4, H, W)
        
        # Two forward passes in one batched call
        noise_pred_both = unet(latent_input, t, context=embeddings)
        noise_pred_uncond, noise_pred_cond = noise_pred_both.chunk(2)
        
        # Guidance formula
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        
        # Scheduler step: z_t → z_{t-1}
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents to pixel space
    latents = latents / 0.18215   # reverse the encoder scale factor
    image = vae_decoder(latents)  # (1, 3, H, W), range [-1, 1]
    return (image + 1) / 2        # normalize to [0, 1]
```

## Evolution: SDXL, SD3, and FLUX

### Stable Diffusion XL (SDXL)

- Two text encoders concatenated (OpenCLIP ViT-bigG + CLIP ViT-L → 2816-dim context)
- Larger UNet with more attention heads and cross-attention at more resolutions
- Native 1024×1024 resolution training (1024→128 latent)
- Refiner model (specialized for fine detail at the final denoising steps)

### Stable Diffusion 3 (SD3)

- Replaces the UNet with a **Multimodal Diffusion Transformer (MMDiT)**
- Text and image tokens processed jointly with full bidirectional attention
- Three text encoders (CLIP L, CLIP G, T5-XXL) for richer semantic conditioning
- Flow Matching training objective (rectified flow) instead of DDPM

### FLUX (Black Forest Labs)

- Hybrid architecture: Transformer blocks alternating between full joint attention and single-stream image-only attention
- Rotary Position Embeddings (RoPE) in image space
- Flow Matching on continuous-time noise schedules
- Scales to FLUX.1-pro with state-of-the-art text-image alignment

The LDM architecture established the template that every major image generation system now builds on: compress to latent space, denoise with a transformer-enhanced UNet, and condition via cross-attention from a powerful text encoder.
