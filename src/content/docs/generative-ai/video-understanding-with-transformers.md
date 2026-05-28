---
title: Video Understanding with Transformers
description: Learn how transformers are applied to video understanding — covering spatiotemporal attention in TimeSformer and Video BERT, masked video modeling with VideoMAE, efficient video transformers, temporal video grounding, action recognition benchmarks, and how video understanding connects to multimodal models.
---

Video is the dominant format for human visual communication — yet it presents fundamental challenges that image models cannot address: temporal dependencies, motion dynamics, scene transitions, and causal event structure. Transformers, which model arbitrary pairwise relationships through attention, are particularly well-suited to video: they can attend across time as naturally as across space. The past four years have seen transformers displace CNNs as the dominant architecture for video understanding.

## The Core Challenge: Spatiotemporal Modeling

A video clip of $T$ frames at resolution $H \times W$ contains $T \times H \times W$ spatiotemporal positions. For a modest 8-frame, 224×224 clip:

$$8 \times 14 \times 14 = 1568 \text{ tokens (with 16×16 patches)}$$

Full 3D attention over 1568 tokens is feasible, but scaling to longer videos (32+ frames) makes full spatiotemporal attention prohibitively expensive. The field has developed several decompositions.

## TimeSformer: Divided Space-Time Attention

**TimeSformer** (Bertasius et al., 2021) factorizes full spatiotemporal attention into separate temporal and spatial attention, applied sequentially within each Transformer block:

```text
Input tokens (T×H×W patches)
        ↓
  Temporal Attention     ← each spatial position attends across T frames
        ↓
  Spatial Attention      ← each temporal position attends across H×W patches
        ↓
  MLP block
```

This reduces attention complexity from $O(T^2H^2W^2)$ to $O(T^2 \cdot HW + (HW)^2 \cdot T)$ — a significant saving for long videos.

```python
import torch
import torch.nn as nn
import einops


class DividedSpaceTimeAttention(nn.Module):
    """TimeSformer-style factorized spatiotemporal attention."""

    def __init__(self, dim: int, num_heads: int = 8, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T*N, D)  where N = H*W/patch²
        B, TN, D = x.shape
        T = self.num_frames
        N = TN // T  # spatial tokens per frame

        # Temporal attention: each spatial position attends across frames
        x_t = einops.rearrange(x, "b (t n) d -> (b n) t d", t=T, n=N)
        x_t_attn, _ = self.temporal_attn(x_t, x_t, x_t)
        x = x + einops.rearrange(x_t_attn, "(b n) t d -> b (t n) d", b=B, n=N)
        x = self.norm1(x)

        # Spatial attention: each frame attends within its own spatial tokens
        x_s = einops.rearrange(x, "b (t n) d -> (b t) n d", t=T, n=N)
        x_s_attn, _ = self.spatial_attn(x_s, x_s, x_s)
        x = x + einops.rearrange(x_s_attn, "(b t) n d -> b (t n) d", b=B, t=T)
        x = self.norm2(x)
        return x
```

## Video BERT and Masked Video Language Modeling

**Video BERT** (Sun et al., 2019) extended BERT to video by jointly modeling video clips and captions. It treats video frames as a sequence of visual tokens (from a pretrained image feature extractor) and text tokens, applying masked token prediction across both modalities:

- Randomly mask 15% of video tokens and 15% of text tokens
- Predict masked tokens from the surrounding context
- Learn to align visual and linguistic representations

This produced a model capable of video captioning, video question answering, and zero-shot action retrieval.

## VideoMAE: Masked Autoencoders for Video

**VideoMAE** (Tong et al., 2022) applies masked autoencoder pretraining to video with an extremely high masking ratio — 90% of patches are masked. The key insight is that video has massive temporal redundancy (adjacent frames are similar), so the network must learn genuine temporal reasoning to reconstruct from sparse observations.

The architecture:

1. Divide each frame into 16×16 spatial patches and group into tubelets (e.g., 2 frames per tubelet)
1. Randomly mask 90% of tubelets — retain only 10%
1. Encode unmasked tubelets with a ViT encoder
1. Decode all tubelets (masked tokens are learnable vectors) with a lightweight decoder
1. Predict raw pixel values of masked tubelets

```python
from transformers import VideoMAEModel, VideoMAEConfig, AutoVideoProcessor
import torch

config = VideoMAEConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_frames=16,
    tubelet_size=2,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    decoder_hidden_size=384,
    decoder_num_hidden_layers=4,
    decoder_num_attention_heads=6,
)

model = VideoMAEModel(config)
processor = AutoVideoProcessor.from_pretrained("MCG-NJU/videomae-base")

# Load a video as (num_frames, C, H, W) tensor
video_frames = torch.randn(16, 3, 224, 224)
inputs = processor(list(video_frames.numpy()), return_tensors="pt")

# Bool mask: True = masked (90% of patches)
num_patches = (224 // 16) ** 2 * (16 // 2)   # spatial × temporal patches
mask = torch.zeros(1, num_patches, dtype=torch.bool)
mask_indices = torch.randperm(num_patches)[:int(0.9 * num_patches)]
mask[0, mask_indices] = True

outputs = model(**inputs, bool_masked_pos=mask)
```

## Action Recognition Benchmarks

Action recognition evaluates how well models classify human actions from video clips:

| Dataset | Classes | Clips | Typical Challenge |
| --- | --- | --- | --- |
| Kinetics-400 | 400 | 240K | General action diversity |
| Kinetics-700 | 700 | 650K | Fine-grained action classes |
| Something-Something V2 | 174 | 220K | Temporal reasoning (direction matters) |
| AVA | 80 | 430K spatio-temporal labels | Localized actions in clips |
| UCF-101 | 101 | 13K | Classic benchmark |

Something-Something V2 is particularly interesting for temporal reasoning: classes like "Moving something left" and "Moving something right" require understanding motion direction, not just object appearance — a test of genuine temporal modeling rather than spatial shortcut learning.

## Efficient Video Transformers

### MViT: Multiscale Vision Transformers

**MViT** (Fan et al., 2021) and **MViTv2** process video in a multiscale pyramid: early layers operate at high resolution with few channels, later layers at lower resolution with more channels — mirroring the design of CNNs. Pooling attention reduces the key/value sequence length in early layers by 2–8×, dramatically reducing memory while preserving spatial detail.

### Video Swin Transformer

**Video Swin** (Liu et al., 2022) extends Swin Transformer's shifted-window attention to 3D spatiotemporal windows. Each attention head attends within a local $T_w \times H_w \times W_w$ window, and windows are shifted between layers to enable cross-window connections. Video Swin achieves state-of-the-art action recognition with linear complexity in video length.

## Temporal Video Grounding

Beyond classification, **temporal grounding** localizes events in long videos given text queries:

- Input: a video + text query (e.g., "when does the person pour water?")
- Output: a temporal segment $(t_{\text{start}}, t_{\text{end}})$

**Moment-DETR** and **QD-DETR** apply DETR-style detection (bipartite matching + set prediction) to temporal grounding, predicting moment spans as a set rather than using sliding-window proposals.

## Video Understanding in Multimodal LLMs

Modern multimodal LLMs extend image understanding to video by:

- Sampling a fixed number of frames from the video (e.g., 8–32 frames)
- Encoding each frame independently with a vision encoder (CLIP ViT)
- Concatenating temporal frame tokens as the visual input to the LLM
- Fine-tuning with video QA pairs

**Video-LLaVA**, **LLaVA-NeXT-Video**, and **InternVL-Chat-V2** follow this recipe, achieving strong performance on video QA benchmarks (Video-MME, MVBench) while leveraging pretrained image encoders without video-specific pretraining.

```python
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import torch

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "What activity is the person performing?"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=prompt, videos=[video_frames], return_tensors="pt").to("cuda")

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=200)

print(processor.decode(output[0], skip_special_tokens=True))
```

## Summary

Transformers have become the dominant architecture for video understanding through a combination of flexible attention and self-supervised pretraining:

- **TimeSformer's factorized attention** separates temporal and spatial processing to scale to longer clips without quadratic cost in all dimensions
- **VideoMAE's 90% masking ratio** forces the model to learn genuine temporal reasoning rather than exploiting spatial redundancy
- **Video Swin and MViT** adapt efficient spatial window attention to the spatiotemporal domain, achieving strong accuracy with manageable compute
- **Action recognition benchmarks** like Something-Something V2 specifically test temporal reasoning, pushing beyond appearance-based shortcuts
- **Multimodal LLMs** handle video by treating temporal frame sequences as visual context, enabling natural language video QA without video-specific architecture changes
