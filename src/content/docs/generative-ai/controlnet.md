---
title: ControlNet
description: Learn how ControlNet adds precise spatial conditioning to diffusion models — enabling generation controlled by edge maps, depth, pose skeletons, segmentation masks, and normals — through a trainable copy of the UNet encoder connected with zero-initialized convolutions.
---

Text prompts alone cannot reliably control the spatial layout, pose, structure, or depth of generated images. **ControlNet** (Zhang et al., 2023) solves this by adding a parallel trainable copy of the diffusion UNet encoder that accepts spatial conditioning inputs — Canny edges, depth maps, human pose skeletons, segmentation masks, surface normals — and injects this structural guidance directly into the generation process while preserving the quality and diversity of the pre-trained model.

## The Control Problem

A text prompt like "a person standing in a forest" gives the model enormous freedom in pose, composition, and structure. For applications in character animation, architectural visualization, product photography, and medical illustration, you need pixel-precise control over spatial structure that text cannot express. Earlier approaches (SDEdit, img2img) modify the noisy input but don't provide reliable structural guidance.

ControlNet introduces an efficient fine-tuning architecture that adds spatial conditioning without degrading the base model's generation capability.

## Architecture

ControlNet adds a trainable parallel branch to the frozen diffusion UNet:

```text
Condition Image (edge map, depth, pose, etc.)
        │
        ▼
   ControlNet Encoder (trainable copy of UNet encoder)
   ┌────────────────────────────────────┐
   │  Down Block 1 → Zero Conv          │──────────────────────────────┐
   │  Down Block 2 → Zero Conv          │─────────────────────────┐    │
   │  Down Block 3 → Zero Conv          │────────────────────┐    │    │
   │  Mid Block    → Zero Conv          │───────────────┐    │    │    │
   └────────────────────────────────────┘               │    │    │    │
                                                        ▼    ▼    ▼    ▼
Text Prompt → CLIP Encoder → Cross-Attention      ┌──────────────────────┐
                                                   │  Frozen UNet Decoder  │
Noisy Latent ─────────────────────────────────►   │  Up Block 3           │
                                                   │  Up Block 2           │
                                                   │  Up Block 1           │
                                                   └──────────────────────┘
                                                           │
                                                           ▼
                                                   Denoised Latent
```

The ControlNet encoder is an exact copy of the UNet encoder — same architecture, same initial weights (copied from the pre-trained model). Only the ControlNet branch is trained; the original UNet is completely frozen.

## Zero Convolutions

The critical innovation enabling stable training is **zero-initialized convolutions** (zero convs): 1×1 convolutional layers with weights and biases both initialized to zero.

Before training begins, every zero conv outputs exactly zero — so the ControlNet branch contributes nothing. The frozen UNet generates exactly the same outputs as the original pre-trained model. Training signals from the condition then gradually teach the ControlNet to inject meaningful guidance, starting from a guaranteed stable initialization.

Formally, if $\mathcal{F}$ is the frozen UNet block output and $\mathcal{Z}(\mathcal{C})$ is the ControlNet branch output passed through a zero conv:

$$\text{output} = \mathcal{F}(\mathbf{x}) + \mathcal{Z}(\mathcal{C}(\mathbf{x}, \mathbf{c}))$$

At initialization, $\mathcal{Z} = 0$, so $\text{output} = \mathcal{F}(\mathbf{x})$ — identical to the original model. Gradient flow is well-defined from the start.

## Condition Types

ControlNet supports a wide range of spatial conditioning signals:

| Condition | Description | Use Case |
| --- | --- | --- |
| Canny edges | Edge map from Canny detector | Structure-preserving stylization |
| Depth map | Monocular depth estimation (MiDaS) | 3D composition control |
| HED edges | Soft holistically-nested edges | Sketch-to-image |
| MLSD lines | Straight line segments | Architectural / interior scenes |
| OpenPose | 18-point human body skeleton | Character pose control |
| DensePose | Full body UV surface mapping | Detailed body control |
| Segmentation | Semantic segmentation mask | Scene layout control |
| Surface normals | Per-pixel surface orientation | Lighting and material control |
| Scribble | Hand-drawn rough sketches | Casual user control |
| Inpainting | Binary mask of regions to fill | Targeted inpainting |

Each condition type requires a separately trained ControlNet checkpoint — a single ControlNet is trained for one condition type.

## Training

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import UniPCMultistepScheduler
import torch

# Training a custom ControlNet
from diffusers.training_utils import compute_snr
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")

# Load pre-trained UNet and VAE (frozen)
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

# Initialize ControlNet from UNet encoder weights
controlnet = ControlNetModel.from_unet(unet)

# Freeze UNet and VAE — only ControlNet trains
unet.requires_grad_(False)
vae.requires_grad_(False)
controlnet.train()

# Training loop
for batch in dataloader:
    # batch contains: pixel_values, conditioning_images, captions
    with accelerator.accumulate(controlnet):
        # Encode images to latents
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample() * vae.config.scaling_factor

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (len(latents),))
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        text_embeds = text_encoder(batch["input_ids"])[0]

        # ControlNet forward pass
        down_samples, mid_sample = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
            controlnet_cond=batch["conditioning_images"],
            return_dict=False,
        )

        # UNet forward pass with ControlNet guidance injected
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

## Inference

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import UniPCMultistepScheduler
import torch
from PIL import Image
import cv2
import numpy as np

# Load ControlNet (Canny edge variant)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Prepare Canny edge condition
image = np.array(Image.open("source.jpg"))
edges = cv2.Canny(image, threshold1=100, threshold2=200)
edges_rgb = Image.fromarray(np.stack([edges, edges, edges], axis=2))

# Generate with spatial control
result = pipe(
    prompt="a highly detailed oil painting of a mountain landscape",
    negative_prompt="blurry, low quality, deformed",
    image=edges_rgb,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,   # ControlNet influence (0.0–2.0)
).images[0]
```

### Conditioning Scale

The `controlnet_conditioning_scale` parameter controls how strongly the spatial condition guides generation:

- **0.0**: ControlNet is ignored — pure text-to-image
- **0.5–0.8**: loose structural guidance, more creative freedom
- **1.0**: balanced control (default)
- **1.5–2.0**: very strict adherence to condition structure

## Multi-ControlNet

Multiple ControlNets can be combined in a single generation, with per-condition weights:

```python
from diffusers import StableDiffusionControlNetPipeline

controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", ...)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", ...)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_depth],
    torch_dtype=torch.float16,
)

result = pipe(
    prompt="a character in a dramatic pose in an ancient temple",
    image=[canny_edges, depth_map],
    controlnet_conditioning_scale=[1.0, 0.8],   # Per-condition weights
).images[0]
```

## ControlNet 1.1 and SDXL ControlNet

ControlNet 1.1 improves the original with additional condition types and better quality:

- **Tile ControlNet**: constrains local patch content for upscaling without hallucinating new details
- **Shuffle ControlNet**: applies structural style transfer from a reference image
- **Instruct Pix2Pix ControlNet**: combines instruction-following with spatial conditioning

For SDXL (Stable Diffusion XL), ControlNet connects to the larger UNet with the same zero-conv mechanism but is trained on the SDXL encoder architecture:

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
```

## ControlNet with LoRA

ControlNet and LoRA can be combined: use LoRA to adapt the overall style/content of the base model, and ControlNet to enforce spatial structure simultaneously:

```python
pipe.load_lora_weights("custom_style_lora.safetensors")
pipe.fuse_lora(lora_scale=0.8)

# Now generation uses: frozen UNet + LoRA style + ControlNet structure
result = pipe(
    prompt="in the style of ukiyo-e woodblock print",
    image=canny_edges,
    controlnet_conditioning_scale=1.0,
).images[0]
```

## T2I-Adapter: A Lighter Alternative

T2I-Adapter achieves similar control with a smaller trainable network (only the adapter trains, not a copy of the encoder). Compared to ControlNet:

| Property | ControlNet | T2I-Adapter |
| --- | --- | --- |
| Trainable params | ~360M (copy of encoder) | ~77M (small adapter) |
| Training cost | Higher | Lower |
| Condition quality | Slightly stronger | Competitive |
| Multi-condition | Via multi-ControlNet | Native composability |
| Base model coupling | Tight (encoder copy) | Loose (additive residuals) |

## Summary

ControlNet enables precise spatial control over diffusion model generation through three key design decisions:

- **Encoder copy**: a trainable copy of the UNet encoder processes the conditioning signal using the same learned representations as the base model
- **Zero convolutions**: zero-initialized connections guarantee stable training by starting from the exact pre-trained model behavior
- **Frozen backbone**: the original UNet's generation quality and diversity are preserved while spatial control is added incrementally

The result is a flexible conditioning framework that works with any spatial signal — edges, depth, pose, segmentation — and composes naturally with other fine-tuning methods like LoRA, enabling production-grade controllable generation.
