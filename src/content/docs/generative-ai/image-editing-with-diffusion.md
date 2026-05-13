---
title: Image Editing with Diffusion Models
description: Learn how diffusion models enable powerful image editing capabilities — including text-guided editing, inpainting, style transfer, and fine-grained control — through techniques like InstructPix2Pix, DreamBooth, SDEdit, and Prompt-to-Prompt.
---

Diffusion models, originally designed for text-to-image generation, have become the dominant framework for image editing. By exploiting the iterative denoising process and the structured latent space of trained models, a rich family of editing techniques has emerged — from text-guided modifications to identity-preserving style transfer — without requiring retraining from scratch.

## The Image Editing Problem

Image editing with generative models requires balancing two competing objectives:

- **Fidelity**: the edited output should preserve the structure, identity, or content of the input image
- **Editability**: the output should faithfully reflect the specified change (text prompt, exemplar, or instruction)

Different techniques make different trade-offs along this spectrum.

## SDEdit: Stochastic Differential Editing

**SDEdit** (Meng et al., 2021) is one of the earliest and most elegant approaches. The key insight: a diffusion model's forward process adds noise, and the reverse process denoises. If you add a controlled amount of noise to an existing image and then run the reverse denoising from that point, the model "repairs" the image while being guided by the text condition.

### Algorithm

Given an input image $x_0$ and a target noise level $t^*$:

1. Add noise to obtain $x_{t^*} = \sqrt{\bar\alpha_{t^*}} x_0 + \sqrt{1 - \bar\alpha_{t^*}} \epsilon$
2. Run the reverse diffusion from $x_{t^*}$ conditioned on the new text prompt $c$

$$x_{t^*-1}, \ldots, x_0' \sim p_\theta(x_{t-1} | x_t, c)$$

The trade-off parameter $t^*$ controls fidelity vs. editability:

- **Small $t^*$** (low noise): output closely resembles input; edits are subtle
- **Large $t^*$** (high noise): more creative freedom; original structure may be lost

SDEdit works with any pre-trained diffusion model and requires no optimization at test time.

## Prompt-to-Prompt

**Prompt-to-Prompt** (Hertz et al., 2022) enables text-guided editing by manipulating the **cross-attention maps** that link text tokens to spatial image regions.

### Cross-Attention Injection

During generation with the original prompt $P$, the cross-attention maps $\{M_t\}$ encode which words attend to which spatial locations. To edit with a new prompt $P^*$ while preserving structure:

1. Generate the original image with $P$, recording all cross-attention maps $M_t$
2. Generate a new image with $P^*$, but **inject** the original maps $M_t$ at each step

$$\hat{x}_t = \text{Denoise}(x_t, P^*, M_t^{\text{original}})$$

Because the spatial layout is determined by cross-attention, injecting the original maps preserves composition while the new prompt changes semantics.

### Edit Types

| Edit Type | Mechanism | Example |
| --- | --- | --- |
| Word swap | Replace token, inject original maps | "a dog" → "a cat" |
| Prompt refinement | Add tokens, blend attention | "a house" → "a red house" |
| Attention re-weighting | Scale specific token attention | Make sky more prominent |

## InstructPix2Pix

**InstructPix2Pix** (Brooks et al., 2023) trains a diffusion model to follow natural language editing instructions directly, using a synthetic training dataset.

### Training Data Construction

1. Generate diverse (image, caption) pairs using a text-to-image model
2. Use GPT-4 to generate plausible editing instructions: "caption" → "edited caption" (e.g., "a sunny beach" → "make it winter")
3. Generate the edited image using Prompt-to-Prompt on the original image/caption pair
4. Train a conditional diffusion model $p_\theta(x | c_T, c_I)$ where $c_T$ is the instruction and $c_I$ is the input image

### Dual Classifier-Free Guidance

At inference, two guidance scales control the trade-off:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, \emptyset, \emptyset) + s_T (\epsilon_\theta(x_t, c_T, c_I) - \epsilon_\theta(x_t, \emptyset, c_I)) + s_I (\epsilon_\theta(x_t, \emptyset, c_I) - \epsilon_\theta(x_t, \emptyset, \emptyset))$$

- $s_T$: text guidance scale (how closely to follow the instruction)
- $s_I$: image guidance scale (how closely to preserve the original image)

This allows intuitive control: high $s_I$ preserves content, high $s_T$ applies stronger edits.

## DreamBooth

**DreamBooth** (Ruiz et al., 2023) enables personalized generation and editing by fine-tuning a diffusion model on a small set (3–30 images) of a specific subject.

### Method

1. Bind the subject to a rare text token: `[V]` (e.g., "a [V] dog")
2. Fine-tune the full diffusion model with two losses:
   - **Reconstruction loss**: minimize denoising error on input images with the rare token prompt
   - **Prior preservation loss**: sample from the original model with a generic class prompt (e.g., "a dog") and include those in training to prevent language drift

$$\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t^{\text{subject}}, t, c_{\text{subject}})\|^2] + \lambda \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t^{\text{prior}}, t, c_{\text{prior}})\|^2]$$

After fine-tuning, the model can generate the specific subject in arbitrary contexts: "a [V] dog on the moon."

### Applications

- Portrait editing (age progression, style changes)
- Product visualization in new scenes
- Personalized avatar generation

## Textual Inversion

**Textual Inversion** (Gal et al., 2022) is a lighter alternative to DreamBooth: instead of fine-tuning the model, it optimizes a new token embedding $v^*$ such that generating with the prompt "a photo of $v^*$" produces the target concept.

$$v^* = \arg\min_v \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, c(v))\|^2]$$

Only the single embedding vector is optimized; the model weights are frozen. This is faster and more memory-efficient but less expressive than DreamBooth.

## DALL-E Inpainting and Outpainting

Inpainting — filling a masked region of an image with semantically consistent content — is a natural application of diffusion models. The model is conditioned on the unmasked region:

$$p_\theta(x_{\text{mask}} | x_{\text{context}}, c)$$

Modern implementations (Stable Diffusion inpainting, DALL-E 3) fine-tune models specifically for inpainting by randomly masking training images. **Outpainting** extends the image beyond its original borders by treating the original image as context for generating new surrounding content.

## ControlNet for Editing

**ControlNet** (Zhang et al., 2023) adds spatial conditioning signals — edge maps, depth maps, pose skeletons, segmentation masks — to a frozen diffusion model via trainable copies of its encoder:

- **Canny edges**: edit while preserving structure
- **Depth conditioning**: change style/texture while maintaining 3D layout
- **Pose conditioning**: transfer human pose to a new character

ControlNet enables fine-grained spatial control that text prompts alone cannot provide, making it valuable for editing images where specific structural elements must be preserved.

## Null-Text Inversion

A core challenge in editing real (not AI-generated) images is **inversion** — finding a noise trajectory that reconstructs the original image when reversed. Standard DDIM inversion accumulates errors in classifier-free guidance settings.

**Null-text inversion** (Mokady et al., 2023) optimizes the null-text embedding (the unconditional prompt) to minimize reconstruction error at each denoising step, without modifying the model or input. This achieves near-perfect reconstruction of real images, enabling high-fidelity editing with Prompt-to-Prompt on photographs.

## IP-Adapter: Image Prompting

**IP-Adapter** (Ye et al., 2023) adds image conditioning to any diffusion model without fine-tuning, using a decoupled cross-attention mechanism:

- A separate cross-attention layer processes image features (from CLIP image encoder) in parallel with the text cross-attention layers
- At inference, both text and image prompts guide generation

This enables style transfer (use a reference image as style), face consistency (keep identity across generated images), and composition control.

## Comparison of Editing Approaches

| Method | Requires Training | Input | Strength |
| --- | --- | --- | --- |
| SDEdit | No | Image + prompt | Simple, general-purpose |
| Prompt-to-Prompt | No | Image + prompts | Structure-preserving word swap |
| InstructPix2Pix | Pretrained | Image + instruction | Natural language instructions |
| DreamBooth | Fine-tuning (~15 min) | 5–30 subject images | Subject personalization |
| Textual Inversion | Optimization (~1 hr) | 5–20 concept images | Lightweight concept capture |
| ControlNet | Pretrained | Image + spatial map | Spatial structure control |
| IP-Adapter | Pretrained | Image + reference | Style and appearance transfer |

## Practical Considerations

### Choosing an Approach

- **Quick text edit on AI image**: Prompt-to-Prompt or SDEdit
- **Edit a real photograph**: Null-text inversion + Prompt-to-Prompt
- **Follow natural language instructions**: InstructPix2Pix
- **Personalize to a specific person/object**: DreamBooth
- **Control spatial structure**: ControlNet

### Artifact Avoidance

Common artifacts in diffusion-based editing include:

- **Identity drift**: the edited image no longer looks like the original subject — reduce noise level in SDEdit or increase $s_I$ in InstructPix2Pix
- **Color shift**: overly saturated or hue-shifted output — use lower guidance scales
- **Inconsistent masking boundaries**: visible seams in inpainting — use Gaussian-blended masks and latent blending

## Summary

Diffusion-based image editing has evolved rapidly from simple noise-and-denoise approaches (SDEdit) to sophisticated attention manipulation (Prompt-to-Prompt), personalization (DreamBooth), and spatial control (ControlNet). The common thread is leveraging the rich internal representations of pre-trained diffusion models — their cross-attention maps, latent codes, and denoising trajectories — to achieve edits that were previously only possible with manual retouching or specialized generative models.
