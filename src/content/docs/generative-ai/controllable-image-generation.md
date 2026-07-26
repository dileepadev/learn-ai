---
title: Controllable Image Generation - Steering Diffusion Models Beyond Text Prompts
description: Learn how techniques like ControlNet, inpainting masks, and reference conditioning give precise spatial control over generated images.
---

Text prompts alone give coarse control over image generation—they describe *what* should appear but not precisely *where* or in *what pose*. Controllable generation methods add extra conditioning signals so users can specify structure directly.

```text
text prompt only:        "a cat sitting on a chair" -> plausible but unpredictable pose/layout
text + control signal:   "a cat sitting on a chair" + edge map / pose skeleton -> matches exact layout
```

## ControlNet

ControlNet augments a pretrained diffusion model with a parallel trainable branch that accepts an additional input—an edge map, depth map, pose skeleton, or segmentation mask—and injects that structural information into the denoising process at multiple layers. Crucially, the original model's weights are frozen and copied, with the new branch's outputs initialized to have no effect at the start of training, so training the control branch doesn't degrade the base model's existing image quality.

## Inpainting and Outpainting

**Inpainting** regenerates only a masked region of an existing image while keeping the rest fixed, useful for object removal or replacement. **Outpainting** extends an image beyond its original borders, generating plausible new content that blends with existing edges. Both rely on conditioning the diffusion process on the unmasked pixels at every denoising step, so the generated region stays consistent with the surrounding context.

## Reference-Based and Identity Conditioning

Some methods condition generation on a reference image rather than (or in addition to) text, letting a model preserve a specific subject's appearance across new scenes and poses. Techniques like DreamBooth fine-tune a model on a handful of images of a specific subject, binding a rare token to that subject's identity so it can be recombined with new prompts (for example, that same subject in a different setting).

## Trade-offs

- **Precision vs. flexibility**: strong structural conditioning (an exact pose skeleton) yields more predictable output but reduces creative variation.
- **Extra compute**: most control mechanisms add extra network branches or conditioning passes, increasing inference cost over plain text-to-image generation.
- **Control leakage**: overly strict conditioning signals (e.g., a depth map) can sometimes force artifacts if the requested content doesn't naturally fit the given structure.

Controllable generation is what makes diffusion models usable for practical creative and design workflows—product mockups, storyboarding, and iterative editing—where exact layout and consistency matter more than open-ended novelty.
