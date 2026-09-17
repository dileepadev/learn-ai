---
title: Image Inpainting Techniques - Filling in Missing or Masked Regions
description: Learn how image inpainting reconstructs missing regions of an image, from classic diffusion-based interpolation to modern generative inpainting.
---

Image inpainting fills in missing, damaged, or masked regions of an image with plausible content, used for photo restoration, object removal, and as a core capability of modern generative image editors.

## Classic Approaches

Diffusion-based inpainting (in the image-processing sense, not the generative-model sense) propagates pixel values from the region's boundary inward, smoothly interpolating color and gradient — effective for small scratches or thin regions but incapable of hallucinating meaningful new structure. Patch-based methods (PatchMatch) search the rest of the image for similar-looking patches to copy into the missing region, producing better texture continuation for larger masked areas by reusing real content already present elsewhere in the image.

## Deep Learning Approaches

CNN-based inpainting models train on images with synthetic masks, learning to predict the masked region's content from the surrounding context, often combined with an adversarial loss (a discriminator judging whether the completed image looks real) to encourage sharp, realistic completions rather than blurry averages.

## Diffusion Model Inpainting

Modern inpainting is typically built on diffusion models: the unmasked region is kept fixed (or lightly noised and re-denoised) while the masked region is denoised from noise, conditioned on the surrounding unmasked content and optionally a text prompt describing what should appear there.

```text
mask region -> replace with noise
run diffusion denoising, conditioning each step on the unmasked pixels
result: masked region filled with new content consistent with the rest of the image
```

This is the mechanism behind "generative fill" and object-replacement features in modern image editors — text-conditioned inpainting lets a user describe what should appear in the masked region rather than only inferring it from surrounding pixels.

## Practical Guidance

For simple object removal where surrounding texture is fairly uniform (a plain sky, a solid wall), classic patch-based methods are fast and often sufficient. For inpainting that requires generating new coherent objects or scene elements, a diffusion-based inpainting model conditioned on a text prompt gives far more control and quality, at the cost of higher compute per edit.
