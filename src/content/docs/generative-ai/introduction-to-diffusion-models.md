---
title: Introduction to Diffusion Models
description: Exploring the generative power behind AI-generated images.
---

Diffusion models are a class of generative models that have revolutionized the field of AI-generated art and images. They work by iteratively refining data starting from random noise.

## How Diffusion Works

The diffusion process consists of two stages:

1. **Forward Diffusion (Adding Noise)**: A clean image is gradually corrupted by adding Gaussian noise until it becomes pure random noise.
2. **Reverse Diffusion (Removing Noise)**: The model is trained to reverse this process—predicting and subtracting the noise at each step—to reconstruct the original image.

Once trained, if you give a diffusion model a patch of random noise and a text prompt, it can "denoise" the noise into a high-quality image that matches the prompt's description.

## Popular Diffusion Models

- **Stable Diffusion**: An open-source model capable of generating high-resolution images.
- **DALL·E**: A proprietary model by OpenAI known for its creativity and prompt adherence.
- **Midjourney**: A popular tool focused on high-quality artistic outputs.

## Key Concepts

- **Latent Space**: The compressed representation where the diffusion process often happens.
- **Guidance Scale**: A parameter that controls how strictly the model follows the text prompt.
- **U-Net Architecture**: A common neural network structure used to predict noise at each step.
