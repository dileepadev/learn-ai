---
title: AI Video Generation
description: Explore how AI systems generate photorealistic video from text, images, and motion prompts — covering video diffusion models, temporal consistency, Sora's world-simulation approach, and the technical challenges unique to video synthesis.
---

AI video generation extends image synthesis into the temporal dimension, producing coherent sequences of frames that depict realistic motion, physics, and scene continuity. From short clips to minute-long cinematic shots, generative video models have advanced from blurry 16-frame outputs in 2022 to high-resolution, physically plausible video in 2024–2025.

## Why Video Is Harder Than Images

Video generation inherits all the challenges of image synthesis and adds several more:

- **Temporal coherence:** Frame $t+1$ must be consistent with frame $t$ — objects cannot flicker, appear, or disappear without physical cause
- **Motion plausibility:** Movement must follow physical laws (gravity, inertia, occlusion)
- **Long-range consistency:** A character's appearance must stay consistent across hundreds of frames
- **Computational scale:** A 10-second video at 24fps is 240 frames — each needing separate synthesis while being mutually constrained
- **Storage and training cost:** Video datasets are orders of magnitude larger than image datasets

## Architectures for Video Generation

### Video Diffusion Models (VDM)
Video Diffusion Models extend image diffusion to 3D (space + time). A noise prediction network operates on a **video tensor** $x \in \mathbb{R}^{T \times H \times W \times C}$:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

The UNet backbone is extended with **temporal attention** and **3D convolutions** to capture dependencies across time:
- **Spatial attention:** Attends over pixels within each frame
- **Temporal attention:** Attends over the same spatial position across different frames
- **3D convolutions:** Learn local spatiotemporal patterns in a volumetric receptive field

### Cascaded Generation
Most production systems use a **cascade** to manage resolution:
1. **Keyframe model:** Generate low-resolution (e.g., 64×64) anchor frames
2. **Temporal interpolation:** Fill in intermediate frames between keyframes
3. **Spatial super-resolution:** Upscale to full resolution (e.g., 1280×720)

This decomposition makes the generation tractable and allows independent scaling of each stage.

### Latent Video Diffusion
Analogous to Stable Diffusion for images, **latent video diffusion** trains a video VAE to compress temporal sequences into a compact latent space, then runs diffusion in that compressed space:

$$\text{Video} \xrightarrow{\text{Encoder}} z \in \mathbb{R}^{T' \times h \times w \times c} \xrightarrow{\text{Diffusion}} \hat{z} \xrightarrow{\text{Decoder}} \hat{\text{Video}}$$

This reduces the generation cost dramatically — diffusion operates on a ~4–8x spatially downsampled, ~4x temporally compressed latent.

## Sora: Video as World Simulation

OpenAI's **Sora** (2024) marked a qualitative leap in video generation. Key technical contributions:

### Spacetime Patches
Sora encodes video as **spacetime patches** — fixed-size 3D tiles that tile both space and time — analogous to how Vision Transformers split images into 2D patches. This unified representation handles:
- Videos of any resolution (flexible patch tiling)
- Videos of any aspect ratio
- Videos of any duration

### Diffusion Transformer (DiT) Architecture
Sora uses a **Diffusion Transformer** (DiT) rather than a UNet backbone. The full video latent is processed with global self-attention across all spacetime patches, avoiding the locality bias of convolutions. This enables long-range spatiotemporal coherence.

### World Model Capabilities
Sora demonstrated emergent "physics" — simulated 3D consistency, plausible object permanence, and self-consistent camera motion — behaviors not explicitly trained for but arising from scale. OpenAI describes Sora as a step toward **general world simulators** that model how the physical world evolves over time.

## Key Systems and Capabilities

| System | Developer | Notable Feature |
|---|---|---|
| Sora | OpenAI | Up to 60s, world-simulation quality |
| Veo 2 | Google DeepMind | Cinematic quality, camera control |
| Runway Gen-3 Alpha | Runway | Real-time generation, motion brush |
| Kling | Kuaishou | High motion fidelity, lip-sync |
| CogVideoX | Zhipu AI | Open-source, long video |
| Wan | Alibaba | Open-source, 720P |
| Mochi 1 | Genmo | Open weights, smooth motion |

## Conditioning Modalities

Modern video generators accept multiple input types:

- **Text-to-video:** Natural language prompt → video (most common)
- **Image-to-video:** Single still image → animated video
- **Video-to-video:** Input video + style/motion prompt → transformed output
- **Camera control:** Explicit camera trajectory specification (pan, zoom, orbit)
- **Motion conditioning:** Sparse motion vectors or skeleton poses guide character movement

## Temporal Attention and Consistency Mechanisms

### Optical Flow Supervision
Some models augment the training loss with an optical flow consistency term, penalizing unrealistic motion between frames.

### Cross-Frame Attention
Injecting reference frame features into every generated frame (via cross-attention) enforces appearance consistency for characters and objects.

### Video Prediction Pre-training
Training on next-frame prediction as a pre-training task bootstraps temporal coherence before the full generation objective.

## Evaluation Metrics

| Metric | Measures |
|---|---|
| FID (video) | Realism of individual frames |
| FVD (Fréchet Video Distance) | Temporal realism + motion quality |
| CLIP Score | Text-video alignment |
| Motion Consistency | Flickering, teleporting artifacts |
| Human Preference Rate | Overall quality (RLHF-style evaluation) |

## Challenges and Limitations

- **Long-range consistency:** Most models still struggle with videos longer than ~30 seconds
- **Physics violations:** Objects pass through each other, change shape unexpectedly
- **Fine-grained motion control:** Precise hand/finger motion and lip-sync remain difficult
- **Training data:** High-quality licensed video data at scale is scarce and expensive
- **Inference cost:** A 10-second HD video requires minutes on high-end GPUs

## Ethical Concerns

### Synthetic Media and Deepfakes
Video generation enables creation of photorealistic synthetic video of real people — driving concerns about:
- Non-consensual intimate imagery
- Political disinformation and election interference
- Impersonation fraud

**Countermeasures:** Video watermarking (C2PA standard), deepfake detection classifiers, platform moderation policies.

### Copyright and Training Data
Large-scale video generation models are trained on web-scraped video content. Legal battles over whether this constitutes fair use are ongoing in multiple jurisdictions.

## Applications

- **Film and advertisement production:** Rapid prototyping of scenes, B-roll generation
- **Education:** Animated explanations of abstract concepts
- **Gaming:** Procedurally generated cinematic cutscenes
- **Accessibility:** Describing video content for visually impaired users
- **Scientific visualization:** Animating simulations, molecular dynamics

## Further Reading

- Ho et al. (2022), *Video Diffusion Models*
- Peebles & Xie (2023), *Scalable Diffusion Models with Transformers (DiT)*
- OpenAI (2024), *Sora Technical Report: Video Generation Models as World Simulators*
- Singer et al. (2022), *Make-A-Video: Text-to-Video Generation without Text-Video Data*
