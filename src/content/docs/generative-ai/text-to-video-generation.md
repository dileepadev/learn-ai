---
title: Text-to-Video Generation
description: Explore how AI models generate video from text prompts — covering diffusion-based architectures, temporal consistency, world models, and the state of the art from Sora to open-source alternatives.
---

**Text-to-video generation** is the task of producing a coherent video clip from a natural language description. It extends the breakthroughs of text-to-image diffusion models into the temporal domain, requiring the model to maintain visual consistency across frames while depicting motion, scene transitions, and causally plausible physics.

## Why Video Is Harder Than Images

Generating a single image requires modeling spatial coherence. Video adds the requirement of **temporal coherence** — every frame must be consistent with those before and after it.

Specific challenges:

- **Temporal consistency**: Objects must not change shape, color, or identity across frames.
- **Motion realism**: Movement should follow physical laws — gravity, momentum, fluid dynamics.
- **Long-range coherence**: A 10-second clip at 24 fps has 240 frames to keep consistent.
- **Computational cost**: Generating video requires far more memory and compute than image generation.
- **Evaluation difficulty**: Perceptual quality metrics (FID, CLIP score) are insufficient; video requires temporal metrics.

## Architectures for Video Generation

### Diffusion Models with Temporal Layers

The dominant approach extends image diffusion models by inserting **temporal attention** or **3D convolution** layers that operate across the time dimension.

**U-Net with temporal attention:**

$$z_{t,i} = \text{Attn}_\text{spatial}(z_{t,i}) + \text{Attn}_\text{temporal}(z_{\cdot,i})$$

Where $z_{t,i}$ is the latent at time step $t$ and spatial position $i$.

Representative models:

- **ModelScope / ZeroScope** — early open models adapting Stable Diffusion with temporal convolutions.
- **AnimateDiff** — a plug-in temporal motion module that can be combined with any fine-tuned image diffusion model.
- **Stable Video Diffusion (SVD)** — Stability AI's image-conditioned video generation model trained on a large curated dataset.

### Latent Video Diffusion

Training and running diffusion in pixel space is prohibitively expensive for video. **Latent video diffusion** encodes each frame into a compressed latent space using a **Variational Autoencoder (VAE)**, then runs diffusion in that compressed space.

$$\mathcal{L}(\theta) = \mathbb{E}_{z, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$$

Where $z$ is the latent video tensor, $c$ is the text conditioning, and $\epsilon_\theta$ is the noise prediction network.

### Transformer-Based Video Generation

Replacing the U-Net with a **Transformer (DiT)** backbone enables better scaling and conditioning.

**OpenAI Sora (2024)** is the most prominent example:

- Models video as a sequence of **spacetime patches**.
- Uses a **Diffusion Transformer (DiT)** that processes 3D patch tokens.
- Trained on a large, curated dataset of videos with descriptive captions (generated via video captioning models).
- Capable of generating up to 60 seconds of high-resolution (1080p) video.
- Demonstrated emergent understanding of physics, scene continuity, and camera motion.

**Key Sora insights:**

1. Treating video as spatiotemporal patches rather than frame sequences enables flexible duration/resolution.
2. Scaling laws from image generation transfer to video when using similar transformer architectures.
3. Video generation can serve as a general-purpose **world simulator** when trained at scale.

## Conditioning Mechanisms

Text-to-video models can be conditioned on various signals:

| Conditioning Type | Description | Example Use |
|---|---|---|
| **Text prompt** | CLIP or T5 text embedding | "A golden retriever running on a beach" |
| **Image conditioning** | Starting frame drives video content | Stable Video Diffusion |
| **Motion vectors** | Explicit motion guidance | AnimateDiff motion LoRAs |
| **Audio conditioning** | Sound drives visual movement | Emu Video (audio-driven) |
| **Depth / pose** | Structural control | ControlVideo |
| **Reference video** | Style or motion transfer | Video-to-video editing |

## Temporal Consistency Techniques

Maintaining consistency across frames is a core technical challenge.

### Attention over Time

**Temporal self-attention** allows every frame to attend to all other frames:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Computed along the time axis, enabling information propagation across frames without explicit recurrence.

### Optical Flow Supervision

During training or fine-tuning, **optical flow** losses encourage predicted motion between frames to be consistent with physical motion fields.

### Video VAEs

Specialized VAEs with **3D convolution** in the encoder/decoder compress video temporally as well as spatially, ensuring the latent representation respects temporal structure.

## Evaluation Metrics

| Metric | Measures |
|---|---|
| **FID (Fréchet Inception Distance)** | Frame-level visual quality |
| **FVD (Fréchet Video Distance)** | Temporal + visual quality |
| **CLIP Score** | Semantic alignment with text prompt |
| **SSIM / PSNR** | Pixel-level reconstruction |
| **IS (Inception Score)** | Diversity and quality |
| **Human preference** | Perceptual evaluation by raters |

FVD is the most commonly reported metric for video generation quality.

## State of the Art (2025)

| Model | Organization | Notable Feature |
|---|---|---|
| **Sora** | OpenAI | 60s, 1080p, DiT-based |
| **Veo 2** | Google DeepMind | High realism, camera control |
| **Kling 2** | Kuaishou | 3-min video, strong physics |
| **Wan 2.1** | Alibaba | Open-source, strong benchmark results |
| **CogVideoX** | Zhipu AI | Open-source, multi-round editing |
| **HunyuanVideo** | Tencent | Open-source, high resolution |
| **LTX-Video** | Lightricks | Fast inference, open-source |

## Applications

- **Creative media**: Film pre-visualization, music videos, advertising.
- **Education**: Animated explanations of concepts.
- **Gaming**: Dynamic cutscene generation, procedural cinematic content.
- **Simulation**: Training data for robotics and autonomous driving.
- **Accessibility**: Converting descriptive text to video for users with visual impairments.

## Limitations and Open Problems

- **Hallucinated physics**: Models frequently produce impossible motion (objects passing through each other, gravity reversals).
- **Prompt-to-video alignment**: Long or complex prompts are often only partially reflected.
- **Duration limitations**: Most open-source models are limited to 3–8 seconds of high-quality video.
- **Compute cost**: Generating 10 seconds of 720p video can require minutes on high-end GPUs.
- **Identity consistency**: Characters and objects often drift in appearance over time.
- **Deepfake risk**: Photorealistic video generation raises significant misuse concerns.

## The Road to World Models

Text-to-video generation is increasingly viewed as a step toward **world models** — AI systems that build internal representations of physical reality and can simulate future states. A model that can generate physically plausible video from a description has, implicitly, learned a representation of how the world works.

This positions video generation at the intersection of perception, planning, and simulation — suggesting deep connections to robotics, reinforcement learning, and scientific simulation.

## Further Reading

- [Sora Technical Report — OpenAI (2024)](https://openai.com/research/video-generation-models-as-world-simulators)
- [Align your Latents — Blattmann et al. (2023)](https://arxiv.org/abs/2304.08818)
- [Stable Video Diffusion — Stability AI (2023)](https://arxiv.org/abs/2311.15127)
- [CogVideoX — Yang et al. (2024)](https://arxiv.org/abs/2408.06072)
