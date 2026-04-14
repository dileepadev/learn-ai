---
title: Vision Language Models (VLMs)
description: Learn how Vision Language Models bridge the gap between visual perception and language understanding — covering architectures like CLIP, LLaVA, GPT-4V, and PaliGemma, and their applications in visual question answering, document AI, and grounded reasoning.
---

Vision Language Models (VLMs) are AI systems that jointly understand images and text, enabling tasks that require grounding language in visual perception — such as visual question answering, image captioning, document understanding, and describing what a camera sees in real time. VLMs are among the most practically impactful developments in modern AI.

## The Core Challenge: Connecting Vision and Language

Language models operate on discrete tokens. Images are continuous 2D grids of pixels. Joining these two modalities requires:
1. **A visual encoder** that extracts meaningful image representations
2. **A projection mechanism** that maps visual representations into the same space as text tokens
3. **A language model backbone** that reasons over the joint visual-text context

## CLIP: Contrastive Language-Image Pre-Training

**CLIP** (Radford et al., OpenAI 2021) is the foundational model for vision-language alignment. It trains two encoders — one for images, one for text — using **contrastive learning** on 400 million image-text pairs scraped from the web.

The training objective maximizes the cosine similarity between matching image-text pairs while minimizing it for non-matching pairs:

$$\mathcal{L}_\text{CLIP} = -\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(v_i, t_j)/\tau)}$$

where $v_i$ is the image embedding, $t_i$ is the text embedding, and $\tau$ is a learnable temperature.

**Key properties:**
- Zero-shot image classification: classify images by comparing to text descriptions of each class
- Rich semantic image embeddings that transfer to downstream tasks
- CLIP's image encoder (ViT variant) became the standard visual backbone for most VLMs that followed

## LLaVA: Large Language and Vision Assistant

**LLaVA** (Liu et al., 2023) is a pioneering open-source VLM that connects a CLIP visual encoder to a large language model (LLaMA/Vicuna) with a simple **linear projection layer**:

```
Image → CLIP ViT → Image Tokens (grid features)
                         ↓
               Linear Projection (W)
                         ↓
Text tokens + Image tokens → LLM → Response
```

Despite its architectural simplicity, LLaVA demonstrated that GPT-4-generated instruction-following data (describing images in conversation format) enables strong multimodal chat capabilities.

**LLaVA 1.5 and LLaVA-NeXT** replaced the linear projection with an **MLP connector** and a higher-resolution visual encoder, significantly improving fine-grained visual understanding.

## Architectural Patterns

### Connector Types
The "bridge" between vision encoder and LLM varies across models:

| Connector | Description | Example |
|---|---|---|
| Linear projection | Single linear layer | LLaVA 1.0 |
| MLP projection | 2-layer MLP | LLaVA 1.5 |
| Q-Former | Cross-attention query transformer | BLIP-2, InstructBLIP |
| Perceiver Resampler | Fixed number of visual tokens | Flamingo |
| Pixel shuffle + MLP | 2D to 1D with downsampling | InternVL |

### Visual Token Compression
High-resolution images produce thousands of visual tokens, bloating the context window. Compression strategies:
- **Average pooling:** Downsample feature maps before projection
- **Token merging (ToMe):** Merge similar adjacent tokens
- **Dynamic resolution:** Tile high-res images into sub-images; process each tile separately (LLaVA-NeXT, InternVL)

## GPT-4V, Claude 3, and Gemini

Frontier closed-source VLMs represent the state of the art:

- **GPT-4V / GPT-4o:** Multi-image input, fine-grained OCR, diagram reasoning, real-time video frames
- **Claude 3 (Anthropic):** Strong document understanding, chart interpretation, long document + image reasoning
- **Gemini 1.5 Pro:** Natively multimodal from pre-training; understands interleaved text and images across a 1M token context window; video understanding frame-by-frame

## Key Open-Source VLMs (2024–2025)

| Model | Developer | Highlights |
|---|---|---|
| LLaVA-NeXT (0.5B–72B) | LLaVA team | Strong open baseline |
| PaliGemma 2 | Google DeepMind | Small, highly transferable |
| Phi-3.5-Vision | Microsoft | 4.2B, efficient, on-device capable |
| InternVL 2.5 | Shanghai AI Lab | Top open-source benchmark scores |
| Qwen2-VL | Alibaba | Long context, dynamic resolution |
| MiniCPM-V | Tsinghua/ModelBest | Extremely lightweight (2B) |
| Idefics 3 | HuggingFace | Fully open data and weights |

## Tasks and Benchmarks

### Visual Question Answering (VQA)
The model answers questions about an image in natural language:
- **VQAv2:** "How many people are in this image?"
- **OK-VQA:** Questions requiring external knowledge
- **ScienceQA:** Multi-step scientific reasoning from diagrams

### Document and Chart Understanding
- **DocVQA:** Reading and answering questions from scanned documents
- **ChartQA:** Interpreting charts and infographics
- **TextVQA:** Reading and reasoning about text embedded in natural images

### Multimodal Reasoning
- **MMMU:** Massive Multidisciplinary Multimodal Understanding — graduate-level problems requiring domain knowledge
- **MathVista:** Mathematical reasoning from diagrams
- **SEED-Bench:** Comprehensive VLM evaluation across 19 tasks

### Grounding and Localization
Some VLMs can output bounding boxes or segmentation masks corresponding to objects described in text:
- **Grounding DINO:** Open-vocabulary object detection guided by natural language
- **SAM (Segment Anything Model):** Segment any object with a point, box, or text prompt

## Training Pipeline

1. **Pre-training (alignment stage):** Train only the connector on large-scale image-caption pairs; LLM and vision encoder are frozen. Goal: align visual feature space with language space.
2. **Instruction fine-tuning:** Unfreeze all parameters (or use LoRA); train on diverse visual instruction-following data. Goal: conversational and task-following capability.
3. **RLHF / preference fine-tuning (optional):** Align with human preferences on multimodal outputs.

## Applications

- **Visual chatbots:** Answer questions about uploaded images and documents
- **Industrial inspection:** Detect defects in manufacturing images with natural language queries
- **Medical imaging:** Assist radiologists with report generation from X-rays and MRIs
- **Accessibility:** Describe scenes and read text aloud for visually impaired users
- **Autonomous driving:** Caption and reason about road scenes
- **Retail:** Visual product search, try-on, catalog generation

## Limitations

- **Hallucination:** VLMs sometimes describe objects not present in the image, especially for uncommon scenarios
- **Counting accuracy:** Most VLMs struggle with precise object counting beyond ~5 items
- **Fine-grained spatial reasoning:** Left/right, above/below relationships are inconsistently handled
- **Long-video understanding:** Processing thousands of frames is computationally prohibitive
- **Compositional generalization:** Combining multiple visual concepts correctly remains difficult

## Further Reading

- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*
- Liu et al. (2023), *Visual Instruction Tuning (LLaVA)*
- Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning*
- Li et al. (2023), *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*
