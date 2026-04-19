---
title: Multimodal Embeddings
description: Understand multimodal embeddings — unified vector representations that encode text, images, audio, and video into a shared semantic space — enabling cross-modal search, retrieval, and alignment.
---

**Multimodal embeddings** are vector representations that encode different types of content — text, images, audio, video, structured data — into a **shared semantic space** where similar concepts are close together regardless of their modality. A query "a dog running on the beach" should retrieve both images of dogs on beaches and text descriptions of that scene.

## Why a Shared Embedding Space?

Single-modality embeddings encode text as vectors (e.g., sentence-transformers) or images as vectors (e.g., ViT features), but they cannot be compared across modalities. A text embedding of "sunset" and an image embedding of a sunset photo are in different vector spaces with different metrics — they cannot be directly compared.

Multimodal embeddings solve this by training a model to project different modalities into a **common latent space**:

$$f_\text{text}(t) \in \mathbb{R}^d \quad f_\text{image}(i) \in \mathbb{R}^d$$

Such that:

$$\text{cos}(f_\text{text}(t), f_\text{image}(i)) \approx 1 \quad \text{when } t \text{ describes } i$$

## Contrastive Learning for Alignment

The dominant training paradigm for multimodal embeddings is **contrastive learning** — training paired examples to have similar embeddings while pushing unpaired examples apart.

### CLIP (Contrastive Language-Image Pre-Training)

**CLIP** (Radford et al., 2021, OpenAI) is the foundational multimodal embedding model. It trains two encoders — one for text, one for images — using a contrastive loss on 400 million internet image-caption pairs.

**InfoNCE / Contrastive Loss:**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(t_i, v_j)/\tau)} + \log\frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(v_j, t_i)/\tau)}\right]$$

Where:

- $t_i$ is the text embedding of caption $i$.
- $v_i$ is the image embedding of image $i$.
- $\tau$ is a learned temperature.
- $N$ is the batch size.

The loss pulls paired (text, image) embeddings together and pushes unpaired embeddings apart within each batch.

**CLIP's capabilities:**

- Zero-shot image classification: Compute similarity between an image and text labels like "a photo of a cat."
- Cross-modal search: Query images with text and vice versa.
- Foundation for generative models: Used as the text conditioning backbone in DALL-E, Stable Diffusion, and others.

### SigLIP

**SigLIP** (Zhai et al., 2023, Google) replaces the softmax-normalized contrastive loss with a **sigmoid loss** applied independently to each pair:

$$\mathcal{L}_\text{sigmoid} = -\sum_{i,j} \left[y_{ij} \log \sigma(z_{ij}) + (1 - y_{ij}) \log(1 - \sigma(z_{ij}))\right]$$

Where $y_{ij} = 1$ if pair $(i, j)$ is matched. The sigmoid formulation enables better scaling to larger batch sizes and produces stronger embeddings at smaller model sizes.

## Audio-Text Embeddings

**CLAP (Contrastive Language-Audio Pre-training)** applies the same contrastive training approach to audio-text pairs:

- Audio encoder: Based on CNN or transformer audio models (HTSAT, PANN).
- Text encoder: Transformer-based text encoder.
- Trained on AudioCaps, AudioSet, and other captioned audio datasets.

Applications:

- Text-to-audio retrieval: "Find me the sound of rain on a metal roof."
- Zero-shot audio classification.
- Conditioning for audio generation models (like AudioLDM).

## Video Embeddings

Video presents the challenge of encoding both spatial and temporal information. Approaches include:

**Temporal pooling**: Average or pool frame-level image embeddings. Simple but loses motion information.

**3D convolutions / Video transformers**: Process spacetime patches jointly (e.g., VideoMAE, TimeSformer).

**CLIP4Clip / VideoCLIP**: Extend CLIP to video by encoding multiple frames and aggregating frame embeddings.

**Flamingo / VideoPaLM**: Large-scale video-language models that produce rich video-text embeddings.

## Unified Multimodal Embedding Models

Recent models embed **multiple modalities simultaneously** into a single shared space:

### ImageBind (Meta AI, 2023)

**ImageBind** learns a joint embedding space for six modalities simultaneously: **images, text, audio, depth, thermal, and IMU (inertial motion)**.

Key innovation: Images are used as the "binding" modality — since images co-occur with all other modalities in naturally occurring data (a dog photo co-occurs with audio of barking, text descriptions, depth maps, etc.), aligning all modalities to image space transitively aligns them to each other.

This enables **zero-shot cross-modal retrieval** between modalities that never appeared together in training — e.g., finding an image from an audio query, without ever training on (audio, image) pairs directly.

### Nomic Embed Multimodal

Nomic's multimodal embedding model supports text + image queries for unified vector search across mixed-content databases — useful for building RAG systems over documents containing both text and images.

### Voyage Multimodal

Voyage AI's multimodal embedding model is designed for production retrieval, producing embeddings optimized for semantic similarity across text and image content in document understanding workflows.

## Applications of Multimodal Embeddings

| Application | Description |
|---|---|
| **Cross-modal search** | Find images matching a text query or vice versa |
| **Multimodal RAG** | Retrieve relevant images, tables, charts from mixed documents |
| **Content moderation** | Detect harmful image-text combinations |
| **Product search** | Search product catalog by image or text |
| **Medical imaging** | Link clinical notes to relevant scans |
| **Recommendation** | Recommend videos, songs, or articles from multimodal preferences |
| **Zero-shot classification** | Classify images with text label embeddings, no training needed |

## Multimodal Embeddings in Vector Databases

Production systems store multimodal embeddings in vector databases and perform approximate nearest neighbor (ANN) search at query time:

```python
# Example: text query over image embeddings
query = "a busy intersection at night"
query_embedding = clip_model.encode_text(query)

results = vector_db.search(
    collection="product_images",
    vector=query_embedding,
    limit=10
)
```

The same embedding model must be used for both indexing and querying to ensure vectors occupy the same semantic space.

## Evaluation

Multimodal embedding quality is measured on:

- **Recall@K (R@K)**: For each query, does the correct match appear in the top-K retrieved results?
- **Mean Rank**: Average rank of the correct match across all queries.
- **Zero-shot classification accuracy**: Accuracy on image classification using text label embeddings.
- **Linear probing**: Classification accuracy of a linear classifier trained on top of frozen embeddings.

## Further Reading

- [Learning Transferable Visual Models From Natural Language Supervision (CLIP) — Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- [Sigmoid Loss for Language Image Pre-Training (SigLIP) — Zhai et al., 2023](https://arxiv.org/abs/2303.15343)
- [ImageBind: One Embedding Space To Bind Them All — Girdhar et al., 2023](https://arxiv.org/abs/2305.05665)
- [CLAP: Learning Audio Concepts From Natural Language Supervision — Elizalde et al., 2022](https://arxiv.org/abs/2206.04769)
