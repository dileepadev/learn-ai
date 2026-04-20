---
title: Vision-Language Models (VLMs)
description: How AI systems learn to connect vision and language — covering the architectures behind CLIP, ALIGN, BLIP, LLaVA, GPT-4V, Gemini, and InternVL, along with training strategies, benchmarks, and real-world applications of vision-language understanding.
---

**Vision-Language Models (VLMs)** are AI systems that understand and reason jointly over images and text. They power applications ranging from image captioning and visual question answering to document understanding and robotic perception. The field has undergone a revolution — from task-specific pipelines to large multimodal foundation models capable of zero-shot transfer across diverse visual-linguistic tasks.

## The Core Challenge: Aligning Visual and Linguistic Representations

Text and images live in fundamentally different representational spaces. Words are discrete tokens processed through embedding tables; images are continuous pixel grids processed through convolutional or patch-based networks. Bridging these modalities requires learning a **shared semantic space** where related concepts in both modalities are represented similarly.

## Contrastive VLMs: CLIP and ALIGN

### CLIP (OpenAI, 2021)

**Contrastive Language-Image Pre-Training (CLIP)** is arguably the most influential VLM. It trains two encoders — an image encoder and a text encoder — to produce embeddings that are **aligned across modalities** using a contrastive objective:

- Given a batch of $N$ (image, text) pairs, CLIP maximizes the cosine similarity of the $N$ correct pairs and minimizes the similarity of the $N^2 - N$ incorrect pairings.
- The model was trained on **400 million** image-text pairs scraped from the internet.

**Architecture:**

- Image encoder: ViT (Vision Transformer) or ResNet.
- Text encoder: Transformer (similar to GPT-2).
- Both encoders project to a shared embedding space of dimension 512 or 768.

**Zero-shot classification:** CLIP performs image classification without task-specific training by computing similarity between an image embedding and text embeddings for candidate class names ("a photo of a dog", "a photo of a cat"). It pioneered **zero-shot visual recognition** at scale.

### ALIGN (Google, 2021)

ALIGN (A Large-scale ImaGe and Noisy-text Embedding) used an even noisier but larger dataset (1.8 billion image-text pairs) with a similar contrastive approach. It demonstrated that scale can compensate for data noise, matching or exceeding CLIP on downstream benchmarks.

### Limitations of Contrastive VLMs

Contrastive models learn **global alignment** between an image and its caption, but struggle with:

- Fine-grained spatial reasoning ("the red cube to the left of the blue sphere").
- Counting objects precisely.
- Compositional understanding (order and relationships matter, not just concept presence).

## Generative VLMs: BLIP and BLIP-2

### BLIP (Salesforce, 2022)

**Bootstrapping Language-Image Pre-training (BLIP)** combined contrastive learning with generative objectives:

1. **ITC** (Image-Text Contrastive): CLIP-style alignment.
2. **ITM** (Image-Text Matching): Binary classifier determining if an image-text pair matches.
3. **LM** (Language Modeling): Generate text given image as context.

BLIP also introduced **CapFilt** — a bootstrapping technique that generates and filters synthetic captions to improve noisy web data quality.

### BLIP-2 (2023)

BLIP-2 introduced the **Q-Former (Querying Transformer)** — a lightweight bridge module between a frozen image encoder and a frozen LLM. The Q-Former uses a small set of learned query vectors that attend to image features and produce a compact visual representation consumed by the LLM.

**Key insight**: Freeze both the vision encoder and the LLM; train only the connector. This dramatically reduces training compute while leveraging large pretrained components.

## Large Multimodal Models (LMMs)

The latest generation of VLMs extends large language models with visual input, enabling **conversational visual understanding**.

### LLaVA (Large Language and Vision Assistant)

**LLaVA** (Liu et al., 2023) connected a CLIP vision encoder to LLaMA using a simple linear projection layer. Despite the simplicity, it demonstrated that:

- High-quality **instruction-following data** matters more than architecture complexity.
- GPT-4 can generate visual instruction tuning data from image captions, enabling scalable dataset creation without human annotation.

**LLaVA-1.5** replaced the linear projection with an MLP and used higher-resolution images, significantly improving performance on benchmarks like ScienceQA and TextVQA.

**LLaVA-NeXT (LLaVA-1.6)** introduced dynamic high-resolution image tiling — splitting images into sub-tiles processed independently and concatenated — handling high-resolution documents and dense visual scenes.

### GPT-4V and GPT-4o

**GPT-4V** (OpenAI, 2023) is a proprietary multimodal extension of GPT-4 that accepts interleaved image and text inputs. It demonstrated unprecedented visual reasoning: reading charts, describing complex scenes, solving visual math problems, and understanding humor in images.

**GPT-4o** (2024) extended this to real-time audio, image, and text together, with a natively multimodal architecture (rather than a patched vision adapter) and significantly improved instruction following on visual tasks.

### Gemini

Google's **Gemini** family was designed from the ground up as natively multimodal — trained jointly on text, images, audio, video, and code from the start, rather than bolting vision onto a language model. Gemini 1.5 Pro demonstrated the ability to process entire videos (up to 1 hour) and extract fine-grained information, enabled by its 1M+ token context window.

### InternVL

**InternVL** (Shanghai AI Lab) is a series of open-source VLMs that progressively scale both the vision encoder and language model together. InternVL 2.5 rivals GPT-4V on several benchmarks and supports documents, charts, video frames, and multi-image reasoning.

## Architecture Patterns

Modern VLMs share a common structural pattern:

```text
Image → [Vision Encoder] → Visual Tokens
                               ↓
Text → [Tokenizer] → Text Tokens
                               ↓
             [Connector / Projector]
                               ↓
                   [Language Model]
                               ↓
                         Output Text
```

The main architectural choices are:

| Component | Options |
| --- | --- |
| Vision encoder | CLIP ViT, SigLIP, DINOv2, SAM encoder |
| Connector | Linear projection, MLP, Q-Former, cross-attention |
| Language model | LLaMA, Mistral, Qwen, Gemma, Phi |
| Training strategy | Frozen encoder + LLM, joint fine-tuning, full pretraining |

## Benchmarks

| Benchmark | What It Tests |
| --- | --- |
| VQA v2 | Open-ended visual question answering |
| TextVQA | Reading text in images |
| ChartQA | Understanding charts and plots |
| DocVQA | Document understanding (forms, PDFs) |
| MMMU | Multi-discipline college-level visual reasoning |
| ScienceQA | Science questions with diagrams |
| MMBench | Broad multimodal capability suite |

## Video and Multi-Image Understanding

VLMs are increasingly extended to video by treating frames as a sequence of visual tokens. Key challenges:

- **Temporal reasoning**: Understanding cause and effect across frames.
- **Context length**: A 1-minute video at 1 fps with 256 tokens per frame = 15,360 visual tokens.
- **Efficient frame sampling**: Not all frames are equally informative; intelligent sampling reduces token count.

Models like **Video-LLaMA**, **Video-ChatGPT**, and **Gemini 1.5** tackle video understanding at increasing context lengths.

## Applications

- **Document AI**: Extracting information from invoices, receipts, contracts, and scientific papers.
- **Medical imaging**: Describing X-rays, pathology slides, and MRI scans in natural language.
- **E-commerce**: Visual search, product description generation, attribute extraction.
- **Accessibility**: Screen readers, image descriptions for visually impaired users.
- **Robotics**: Grounding natural language instructions in visual scene understanding.
- **Code from UI**: Generating code from screenshots of user interfaces.

## Open Challenges

- **Hallucination**: VLMs can describe objects not present in the image — the visual equivalent of LLM hallucination.
- **Fine-grained spatial reasoning**: Precise spatial relationships remain difficult.
- **Long document understanding**: Processing multi-page PDFs with dense text and figures.
- **3D spatial understanding**: Most VLMs reason about 2D image space, not 3D world geometry.
- **Compositional generalization**: Understanding novel combinations of attributes and relationships.

Vision-language models have become foundational infrastructure for AI systems that interact with the real world, driving rapid convergence between natural language processing and computer vision.
