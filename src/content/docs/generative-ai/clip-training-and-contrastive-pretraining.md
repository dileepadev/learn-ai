---
title: CLIP and Contrastive Image-Text Pretraining
description: Understand CLIP and its successors — ALIGN, SigLIP, and OpenCLIP — including how contrastive pretraining on image-text pairs produces shared embedding spaces, the InfoNCE loss, zero-shot image classification, cross-modal retrieval, and why CLIP embeddings became the backbone of modern generative and multimodal AI systems.
---

Before CLIP, training a visual model for a new domain meant collecting thousands of labeled images and fine-tuning from ImageNet weights. **CLIP** (Contrastive Language-Image Pretraining, Radford et al., OpenAI 2021) changed this by learning from 400 million image-text pairs scraped from the internet — no manual labels required. The result was a model that could classify images into any category described in natural language, perform cross-modal search, and produce visual embeddings powerful enough to anchor an entire generation of generative AI systems.

## The Core Idea

CLIP trains two encoders jointly:

- An **image encoder** $f_I$ (ResNet or Vision Transformer) that maps images to embedding vectors
- A **text encoder** $f_T$ (Transformer) that maps natural language descriptions to embedding vectors

Training objective: given a batch of $N$ (image, text) pairs, maximize the cosine similarity between the $N$ correct image-text pairs while minimizing similarity for the $N^2 - N$ incorrect pairings.

```text
Batch of N image-text pairs:

  (img_1, txt_1)  ← correct pair: high similarity
  (img_1, txt_2)  ← incorrect pair: low similarity
  (img_1, txt_3)  ← incorrect pair: low similarity
  ...
  (img_N, txt_N)  ← correct pair: high similarity

Loss: maximize diagonal of N×N similarity matrix
```

## The InfoNCE Loss

CLIP uses a symmetric cross-entropy loss over the cosine similarity matrix. For a batch of $N$ pairs, let $s_{ij} = \cos(f_I(\mathbf{x}_i), f_T(\mathbf{t}_j)) / \tau$ where $\tau$ is a learned temperature:

$$\mathcal{L} = -\frac{1}{2N} \left[ \sum_{i=1}^N \log \frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} + \sum_{j=1}^N \log \frac{e^{s_{jj}}}{\sum_i e^{s_{ij}}} \right]$$

The first sum treats each image as a query and texts as keys (image-to-text retrieval). The second treats each text as a query and images as keys (text-to-image retrieval). Together this is the **InfoNCE** loss — a lower bound on mutual information between image and text representations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, initial_temperature: float = 0.07):
        super().__init__()
        # Learned log temperature (clamped to prevent collapse)
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    def forward(
        self,
        image_embeddings: torch.Tensor,   # (B, d), L2-normalized
        text_embeddings: torch.Tensor,    # (B, d), L2-normalized
    ) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(max=100.0)

        # Cosine similarity matrix scaled by temperature
        logits = torch.mm(image_embeddings, text_embeddings.t()) * temperature  # (B, B)

        # Symmetric cross-entropy: diagonal = correct pairs
        labels = torch.arange(len(logits), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)        # image → text
        loss_t2i = F.cross_entropy(logits.t(), labels)    # text → image

        return (loss_i2t + loss_t2i) / 2
```

## Architecture

### Image Encoder

CLIP was trained with both ResNet and Vision Transformer image encoders:

- **ResNet variants**: ResNet-50 to ResNet-101, with modified pooling (attention pooling at the final layer instead of global average)
- **ViT variants**: ViT-B/32, ViT-B/16, ViT-L/14 — the ViT-L/14 model at 336px resolution (ViT-L/14@336) is the strongest CLIP image encoder

### Text Encoder

The text encoder is a 12-layer Transformer (63M parameters) with:

- Byte pair encoding (BPE) tokenization, vocabulary size 49,408
- Max context length: 77 tokens
- The `[EOS]` token representation is used as the text embedding (not CLS pooling)

## Zero-Shot Image Classification

CLIP's most striking capability is **zero-shot classification**: classifying images into arbitrary categories described in natural language, without any labeled training data for those categories.

```python
import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Any image
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# Any categories — zero-shot: never seen during fine-tuning
categories = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of a car",
]
text_tokens = clip.tokenize(categories).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    # Normalize and compute cosine similarities
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    similarities = (image_features @ text_features.t()).squeeze(0)
    probs = similarities.softmax(dim=-1)

for category, prob in zip(categories, probs):
    print(f"{category}: {prob:.1%}")
```

The key insight: for a new category, just write a text description. No retraining needed. On ImageNet, CLIP ViT-L/14 achieves ~76% zero-shot top-1 accuracy — matching supervised ResNet-50 performance without any ImageNet training.

## Prompt Engineering for CLIP

Classification performance is highly sensitive to how categories are described. Prompt engineering improves zero-shot accuracy:

```python
# Naive: poor performance
categories = ["dog", "cat", "airplane"]

# Engineered: much better
templates = [
    "a photo of a {}",
    "a photo of the {}",
    "a photograph of a {}",
    "an image of a {}",
    "a picture of a {}",
]

# Ensemble text embeddings over multiple templates
def get_zeroshot_weights(classnames, templates, model):
    weights = []
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
            embs = F.normalize(embs, dim=-1)
        weights.append(embs.mean(dim=0))
    return F.normalize(torch.stack(weights), dim=-1)
```

OpenAI found that "a photo of a {classname}" consistently outperforms bare class names, and that ensembling over 80 diverse templates further improves accuracy by ~3.5% on ImageNet.

## ALIGN and SigLIP

### ALIGN (Google, 2021)

ALIGN (A Large-scale ImaGe and Noisy-text embedding) scales CLIP's approach to 1.8 billion noisy image-text pairs with minimal filtering. Key findings:

- Scale compensates for noise: larger datasets with noisy labels outperform smaller clean datasets
- EfficientNet image encoder + BERT text encoder
- Achieves better performance than CLIP on several retrieval benchmarks

### SigLIP (Google, 2023)

SigLIP replaces the softmax contrastive loss with **sigmoid loss**. Instead of normalizing across all negatives in a batch, each pair independently predicts a binary label (1 for correct pair, -1 for incorrect):

$$\mathcal{L}_{\text{sigmoid}} = -\frac{1}{N^2} \sum_{i,j} \log \sigma\!\left(z_{ij} \cdot (2y_{ij} - 1)\right)$$

where $z_{ij} = t \cdot \cos(f_I(\mathbf{x}_i), f_T(\mathbf{t}_j)) + b$ and $y_{ij} = 1$ if $i = j$.

Benefits of sigmoid loss over softmax:

- No dependency on batch size for normalization — works well at any batch size
- Allows multiple correct pairings per image (useful for multi-caption datasets)
- Empirically better zero-shot performance at the same compute budget

## OpenCLIP

OpenCLIP (LAION-AI) is an open-source reproduction of CLIP trained on LAION-400M and LAION-5B datasets. Key contributions:

- Fully reproducible training code and pre-trained weights at multiple scales
- ViT-bigG/14 trained on LAION-2B achieves 80.1% zero-shot ImageNet accuracy — surpassing original CLIP
- Enables community research into CLIP training dynamics, data curation effects, and scaling laws

## CLIP as a Backbone

CLIP embeddings became foundational building blocks for downstream systems:

| System | CLIP Role |
| --- | --- |
| DALL-E 2 | CLIP image embeddings used as prior for image generation |
| Stable Diffusion | CLIP text encoder provides text conditioning to the UNet |
| ControlNet | Inherits Stable Diffusion's CLIP text encoder |
| LLaVA, BLIP-2 | CLIP image encoder provides visual tokens to language models |
| CLIP-as-service | Image/text search and deduplication at scale |
| Flamingo | CLIP image encoder with cross-attention into LLM |

The reason CLIP embeddings transfer so well: the contrastive objective forces the model to discard image-specific details not describable in language, producing representations aligned with human semantic concepts.

## Fine-Tuning CLIP

For domain-specific applications (medical imaging, satellite imagery, industrial inspection), CLIP can be fine-tuned:

```python
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Fine-tune only the projection layers (parameter-efficient)
for name, param in model.named_parameters():
    if "projection" not in name:
        param.requires_grad_(False)

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=0.01,
)

# Training loop
for batch in domain_dataloader:
    inputs = processor(
        text=batch["captions"],
        images=batch["images"],
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs, return_loss=True)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Summary

CLIP and its successors established contrastive image-text pretraining as a foundational paradigm in multimodal AI:

- **InfoNCE loss** drives image and text encoders to agree on shared semantic content across 400M+ web-scale pairs
- **Zero-shot classification** emerges naturally: classify by computing similarity to text descriptions of any category
- **Prompt engineering** and template ensembling significantly improve zero-shot accuracy without any labeled data
- **SigLIP's sigmoid loss** removes batch-size dependence and handles multi-caption inputs more naturally
- **CLIP as backbone** powers the text conditioning in Stable Diffusion, the visual processing in LLaVA, and retrieval in image search systems worldwide

The insight that language supervision from web text is sufficient to learn highly transferable visual representations fundamentally changed how visual AI is built.
