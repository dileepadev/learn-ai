---
title: Promptable Visual Segmentation and Zero-Shot Masks
description: Understand promptable visual segmentation from points, boxes, and free-form text queries, exploring interactive mask generation and open-vocabulary spatial perception.
---

In classical computer vision, segmentation was divided into isolated, rigid sub-disciplines: **semantic segmentation** (classifying every pixel into predefined categories), **instance segmentation** (delineating separate instances of known categories), and **panoptic segmentation** (unifying both). Each required training specialized models on painstakingly annotated datasets.

In 2023, Meta AI fundamentally redefined the field by introducing the **Promptable Segmentation Paradigm** with the **Segment Anything Model (SAM)** (Kirillov et al.). Modeled after foundation language models that accept natural language prompts to perform arbitrary text tasks, SAM accepts **visual or spatial prompts** to return high-quality segmentation masks for *any* object in zero-shot fashion.

---

## The Promptable Segmentation Task

A promptable segmentation model takes an image and a **flexible user prompt**, and outputs a valid segmentation mask corresponding to the intent of the prompt:

```
                            [ High-Resolution Image ]
                                       │
                                       ▼
                             [ Heavy Image Encoder ]
                             (Standard ViT Backbone)
                                       │
                             Precomputed Image Embedding
                                       │
[ User Prompt ]                        │
• Foreground/Background Points         │
• Bounding Box Coordinates  ───────────┼──────────► [ Lightweight Mask Decoder ] ──► Predicted Masks
• Free-Form Text Query                 │            (Executes in ~5ms in browser!)    (Resolves Ambiguities)
• Coarse Mask Prior                    │
```

### Supported Prompt Types
1. **Foreground / Background Points:** One or more clicked $(x, y)$ coordinates indicating what to include ($+1$) or exclude ($-1$).
2. **Bounding Boxes:** A rectangle $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ specifying the region of interest.
3. **Rough Masks:** Low-resolution binary sketches or heatmaps indicating the approximate spatial extent.
4. **Free-Form Text (Open-Vocabulary):** Natural language descriptions like *"the red coffee mug on the wooden table"*.

---

## Architectural Mechanics of SAM

To enable smooth interactive user experiences (such as running live in a web browser), SAM decouples computation into two asymmetric components:

### 1. Heavy Image Encoder (Compute Once)
A standard Vision Transformer (ViT-H or ViT-B) with MAE (Masked Autoencoder) pretraining processes the input image ($1024 \times 1024$ pixels) once into a $64 \times 64 \times 256$ spatial embedding tensor. This operation takes $\sim 50\text{--}100\text{ ms}$ on a GPU.

### 2. Prompt Encoder & Lightweight Mask Decoder (Compute Interactively)
Once the image embedding is computed, interactive user prompts are parsed by a lightweight decoder in under **$5\text{ ms}$**:
- **Sparse Prompts (Points, Boxes):** Points and bounding boxes are projected using positional encodings (Fourier features) summed with learned embeddings indicating foreground/background or top-left/bottom-right corners.
- **Dense Prompts (Masks):** Embedded using convolutional layers and added element-wise directly to the image embedding.
- **Two-Way Cross-Attention Decoder:** Computes bidirectional cross-attention between the prompt tokens and the image embedding, followed by an MLP that computes a dot-product with upscaled feature maps to produce pixel masks.

---

## Resolving Semantic Ambiguity: Multimask Output

Visual prompts are frequently ambiguous. If a user clicks on the wheel of a bicycle, did they mean:
1. Just the tire? (subpart)
2. The entire wheel assembly? (part)
3. The whole bicycle? (whole object)

```
Ambiguous User Click: [•] on the tire

Multimask Decoder Outputs 3 Plausible Hypotheses:
Mask 1 (Subpart): [ Rubber Tire Only ]       ──► Predicted IoU: 0.94
Mask 2 (Part):    [ Entire Wheel + Spokes ]   ──► Predicted IoU: 0.96
Mask 3 (Whole):   [ Complete Bicycle Frame ] ──► Predicted IoU: 0.91
```

SAM handles this by outputting **3 candidate masks simultaneously** along with an automated **IoU Confidence Score** for each mask. In interactive settings, the user can provide a second click to instantly resolve the ambiguity.

---

## Open-Vocabulary Segmentation: Grounded-SAM

While SAM's original text encoder had limited capability, the open-source community combined SAM with **Grounding DINO** to create **Grounded-SAM**:

```
Text: "Segment the yellow taxi and the pedestrian in the striped shirt"
                             │
                             ▼
               [ Stage 1: Grounding DINO ]
         (Detects open-vocabulary bounding boxes)
                             │
            Bounding Box 1: [Taxi Coordinates]
            Bounding Box 2: [Pedestrian Coordinates]
                             │
                             ▼
                 [ Stage 2: Segment Anything ]
         (Consumes bounding boxes as spatial prompts)
                             │
                             ▼
        Pixel-Perfect Instance Masks for Described Concepts
```

---

## Practical Python Example: Interactive Box Prompting

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Load SAM checkpoint and initialize predictor
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# 1. Set image (computes heavy image embedding once)
image_rgb = np.array(...) # Load 1024x1024 image
predictor.set_image(image_rgb)

# 2. Provide prompt coordinates: [x_min, y_min, x_max, y_max]
input_box = np.array([120, 85, 340, 410])

# 3. Predict mask (executes in milliseconds)
masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)

best_mask = masks[0] # Binary boolean mask of shape (H, W)
print(f"Segmented mask with confidence score: {scores[0]:.4f}")
```

---

## Key Takeaways

- Promptable segmentation transforms mask prediction into a generalized foundation task controlled by spatial or language prompts.
- Decoupling heavy ViT image encoding from lightweight two-way cross-attention decoders enables real-time interactive performance.
- Integrating open-vocabulary detectors (Grounding DINO) with promptable segmenters (SAM) delivers fully autonomous, open-vocabulary panoptic perception.
