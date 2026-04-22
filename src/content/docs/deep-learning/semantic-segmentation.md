---
title: Semantic Segmentation
description: A deep dive into semantic and instance segmentation — covering FCN, U-Net, DeepLab, Mask R-CNN, and transformer-based approaches like Segment Anything Model (SAM), with key metrics and real-world applications.
---

**Semantic segmentation** is the computer vision task of assigning a class label to **every pixel** in an image. Unlike object detection, which draws bounding boxes around objects, semantic segmentation produces a dense pixel-wise classification map that delineates the exact shape and boundaries of each region. A street scene, for example, would be partitioned into pixels belonging to road, sky, pedestrian, vehicle, building, and vegetation classes.

Segmentation sits at the intersection of perception and scene understanding, and is foundational to autonomous driving, medical image analysis, satellite imagery interpretation, and augmented reality.

## Types of Segmentation

The segmentation landscape includes several related but distinct tasks:

**Semantic segmentation**: Every pixel is assigned a class label. All instances of a class share the same label — two pedestrians are both labeled "person," with no distinction between them.

**Instance segmentation**: Extends semantic segmentation to distinguish between individual instances of the same class. Each pedestrian gets a unique instance ID in addition to the class label.

**Panoptic segmentation**: Unifies semantic and instance segmentation. "Things" (countable objects like cars and people) are segmented at the instance level; "stuff" (amorphous regions like sky and road) is segmented at the semantic level.

**Interactive segmentation**: A user provides prompts (points, boxes, or scribbles) to specify which region to segment. The Segment Anything Model (SAM) is the leading example.

## Fully Convolutional Networks (FCN)

**FCN** (Long et al., 2015) established the modern paradigm for semantic segmentation by replacing the fully connected layers in a classification CNN (VGG, AlexNet) with convolutional layers, enabling dense predictions for inputs of any size.

The key insight: a classification CNN's convolutional layers produce a spatially downsampled feature map where each location encodes information about a local region of the input. If we decode this feature map back to the input resolution, we can make per-pixel predictions.

**Upsampling** the downsampled feature map back to input resolution is achieved through:

- **Bilinear interpolation**: Fast but blurry at boundaries.
- **Transposed convolution (deconvolution)**: Learned upsampling that can recover fine details.
- **Skip connections**: FCN-32s, FCN-16s, FCN-8s progressively added skip connections from earlier (higher-resolution) layers, combining coarse semantic information from deep layers with fine spatial information from shallow layers.

## U-Net

**U-Net** (Ronneberger et al., 2015) — originally designed for biomedical image segmentation — introduced the **encoder-decoder with skip connections** architecture that became the dominant pattern for segmentation.

### Architecture

```
Input
  │
  ▼
Encoder (contracting path)
  │ conv + conv + maxpool (×4)
  ▼
Bottleneck
  │
  ▼
Decoder (expanding path)
  │ transposed conv + concat(skip) + conv + conv (×4)
  ▼
Output segmentation map
```

**Skip connections** concatenate feature maps from each encoder level to the corresponding decoder level, allowing the decoder to access both:

- **Semantic context** from the bottleneck (what is in the image).
- **Spatial detail** from the encoder (exactly where it is).

U-Net's symmetric structure and skip connections make it exceptionally effective for medical imaging, where precise boundary delineation is critical. Modern variants (U-Net++, Attention U-Net, Swin U-Net) have extended the original with nested decoders, attention gates, and transformer encoders.

## DeepLab Series

Google's **DeepLab** series focused on two challenges in semantic segmentation:

1. **Loss of spatial resolution**: Standard CNN pooling and strided convolutions progressively reduce feature map resolution, losing boundary detail.
2. **Multi-scale context**: Objects appear at different scales; a fixed receptive field captures context at only one scale.

### Dilated (Atrous) Convolution

**Dilated convolution** inserts gaps (dilation rate $r$) between kernel elements, exponentially increasing the receptive field without reducing spatial resolution or adding parameters:

A standard $3 \times 3$ conv has receptive field $3 \times 3$. With dilation $r = 2$, the effective receptive field is $5 \times 5$ (covering the same area as a $5 \times 5$ conv but with only $3 \times 3$ parameters). With $r = 4$, it becomes $9 \times 9$.

DeepLab replaces the last few pooling layers with dilated convolutions, maintaining high-resolution feature maps while preserving large receptive fields.

### ASPP (Atrous Spatial Pyramid Pooling)

**DeepLabv3** introduced ASPP — applying dilated convolutions at multiple rates in parallel and combining their outputs. This captures multi-scale context with a single module:

- Dilation rates: 6, 12, 18 (for stride 16 features).
- Also includes global average pooling to capture image-level context.

**DeepLabv3+** added a simple decoder with skip connections from the encoder, further recovering boundary sharpness — combining the benefits of the DeepLab dilated convolution approach with the U-Net decoder approach.

## Instance Segmentation: Mask R-CNN

**Mask R-CNN** (He et al., 2017) extends Faster R-CNN for instance segmentation by adding a **mask prediction head** to the existing classification and box regression heads.

### Architecture

For each RoI (region of interest):

- **Classification head**: Predicts the object class.
- **Box regression head**: Refines the bounding box.
- **Mask head**: Predicts a binary segmentation mask ($28 \times 28$ pixels) for the object within the RoI.

The mask head is a small FCN applied to each RoI independently, predicting a binary mask for each class (the mask of the predicted class is used at inference). This design decouples mask and class prediction, enabling better segmentation.

**RoIAlign** (vs. RoIPool): Mask R-CNN introduced RoIAlign, which uses bilinear interpolation instead of quantization when extracting RoI features. This eliminates the spatial misalignment introduced by rounding in RoIPool — critical for the pixel-level precision required by mask prediction.

## Panoptic Segmentation

**Panoptic FPN** (Kirillov et al., 2019) extends Mask R-CNN with a semantic segmentation branch that uses the FPN feature pyramid to produce a dense semantic map. The instance predictions from Mask R-CNN and the semantic map are fused into a unified panoptic output.

**Panoptic-DeepLab** takes a bottom-up approach: predicting semantic labels and object center heatmaps, then grouping pixels to centers to produce instance labels — enabling real-time panoptic segmentation without a region proposal step.

## Transformer-Based Segmentation

Transformers have transformed segmentation as they have detection, enabling better global context modeling.

### SETR (Segmentation Transformer)

**SETR** replaces the CNN encoder in a segmentation model with a **Vision Transformer (ViT)**, treating image patches as sequence tokens. The ViT encoder captures long-range dependencies across the full image — addressing a limitation of CNNs which only capture local context within the receptive field. A progressive upsampling decoder produces the final segmentation map.

### Segmenter

**Segmenter** uses a ViT encoder and a **transformer decoder** that attends to class-specific query tokens to produce class masks — a cleaner design than SETR.

### MaskFormer and Mask2Former

**MaskFormer** (Cheng et al., 2021) reframes segmentation as a **mask classification** problem: predict a set of binary masks and a class for each, then combine. This unified formulation handles both semantic and instance segmentation.

**Mask2Former** adds **masked attention** — restricting attention to predicted mask regions rather than the full image — improving efficiency and enabling state-of-the-art performance on semantic, instance, and panoptic segmentation benchmarks.

## Segment Anything Model (SAM)

**SAM** (Kirillov et al., Meta AI, 2023) represents a paradigm shift toward a **promptable, foundation model for segmentation**.

### Architecture

- **Image encoder**: A heavyweight ViT that processes the image once and produces embeddings (computationally expensive but done once per image).
- **Prompt encoder**: Encodes prompts — points, boxes, free-form text, or rough masks — into prompt embeddings.
- **Mask decoder**: A lightweight transformer decoder that combines image and prompt embeddings to predict one or three masks and associated confidence scores.

### Prompting Modes

- **Point prompt**: Click on a point; SAM segments the object at that point.
- **Box prompt**: Draw a bounding box; SAM produces a precise mask within the box.
- **Mask prompt**: Provide a rough initial mask; SAM refines it.
- **Text prompt**: (SAM 2 and future extensions) Natural language description of what to segment.

### Training Data

SAM was trained on **SA-1B** — a dataset of 1 billion masks across 11 million images, partially generated by SAM itself in a model-in-the-loop annotation process. The scale of training data is a key factor in SAM's generalization.

**SAM 2** (2024) extends SAM to **video segmentation**, tracking and segmenting objects across frames in real time with streaming memory.

## Evaluation Metrics

### Mean Intersection over Union (mIoU)

The standard metric for semantic segmentation. For each class $c$:

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

mIoU averages IoU across all classes. Classes with zero ground-truth pixels in the evaluation set are excluded.

### Pixel Accuracy

The fraction of correctly classified pixels:

$$\text{PA} = \frac{\sum_c TP_c}{\text{Total pixels}}$$

Less informative than mIoU because it is dominated by large background classes.

### Boundary F1 Score (BF)

Evaluates how accurately the predicted segmentation boundary aligns with the ground-truth boundary — important for applications requiring precise delineation.

### Standard Benchmarks

| Benchmark | Domain | Classes | Notes |
|-----------|--------|---------|-------|
| **Cityscapes** | Urban driving | 19 | High-res, fine + coarse annotations |
| **ADE20K** | Indoor + outdoor scenes | 150 | Diverse, challenging long tail |
| **PASCAL VOC** | General | 21 | Historic benchmark |
| **COCO Panoptic** | General | 133 | 80 things + 53 stuff |
| **Medical Decathlon** | Medical imaging | Task-specific | Multi-organ CT/MRI segmentation |

## Real-World Applications

**Autonomous driving**: Road, lane, drivable area, pedestrian, vehicle, and obstacle segmentation for path planning and safety systems. Cityscapes and BDD100K benchmarks directly target this domain.

**Medical imaging**: Organ and lesion segmentation in MRI, CT, and ultrasound scans. U-Net and its variants dominate this domain, where annotated data is scarce and precision is critical.

**Satellite and aerial imagery**: Land use classification, forest cover mapping, urban change detection, and disaster assessment from satellite images.

**Augmented reality**: Segmenting foreground objects for background replacement, scene understanding for virtual object placement, and portrait mode effects in smartphone cameras.

**Industrial quality control**: Pixel-precise defect detection on manufactured surfaces, identifying micro-cracks, scratches, or contamination that bounding boxes would not capture adequately.

**Agriculture**: Crop health assessment from drone imagery, weed detection, and yield estimation through field segmentation.

## Challenges in Segmentation

**Class imbalance**: Background pixels vastly outnumber foreground pixels in most datasets. Strategies include weighted loss, focal loss, and careful sampling.

**Boundary ambiguity**: Precisely annotating object boundaries is difficult and subjective, leading to noisy labels at edges that constrain maximum achievable accuracy.

**Small objects**: Very small objects may be represented by only a handful of pixels, making them difficult to segment accurately. High-resolution processing and multi-scale architectures help.

**Domain shift**: Segmentation models trained on one domain (e.g., sunny daytime driving) often degrade significantly in another (nighttime, rain, fog). Domain adaptation and test-time augmentation are active areas of research.

**Real-time requirements**: Applications like autonomous driving require inference at 30+ fps. Efficient architectures (EfficientDet, BiSeNet, DDRNet) target the accuracy-latency trade-off on edge hardware.
