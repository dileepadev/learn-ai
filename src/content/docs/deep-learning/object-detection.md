---
title: Object Detection
description: A comprehensive guide to object detection — covering two-stage detectors (Faster R-CNN), single-stage detectors (YOLO, SSD, RetinaNet), transformer-based approaches (DETR), and key evaluation metrics like mAP.
---

**Object detection** is the computer vision task of simultaneously identifying *what* objects are present in an image and *where* they are located, expressed as bounding boxes with associated class labels and confidence scores. Unlike image classification, which assigns a single label to an entire image, object detection must handle a variable number of objects of multiple classes, potentially overlapping, at different scales.

Object detection is foundational to applications ranging from autonomous driving and medical imaging to retail analytics and security systems.

## The Object Detection Problem

Given an input image, an object detector must output:

- A set of **bounding boxes**, typically parameterized as $(x, y, w, h)$ (center coordinates, width, height) or $(x_1, y_1, x_2, y_2)$ (top-left and bottom-right corners).
- A **class label** for each bounding box.
- A **confidence score** indicating the model's certainty that the box contains an object of that class.

The challenge: the number and location of objects in an image is unknown at inference time. Unlike classification (one output per image), detection requires predicting a variable-length set of predictions.

## Anchor Boxes

A foundational concept in object detection is the **anchor box** — a set of predefined bounding boxes of fixed sizes and aspect ratios tiled across the image. The detector predicts *offsets* from these anchors rather than absolute box coordinates, making the regression problem easier.

Anchors are defined by:

- **Scale**: Multiple sizes to detect objects of different sizes (small, medium, large).
- **Aspect ratio**: Multiple ratios (e.g., 1:1, 1:2, 2:1) to handle objects with different proportions.

At each spatial location in the feature map, $k$ anchors are placed (where $k$ is the number of scale/aspect ratio combinations). The detector predicts, for each anchor: (1) whether an object is present, and (2) the regression offsets to transform the anchor into a tight bounding box.

## Two-Stage Detectors

**Two-stage detectors** follow a sequential pipeline:

1. **Stage 1 — Region Proposal Network (RPN)**: Generate candidate bounding box proposals that likely contain objects.
2. **Stage 2 — Detection Head**: Classify each proposal and refine the box coordinates.

This separation allows stage 1 to focus on object/background discrimination (a simpler task) and stage 2 to focus on fine-grained classification.

### R-CNN Family

**R-CNN** (2013) introduced the region proposal + CNN classification paradigm:

1. Use selective search to extract ~2,000 region proposals.
2. Warp each proposal to a fixed size and run through a CNN.
3. Classify with SVMs; refine boxes with a linear regressor.

**Limitation**: Extremely slow — each proposal requires a separate CNN forward pass.

**Fast R-CNN** (2015) solved the efficiency problem with **RoI Pooling**:

1. Run the entire image through the CNN once → produce a shared feature map.
2. Project region proposals onto the feature map → extract fixed-size RoI features via RoI Pooling.
3. Classify and regress from RoI features.

**Faster R-CNN** (2015) replaced selective search with a learned **Region Proposal Network (RPN)** that runs on the shared feature map — making the entire pipeline end-to-end trainable and much faster.

### Faster R-CNN Architecture

```
Image → Backbone (ResNet/VGG) → Feature Map
                                     │
                              ┌──────┴───────┐
                              ▼              ▼
                             RPN        RoI Pooling
                         (proposals)   (per-proposal features)
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                             Class Scores         Box Regression
```

The RPN is trained with a binary classification loss (object vs. background for each anchor) and a box regression loss (for positive anchors). The second stage is trained with a multi-class classification loss and a box regression loss.

**Feature Pyramid Network (FPN)**: Extends Faster R-CNN by building a multi-scale feature pyramid — detecting small objects at high-resolution feature maps and large objects at low-resolution feature maps. FPN dramatically improved small object detection performance.

## Single-Stage Detectors

**Single-stage detectors** skip the region proposal step and directly predict bounding boxes and class labels from a grid over the image, achieving real-time speeds at the cost of some accuracy (a gap that has largely closed in modern architectures).

### YOLO (You Only Look Once)

**YOLOv1** (Redmon et al., 2016) divided the image into a $S \times S$ grid. Each cell predicts $B$ bounding boxes and $C$ class probabilities, all in a single forward pass.

**YOLOv3** added multi-scale predictions (at 3 different feature map scales), anchor boxes, and a darknet-53 backbone — substantially improving accuracy, especially for small objects.

**YOLOv5, YOLOv7, YOLOv8** (community iterations) added further improvements: mosaic augmentation, anchor-free designs, knowledge distillation, and efficient architectures.

**YOLOv8** (Ultralytics, 2023) is anchor-free, supports detection, segmentation, pose estimation, and classification in a unified interface, and achieves state-of-the-art speed-accuracy trade-offs for real-time applications.

**YOLOv10** (2024) eliminates NMS post-processing through dual-label assignment during training, further reducing latency.

### SSD (Single Shot MultiBox Detector)

SSD predicts detections at **multiple feature map scales** simultaneously, using different anchor sizes at different layers of the network. Earlier layers (higher resolution) detect small objects; later layers (lower resolution) detect large objects. This multi-scale strategy enables detection of objects at varying sizes without a feature pyramid.

### RetinaNet and Focal Loss

The main challenge for single-stage detectors is the **extreme class imbalance** between background anchors (negative examples) and object anchors (positive examples). A typical image has thousands of background anchors but only tens of object anchors. Standard cross-entropy loss is dominated by easy negatives.

**Focal Loss** (Lin et al., 2017) addresses this by down-weighting the loss contribution of easy examples:

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

The focusing parameter $\gamma$ (typically 2) reduces the loss for confident correct predictions, focusing training on the hard examples where the model is uncertain. **RetinaNet** with focal loss matched two-stage detector accuracy for the first time with a single-stage architecture.

## Anchor-Free Detectors

Anchor boxes introduce complexity: choosing anchor sizes, aspect ratios, and thresholds requires careful hyperparameter tuning, and anchors are poorly suited for irregularly shaped objects.

**Anchor-free detectors** predict object locations directly from feature map points without predefining anchor shapes:

**FCOS (Fully Convolutional One-Stage)**: For each point in the feature map, predicts:

- Whether it is within a ground-truth object (foreground vs. background).
- The distances $(l, t, r, b)$ from the point to the four sides of the bounding box.
- A centerness score to suppress low-quality detections at the object periphery.

**CenterNet**: Detects objects by locating keypoints at the object center, then predicting width and height offsets from the center.

## Transformer-Based Detection: DETR

**DETR (Detection Transformer)** (Carion et al., 2020) introduced a fully transformer-based approach that eliminates hand-designed components entirely — no anchors, no NMS, no multi-scale feature pyramids.

### DETR Architecture

1. **CNN backbone**: Extracts image features ($C \times H/32 \times W/32$).
2. **Positional encoding**: 2D sine-cosine positional encodings added to the feature map.
3. **Transformer encoder**: Processes the flattened feature sequence.
4. **Transformer decoder**: Takes $N$ learned **object queries** and attends to encoder output.
5. **Prediction heads**: Each of the $N$ decoder outputs predicts a class label and bounding box.

**Set prediction loss**: DETR casts detection as a set prediction problem and uses the **Hungarian algorithm** to find an optimal bipartite matching between predictions and ground-truth objects — ensuring each object is predicted exactly once.

### DETR Advantages and Limitations

**Advantages**:

- Elegantly simple — no NMS, no anchor design.
- Naturally handles global reasoning via self-attention.
- Extends naturally to panoptic segmentation.

**Limitations**:

- Very slow convergence — requires 500 epochs vs. 12 for Faster R-CNN.
- Poor performance on small objects (low-resolution feature maps fed to the transformer).

**Deformable DETR** (2020) addressed these issues with deformable attention — each query attends to only a small set of reference points rather than the full feature map, dramatically reducing memory and accelerating convergence to 50 epochs.

**DINO (DETR with Improved DeNoising)** (2022) further improved DETR with denoising training, mixed query selection, and look-forward-twice scheme — establishing transformer-based detectors as state of the art on COCO.

## Evaluation Metrics

### Intersection over Union (IoU)

IoU measures the overlap between a predicted bounding box $\hat{B}$ and the ground-truth box $B$:

$$\text{IoU} = \frac{\hat{B} \cap B}{\hat{B} \cup B}$$

A prediction is considered a true positive if IoU exceeds a threshold (commonly 0.5).

### Precision and Recall

- **Precision**: Of all the objects the detector found, what fraction were correct?
- **Recall**: Of all true objects, what fraction did the detector find?

### Average Precision (AP)

AP summarizes the precision-recall curve by computing the area under it. Predictions are ranked by confidence score, and precision/recall are computed at each threshold.

### mean Average Precision (mAP)

**mAP** averages AP across all object classes. COCO evaluation further averages across multiple IoU thresholds (0.5 to 0.95 in steps of 0.05), denoted $\text{AP}^{[.50:.95]}$.

Standard benchmarks:

| Benchmark | Scale | Notes |
|-----------|-------|-------|
| **COCO** | 118K train / 5K val | 80 classes, primary detection benchmark |
| **Pascal VOC** | 11K images | 20 classes, IoU@0.5 only |
| **Open Images** | 9M images | 600 classes, large scale |
| **Objects365** | 2M images | 365 classes, challenging |

## Non-Maximum Suppression (NMS)

Detectors often produce multiple overlapping predictions for the same object. **Non-Maximum Suppression** removes duplicates:

1. Sort predictions by confidence score.
2. Iteratively: select the highest-confidence prediction; remove all other predictions with IoU > threshold with this prediction.
3. Repeat until no predictions remain.

**Soft-NMS** decays (rather than eliminates) the scores of overlapping predictions, improving recall for crowded scenes. **Learning-based NMS** uses a small network to determine which predictions to keep, better handling occlusion.

## Data Augmentation for Detection

Object detection benefits from specialized augmentation strategies:

- **Horizontal flipping**: Bounding box coordinates must be mirrored accordingly.
- **Multi-scale training**: Randomly resize images during training to improve scale invariance.
- **Random cropping**: Must handle partially visible objects at crop boundaries.
- **Mosaic augmentation** (YOLO): Combines four images into one, increasing context diversity and small object frequency.
- **MixUp and CutMix**: Blend images and labels to regularize the detector.
- **Copy-paste augmentation**: Copy object instances from one image and paste them into another, controlling object density and context.

## Domain-Specific Object Detection

Object detection is adapted for various domains:

- **Medical imaging**: Detecting lesions, tumors, fractures — requires high recall (missing a finding is costly) and often works on 3D volumetric data.
- **Satellite imagery**: Objects are tiny, rotated arbitrarily, and densely packed (aerial vehicles, ships, buildings).
- **Industrial inspection**: Detecting surface defects, missing components, or assembly errors on production lines.
- **Document detection**: Locating text regions, tables, figures, and form fields in document images.

## The Current Landscape

The convergence of convolutional and transformer-based approaches has produced architectures combining both paradigms:

- **Co-DETR** and **Grounding DINO** combine transformer decoders with strong CNN backbones.
- **Open-vocabulary detection** (OWL-ViT, GLIP, Grounding DINO) uses vision-language alignment to detect any object described in natural language — not just predefined categories.
- **Segment Anything Model (SAM)** generalizes detection toward universal segmentation, further blurring the boundaries between detection, segmentation, and open-world recognition.

Object detection has evolved from hand-crafted features and sliding windows to the sophisticated end-to-end learned systems of today — with real-time performance now achievable even on mobile hardware.
