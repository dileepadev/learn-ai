---
title: The YOLO Family of Real-Time Object Detectors
description: Learn how YOLO frames detection as a single-pass regression problem, and how the family evolved through anchors, multi-scale heads, and anchor-free designs.
---

YOLO ("You Only Look Once") is a family of single-stage object detectors prized for real-time speed. Instead of proposing regions and then classifying them, YOLO predicts boxes and classes in a single forward pass over the image.

## The Original Idea

YOLOv1 divided the image into an SxS grid. Each cell predicted bounding boxes with confidence scores and class probabilities. Framing detection as one regression problem made the system fast and let it reason about global context, at the cost of weaker localization for small or crowded objects.

## What a Detection Head Predicts

For each candidate location:

```text
(x, y, w, h)   box geometry
objectness     probability that an object is present (in some versions)
class scores   probability per class
```

## Evolution of the Family

| Stage | Notable changes |
|-------|-----------------|
| YOLOv2 / v3 | Anchor boxes, batch normalization, multi-scale prediction heads with feature pyramids |
| YOLOv4 / v5 | Better backbones (CSP), mosaic augmentation, stronger training recipes, user-friendly tooling |
| YOLOX / v6 / v7 / v8 | Decoupled heads, anchor-free prediction, advanced label assignment |
| Later versions | NMS-free training, attention modules, improved efficiency and unified multi-task support |

## Key Concepts

- **Anchor boxes** (older versions): predefined box shapes the network adjusts; anchor-free models predict box distances from a point directly
- **Non-maximum suppression (NMS)**: removes duplicate overlapping predictions at inference
- **IoU-based losses** (GIoU, DIoU, CIoU): optimize overlap with the ground truth rather than coordinate error
- **Label assignment**: deciding which predictions are matched to which ground-truth objects, which strongly affects accuracy

## Intersection over Union

```python
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)
```

## Typical Usage

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=50, imgsz=640)
results = model("street.jpg")
```

## Choosing a Variant

Small variants (nano, small) run on edge devices; larger ones trade speed for accuracy. Check the license of each implementation before commercial use, since terms differ between versions and maintainers.

## Strengths and Limitations

Strengths: speed, simple deployment, strong tooling for training and export. Limitations: small and heavily overlapping objects remain hard, and a closed label set means new categories require retraining, which open-vocabulary detectors address.
