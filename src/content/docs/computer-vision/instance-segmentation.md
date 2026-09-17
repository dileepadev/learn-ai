---
title: Instance Segmentation - Separating Individual Objects Pixel by Pixel
description: Learn how instance segmentation distinguishes between individual object instances of the same class, and how it differs from semantic and panoptic segmentation.
---

Instance segmentation assigns a pixel-precise mask to each individual object in an image, distinguishing between separate instances of the same class — three people in a photo get three separate masks, not one combined "person" region.

## How It Differs from Semantic Segmentation

Semantic segmentation labels every pixel with a class but does not distinguish between instances — all pixels belonging to any person are labeled "person" as one undivided region. Instance segmentation adds identity: pixel-level masks are grouped by which specific object they belong to, so overlapping or adjacent objects of the same class remain separable.

```text
Semantic:  {person pixels} -> one "person" mask covering everyone
Instance:  {person pixels} -> "person #1" mask, "person #2" mask, "person #3" mask
```

Panoptic segmentation unifies both: it labels every pixel with a class the way semantic segmentation does, while additionally separating distinct instances of "thing" classes (countable objects like people or cars) the way instance segmentation does, leaving uncountable "stuff" classes (sky, road, grass) as undivided regions.

## Architecture Approaches

Two-stage detectors like Mask R-CNN first propose candidate object regions (as in object detection), then predict a pixel mask within each proposed region — effectively adding a mask-prediction branch on top of a standard object detector. One-stage and transformer-based approaches predict masks more directly, often framing instance segmentation as a set-prediction problem where the model outputs a fixed number of candidate masks and labels, then matches them to ground truth during training.

## Evaluation

Mask-level mean average precision (mask mAP) evaluates instance segmentation, computed similarly to object detection mAP but using mask overlap (intersection over union between predicted and ground-truth pixel masks) rather than bounding box overlap to determine whether a prediction counts as correct.

## Practical Guidance

Use instance segmentation when downstream logic needs precise per-object boundaries — robotic grasping, medical image analysis counting individual cells, or video editing tools that isolate individual objects for separate manipulation. If you only need to count or classify objects without needing pixel-exact boundaries, plain object detection is cheaper to run and label.
