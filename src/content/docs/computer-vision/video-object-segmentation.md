---
title: Video Object Segmentation - Tracking Masks Across Frames
description: Learn how video object segmentation propagates pixel-level object masks across time, and how it differs from single-frame segmentation and tracking.
---

Video object segmentation (VOS) tracks pixel-level masks for objects across a video sequence, combining the spatial precision of segmentation with temporal consistency across frames.

## Semi-Supervised vs. Unsupervised VOS

Semi-supervised VOS (the more common practical setting) is given the ground-truth mask of the target object in the first frame and must propagate a consistent mask for that object through the rest of the video, even as it moves, is partially occluded, or changes appearance. Unsupervised VOS receives no initial mask and must automatically discover and segment the most salient object or objects in the video without human input.

```text
Frame 1: user-provided mask for "the dog"
Frame 2-N: model propagates and updates "the dog" mask as it moves, turns, and is briefly occluded by a tree
```

## Core Techniques

Matching-based methods build a memory of the target object's appearance from previous frames (including the initial reference mask) and, for each new frame, match pixels against this memory to determine which pixels belong to the object now — this handles appearance changes better than assuming the object looks identical to the very first frame. Propagation-based methods instead pass the mask from the previous frame forward, refining it using optical flow or a learned refinement network, which works well for smooth motion but can drift over long sequences without correction.

## Handling Occlusion and Reappearance

A hard case for VOS is an object leaving the frame or being fully occluded and then reappearing — the model must recognize the reappearing object as the same tracked instance rather than starting a new, unlinked segmentation. Memory-based approaches that retain appearance information from multiple past frames (not just the immediately preceding one) handle this better than simple frame-to-frame propagation.

## Practical Guidance

Foundation segmentation models with strong per-frame quality (built on architectures like Segment Anything extended with video memory) have significantly simplified VOS pipelines, often outperforming older specialized propagation networks while requiring only a single reference mask as input. For applications needing real-time performance (live video editing, robotics), weigh mask quality against inference latency explicitly, since the strongest VOS models are not always the fastest.
