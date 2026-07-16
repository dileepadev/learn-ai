---
title: Pose Estimation - Finding Human Keypoints in Images and Video
description: Learn how pose estimation detects body joints, how top-down and bottom-up systems differ, and how to evaluate keypoint models.
---

Pose estimation locates meaningful points on a body, such as shoulders, elbows, hips, and knees. The resulting skeleton is useful when a bounding box is not enough: it describes *how* a person is moving rather than just where they are.

## From Pixels to Keypoints

A common 2D pose contains 17 keypoints from the COCO convention. Each prediction has an `(x, y)` position and confidence:

```text
left wrist: (214, 318), confidence: 0.94
left elbow: (188, 251), confidence: 0.91
```

Models can also estimate a third coordinate, infer 3D joints from multiple cameras, or predict hands and faces.

## Two Main Designs

### Top-Down

1. Detect every person.
2. Crop each detected person.
3. Run a single-person pose model on each crop.

This usually produces accurate poses but becomes slower as crowds grow. HRNet and ViTPose are common top-down families.

### Bottom-Up

1. Detect all keypoints in the image.
2. Group them into individual people.

Bottom-up approaches, such as OpenPose, avoid running a model per person. Grouping becomes difficult when people overlap or keypoints are missing.

## Heatmaps and Regression

Many models predict one heatmap per joint. The highest-valued pixel is the joint location:

```text
image -> backbone -> 17 heatmaps -> peak location for each joint
```

Direct regression predicts coordinates in one step and can be faster, but heatmaps often preserve spatial detail better. Modern transformer models can combine both approaches with learned queries.

## Evaluation

COCO uses **Object Keypoint Similarity (OKS)**, which is analogous to IoU for keypoints. A prediction is rewarded when its joints are close to annotated joints, with tolerance adjusted for person scale and keypoint type.

Useful practical checks include:

- accuracy under occlusion and unusual viewpoints
- latency per person and per frame
- stability of joints across adjacent video frames
- performance across clothing, skin tones, mobility aids, and body types

## Applications and Limits

Pose estimation supports exercise feedback, sports analysis, animation, ergonomics, gesture interfaces, and fall detection. It should not be treated as a reliable proxy for identity, emotion, intent, health status, or ability. Obtain consent where people can be identified, minimize retention of video, and test the model on the population and camera conditions where it will run.

## Practical Workflow

Start with a pretrained model and a small labeled sample from the real camera setup. Define which joints and failure modes matter, measure accuracy and frame rate together, then add temporal smoothing only after measuring whether it hides important rapid motion. For sensitive applications, keep a human review path rather than making high-impact decisions from pose alone.

