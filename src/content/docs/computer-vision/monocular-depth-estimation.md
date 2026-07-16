---
title: Monocular Depth Estimation - Inferring Distance from One Image
description: Learn how depth models infer scene geometry from a single camera, the difference between relative and metric depth, and how to evaluate them.
---

Monocular depth estimation predicts the distance from a camera to every visible pixel using a single image. Unlike stereo vision, it has no direct geometric triangulation, so the model must learn visual cues such as perspective, texture, object size, shadows, and occlusion.

## Relative vs Metric Depth

**Relative depth** ranks what is closer or farther:

```text
floor near camera < chair < wall
```

It is useful for image editing and scene layout but does not guarantee meters. **Metric depth** predicts an absolute value, such as `2.4 m`, and needs camera calibration, suitable training data, or additional sensors. Mixing these meanings is a common deployment error.

## Learning Approaches

### Supervised Learning

Train on images paired with LiDAR, structured-light, or manually produced depth maps. This can achieve accurate metric predictions in a familiar domain, but collecting labels is expensive and sensors have their own gaps.

### Self-Supervised Learning

Use adjacent video frames or stereo pairs. A model predicts depth and camera motion, then reconstructs one view from another. Reconstruction error provides the training signal without dense depth labels.

### Foundation Models

Large depth models trained across varied imagery can generalize well for relative depth. They are convenient starting points, but their output scale and failure patterns must still be calibrated for the target camera.

## Evaluation

Common metrics include:

| Metric | Meaning |
| --- | --- |
| Abs Rel | Average relative distance error |
| RMSE | Penalizes large absolute errors |
| $\delta < 1.25$ | Fraction of predictions sufficiently close to truth |

Evaluate by scene type, lighting, distance range, and camera configuration. A single aggregate score can conceal dangerous failures on glossy surfaces, thin objects, or low light.

## Applications

Depth supports robot navigation, augmented reality, background blur, obstacle awareness, and 3D scene understanding. It is not a substitute for a safety-rated range sensor in applications where an incorrect distance can cause harm. Treat predicted depth as an uncertain estimate, fuse it with other sensors when possible, and define a conservative behavior for low-confidence regions.

