---
title: Optical Flow - Estimating Motion Between Video Frames
description: Explore optical flow, its core assumptions, modern deep models, and practical uses in video understanding.
---

Optical flow estimates how pixels move between two video frames. Rather than detecting objects, it produces a dense vector field:

```text
pixel (x, y) -> displacement (u, v)
```

The field can reveal camera motion, moving vehicles, gestures, or deformation.

## The Brightness Constancy Assumption

Classical optical flow assumes a point keeps approximately the same brightness as it moves:

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

After a small-motion approximation, this yields the optical-flow constraint:

$$I_xu + I_yv + I_t = 0$$

One equation cannot determine two unknown motion components, which is called the aperture problem. Classical methods add local smoothness assumptions to make the problem solvable.

## Classical and Learned Methods

Lucas-Kanade estimates a shared motion for a small neighborhood. Horn-Schunck estimates dense flow while penalizing abrupt changes. Modern networks such as RAFT learn feature matching and iterative refinement from labeled or synthetic motion data.

## What Makes Flow Difficult

- occluded pixels have no true match in the next frame
- reflections and changing lighting violate brightness constancy
- fast motion can move beyond the model's search range
- repeated textures create ambiguous matches

Occlusion masks and forward-backward consistency checks help identify unreliable vectors.

## Evaluation and Use

**Endpoint error (EPE)** is the average Euclidean distance between predicted and true motion vectors. Report errors separately for slow and fast motion and for occluded areas.

Flow is used for video stabilization, frame interpolation, action recognition, motion segmentation, and tracking. It is a motion estimate—not proof of object identity or intent—and should be combined with detections and temporal context when decisions depend on it.

