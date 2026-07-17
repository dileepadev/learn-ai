---
title: Visual Object Tracking - Maintaining Identity Across Video Frames
description: Understand tracking-by-detection, motion models, appearance matching, and the metrics used to track objects through video.
---

Object tracking assigns a persistent identity to an object as it moves through video. A detector can say “there is a car in this frame”; a tracker connects that car to the same car in the next frame.

## Tracking by Detection

The most common production pipeline separates detection and association:

```text
frame -> detector -> bounding boxes
                    -> match boxes to existing tracks -> track IDs
```

For each new frame, the tracker predicts where each known object should be, matches detections to those predictions, starts tracks for new objects, and retires tracks that have disappeared.

## Motion and Appearance

A **Kalman filter** estimates position and velocity, allowing a tracker to predict a box through a short detector miss. Matching commonly combines:

- **motion distance:** is the new box near the predicted box?
- **IoU:** how much do the old and new boxes overlap?
- **appearance embedding:** do cropped objects look alike?

SORT uses a Kalman filter and IoU matching. DeepSORT adds appearance embeddings, improving performance when objects cross paths. ByteTrack also associates lower-confidence detections, which helps retain partially occluded objects.

## Hard Cases

Tracking fails most often when:

- two similar objects overlap
- an object leaves the frame and returns
- illumination, scale, or camera angle changes sharply
- the detector misses several consecutive frames

Long-term re-identification requires stronger appearance models and should be evaluated carefully because it introduces substantial privacy risk.

## Metrics

**MOTA** combines missed objects, false positives, and identity switches. **IDF1** emphasizes whether an object retains the correct identity. **HOTA** balances detection quality with association quality. Report more than one metric: a tracker can have good detection coverage while frequently swapping identities.

## Design Choices

Use a lightweight tracker for short-lived in-camera tracking, such as counting vehicles at an intersection. Use appearance features only when the task needs them and the data handling is justified. For safety or billing workflows, retain uncertainty: a low-confidence, recently occluded track should not be treated as an indisputable identity.

