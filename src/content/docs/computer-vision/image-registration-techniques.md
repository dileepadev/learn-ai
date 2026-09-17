---
title: Image Registration Techniques - Aligning Multiple Images into One Coordinate System
description: Learn how image registration aligns images taken from different viewpoints, times, or sensors into a common coordinate frame.
---

Image registration aligns two or more images of the same scene, taken from different viewpoints, at different times, or with different sensors, into a single common coordinate system so they can be directly compared or combined.

## Why Registration Is Needed

A satellite image of the same field taken in spring and autumn will show the same physical location at different pixel coordinates due to slight differences in orbit or camera angle; two MRI scans of the same patient taken months apart will show anatomy at slightly different positions and orientations. Comparing these images meaningfully — detecting change over time, fusing information from different sensors — requires first aligning them so corresponding physical points fall at corresponding pixel coordinates.

## Feature-Based Registration

Feature-based methods detect distinctive keypoints in each image (using detectors like SIFT or ORB), match corresponding keypoints between images based on their local descriptors, and then estimate a geometric transformation (rigid, affine, or projective) that best maps matched points from one image onto the other.

```text
detect keypoints in image A and image B
match keypoints between A and B by descriptor similarity
estimate transformation (e.g. via RANSAC to reject bad matches)
warp image B into image A's coordinate frame using the estimated transformation
```

RANSAC (Random Sample Consensus) is critical here because keypoint matching produces some incorrect matches, and RANSAC robustly estimates the transformation by repeatedly fitting to random small subsets of matches and keeping the fit that most matches agree with.

## Intensity-Based and Deep Learning Registration

Intensity-based methods skip explicit keypoint detection and instead directly optimize a transformation that maximizes a similarity measure (like mutual information, useful for multi-modal registration such as aligning an MRI to a CT scan of the same patient) between the warped and reference images. Deep learning approaches train a neural network to directly predict the transformation, or to predict a dense deformation field for non-rigid registration where different parts of the image need to move differently — common in medical imaging where organs and tissue deform between scans.

## Practical Guidance

Use feature-based registration for scenes with distinctive, stable visual features (buildings, terrain, printed documents), which is generally faster and more interpretable. Use intensity-based or learned deformation-field registration for smoother, feature-sparse content like medical images or when registering images from fundamentally different sensor modalities where matching local features directly is unreliable.
