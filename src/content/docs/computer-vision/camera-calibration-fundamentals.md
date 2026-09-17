---
title: Camera Calibration Fundamentals
description: Learn how camera calibration recovers intrinsic and extrinsic parameters needed to relate pixel coordinates to real-world measurements.
---

Camera calibration determines the parameters that describe exactly how a camera projects the 3D world onto a 2D image, a necessary step before a vision system can make accurate real-world measurements from pixel coordinates rather than just qualitative visual judgments.

## Intrinsic vs. Extrinsic Parameters

Intrinsic parameters describe properties of the camera itself, independent of where it's positioned: focal length, the principal point (where the optical axis meets the image sensor), and lens distortion coefficients. Extrinsic parameters describe the camera's position and orientation relative to the world — its rotation and translation — which change every time the camera moves but don't depend on the camera's internal optics at all.

```text
pixel coordinates = Intrinsic Matrix × Extrinsic Matrix × world coordinates
```

## The Pinhole Camera Model

The standard pinhole camera model relates a 3D world point to its 2D image projection through a perspective projection, parameterized by the intrinsic matrix `K`, which encodes focal length and principal point:

```text
K = [ fx   0   cx ]
    [ 0    fy  cy ]
    [ 0    0   1  ]
```

`fx` and `fy` are the focal length expressed in pixel units along each axis, and `(cx, cy)` is the principal point, typically near the image center. Real lenses also introduce radial and tangential distortion, which bends straight lines in the world into slightly curved lines in the image, especially toward the edges of the frame — calibration also estimates distortion coefficients to correct for this.

## Calibration Using a Known Pattern

The standard calibration procedure captures multiple images of a checkerboard or similar pattern with precisely known geometry from different angles, detects the pattern's corner points in each image, and solves for the intrinsic and distortion parameters that best explain the observed corner positions across all the captured views, given the known real-world geometry of the pattern.

## Why This Matters for Downstream Vision Tasks

Accurate calibration is a prerequisite for any vision task requiring real-world metric measurements from images: 3D reconstruction, structure from motion, augmented reality overlay alignment, robot vision for grasping objects at precise real-world coordinates, and stereo depth estimation, which specifically requires accurate calibration of both cameras in a stereo pair relative to each other to correctly triangulate depth.

## Practical Guidance

Recalibrate whenever a camera's lens, zoom setting, or mounting changes, since intrinsic parameters are specific to a fixed lens and focus configuration and don't transfer across cameras or even across a zoom change on the same camera. For applications only needing qualitative object detection or classification rather than metric measurements, calibration is often unnecessary — reserve the effort for tasks that genuinely need to convert pixels into real-world distances or angles.
