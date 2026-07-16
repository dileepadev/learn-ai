---
title: 3D Reconstruction - Building Scenes from Images
description: Learn the main ways computer vision recovers 3D shape from images, from multi-view geometry to neural scene representations.
---

3D reconstruction turns photographs or video into a representation of scene geometry. The result may be a sparse point cloud, dense mesh, depth map, or a neural representation that can render new views.

## Multi-View Geometry

When a scene is viewed from multiple camera positions, corresponding image points constrain their 3D location. A traditional pipeline is:

1. detect and match visual features across images
2. estimate camera poses with structure from motion
3. triangulate matched points into a sparse cloud
4. densify the cloud and reconstruct a surface

Accurate camera calibration, overlap, texture, and varied viewpoints are essential. A blank wall supplies too few reliable features; a glossy object can produce inconsistent matches.

## Representations

| Representation | Strength | Tradeoff |
| --- | --- | --- |
| Point cloud | Simple, captures measured locations | No explicit surface |
| Mesh | Easy to render and edit | Surface extraction can be fragile |
| Voxel grid | Regular 3D structure | Memory grows quickly with resolution |
| Neural field | High-quality novel views | Training and rendering can be costly |

Neural radiance fields (NeRFs) learn a function that maps 3D position and view direction to color and density. They can render convincing new views but do not automatically provide a clean, physically accurate mesh.

## Measuring Quality

For known geometry, compare predicted and reference surfaces with Chamfer distance, point-to-surface distance, or completeness. For novel-view synthesis, use held-out camera views and image metrics alongside human inspection. Visual realism alone can hide missing or distorted geometry.

## Practical Capture Guidance

Walk around the object with consistent exposure, include overlap between images, avoid motion blur, and record calibration information. Reconstruction is especially unreliable for transparent, reflective, thin, or moving objects. In surveying, healthcare, or robotics, validate scale and accuracy against physical measurements rather than trusting a rendered scene.

