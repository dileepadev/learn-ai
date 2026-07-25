---
title: Neural Radiance Fields (NeRF) - Rendering 3D Scenes from 2D Images
description: Learn how NeRF represents a scene as a continuous volumetric function and renders novel views through ray marching.
---

A Neural Radiance Field represents a scene not as a mesh or point cloud but as a continuous function learned by a neural network. Given a 3D position and viewing direction, it predicts a color and density:

```text
F(x, y, z, theta, phi) -> (r, g, b, sigma)
```

`sigma` is volume density (how opaque that point is), and the color can change with viewing direction to capture reflections and specular highlights.

## Volume Rendering

To render a pixel, NeRF casts a ray through the scene and samples points along it. The final color is a weighted integral of sampled colors, where each sample's contribution depends on the density accumulated so far:

$$C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(t)\,c(t)\,dt, \quad T(t) = \exp\left(-\int_{t_n}^{t} \sigma(s)\,ds\right)$$

`T(t)` is accumulated transmittance—the probability a ray travels to distance `t` without being blocked. In practice this integral is approximated with discrete quadrature over sampled points.

## Positional Encoding

A plain MLP struggles to represent high-frequency detail (sharp edges, fine textures) because low-dimensional inputs bias networks toward smooth functions. NeRF maps each coordinate through sinusoidal functions at increasing frequencies before feeding the network:

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

This lets the same network represent both coarse geometry and fine detail.

## Training and Limitations

A NeRF is trained per scene from a set of posed photographs (images with known camera position and orientation), minimizing the difference between rendered and observed pixels. No 3D supervision is needed—geometry emerges purely from multi-view photometric consistency.

Classic NeRF is slow: rendering one image requires evaluating the network millions of times. Follow-up work (Instant-NGP, Plenoxels, 3D Gaussian Splatting) replaces or augments the MLP with explicit spatial data structures to reach real-time rendering. Common limitations include difficulty with dynamic scenes, transient occluders (people walking through a shot), and reflective or transparent surfaces that violate the static, opaque-ish assumptions of the model.
