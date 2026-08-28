---
title: 3D Gaussian Splatting
description: Explore 3D Gaussian Splatting for real-time radiance field rendering, rasterization mechanics, adaptive density control, and architectural comparison with Neural Radiance Fields (NeRF).
---

**3D Gaussian Splatting (3DGS)** is a breakthrough technique in computer vision and computer graphics for novel view synthesis and 3D scene reconstruction. Introduced by Kerbl et al. in 2023, 3DGS achieves visual quality comparable to or exceeding **Neural Radiance Fields (NeRF)** while unlocking **real-time rendering speeds** (typically $\ge 100\text{ FPS}$ at 1080p resolution) and drastically reducing training time from hours to minutes.

Unlike volumetric neural representations that query an expensive multilayer perceptron (MLP) for every sample along each camera ray, 3D Gaussian Splatting represents 3D scenes as explicit collections of millions of flexible, parameterized 3D Gaussians that are projected and rasterized directly onto the 2D image plane.

---

## NeRF vs. 3D Gaussian Splatting

| Feature | Neural Radiance Fields (NeRF) | 3D Gaussian Splatting (3DGS) |
| :--- | :--- | :--- |
| **Scene Representation** | Implicit (continuous MLP or voxel/hash grid) | Explicit (millions of anisotropic 3D Gaussians) |
| **Rendering Algorithm** | Volumetric ray marching with numerical integration | Differentiable tile-based rasterization (splatting) |
| **Inference / Render Speed** | Slow ($\sim 1\text{ to }30\text{ FPS}$ depending on hash-grids) | Ultra-fast ($100\text{--}200+\text{ FPS}$ at 1080p) |
| **Training Time** | Hours to tens of minutes | $10\text{--}30\text{ minutes}$ for complex scenes |
| **Editability & Composition** | Difficult (weights are globally entangled) | Direct (Gaussians can be moved, filtered, or merged) |
| **Memory Footprint** | Small (MLP weights) to Moderate (hash tables) | Moderate to High (millions of Gaussian parameters) |

```
NeRF (Volumetric Ray Marching):
Camera Origin ---> [ Ray ] ---> Sample points -> [ MLP Query ] -> Density + Color -> Volume Render -> Pixel

3D Gaussian Splatting (Forward Splatting):
3D Gaussians (μ, Σ, α, SH) ---> [ Camera Projection ] ---> 2D Ellipses ---> [ Tile-Based Rasterizer ] ---> Framebuffer
```

---

## Mathematical Formulation of 3D Gaussians

A 3D scene is modeled as an unstructured cloud of 3D Gaussians. Each individual Gaussian $i$ is parameterized by:

1. **Center Position (Mean):** $\mu \in \mathbb{R}^3$
2. **3D Covariance Matrix:** $\Sigma \in \mathbb{R}^{3 \times 3}$
3. **Opacity (Alpha):** $\alpha \in [0, 1]$
4. **Color / Directional Radiance:** Spherical Harmonics (SH) coefficients $\mathbf{c}$ modeling view-dependent appearance.

The probability density of a 3D Gaussian centered at $\mu$ is given by:

$$G(x) = \exp\left(-\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)\right)$$

### Covariance Factorization (Scale and Rotation)

To ensure the covariance matrix $\Sigma$ remains positive semi-definite during gradient descent optimization, it is factorized into an ellipsoidal scale matrix $S = \text{diag}(s_x, s_y, s_z)$ and a rotation matrix $R$ parameterized by a unit quaternion $q \in \mathbb{R}^4$:

$$\Sigma = R S S^\top R^\top$$

This formulation guarantees valid physical geometry while allowing independent optimization of shape, orientation, and position.

---

## The Splatting & Rasterization Pipeline

To render 3D Gaussians from a camera viewpoint, they must be projected onto a 2D image plane—a process known as **splatting**.

### 1. 2D Projection (Zwicker et al. Formulation)
Given a projective camera transformation matrix $W$ and the Jacobian of the affine perspective approximation $J$, the 2D covariance matrix $\Sigma'$ on the image sensor is:

$$\Sigma' = J W \Sigma W^\top J^\top$$

The resulting 2D Gaussian defines an elliptical splat on the screen centered at projected coordinates $\mu' \in \mathbb{R}^2$.

### 2. Fast Tile-Based Differentiable Rasterizer
Traditional NeRF ray marchers test millions of empty space locations. 3DGS introduces an ultra-fast hardware-accelerated rasterization algorithm:

1. **Screen Tiling:** The screen is divided into $16 \times 16$ pixel tiles.
2. **Culling & Bounding:** Gaussians with 99% confidence radii intersecting each tile are identified and instanced.
3. **Radix Sorting:** Each Gaussian is assigned a 64-bit key combining its Tile ID and camera view depth. A fast GPU radix sort orders all splats from front to back.
4. **$\alpha$-Blending Integration:** For each pixel, color is accumulated using standard front-to-back compositing:

$$C(p) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

where $c_i$ is the view-dependent color evaluated from spherical harmonics and $\alpha_i$ is the modulated opacity:

$$\alpha_i = \alpha_{\text{base}} \cdot \exp\left(-\frac{1}{2} (p - \mu'_i)^\top (\Sigma'_i)^{-1} (p - \mu'_i)\right)$$

When cumulative transmittance drops below a threshold (e.g., $T_i < 0.0001$), ray evaluation terminates early.

---

## Adaptive Density Control

During training, 3DGS dynamically adjusts the distribution of Gaussians through three core operations:

1. **Cloning Under-Reconstructed Regions:** Large regions with high position error gradients but small Gaussian footprints are cloned and shifted along the gradient direction.
2. **Splitting Over-Reconstructed Regions:** Oversized Gaussians covering high-variance geometry are split into two smaller Gaussians with scaled-down variance.
3. **Pruning Transparent Gaussians:** Gaussians with opacity $\alpha$ dropping below a minimal threshold (e.g., $\alpha < 0.005$) or volume exceeding scene boundaries are periodically removed to prevent floaters.

```
       [ Initialization: Sparse SFM Point Cloud ]
                         │
                         ▼
             [ Differentiable Rasterization ]
                         │
                         ▼
        [ Loss Computation (L1 + D-SSIM) ]
                         │
                         ▼
      ┌──────────────────┴──────────────────┐
      ▼                                     ▼
[ Large Gradient & Small Scale ]     [ Large Gradient & Large Scale ]
      │                                     │
      ▼                                     ▼
   Clone Gaussian                        Split into Two Gaussians
      └──────────────────┬──────────────────┘
                         │
                         ▼
               [ Periodic Alpha Pruning ]
```

---

## Loss Function

Training minimizes a combined photometric loss combining pixel-level $L_1$ loss and structural similarity ($D\text{-SSIM}$):

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_{1} + \lambda \mathcal{L}_{D\text{-SSIM}}$$

Typically, $\lambda \approx 0.2$. The gradients flow backwards through the differentiable rasterizer directly into Gaussian positions $\mu$, quaternions $q$, scale vectors $s$, opacities $\alpha$, and spherical harmonics coefficients $\mathbf{c}$.

---

## Applications and Future Directions

- **Real-Time Digital Twins & Robotics:** Autonomous vehicles and robots can build photorealistic, metric 3D scene maps in real time.
- **VR/AR Asset Streaming:** Explicit 3D Gaussian representations can be compressed, streamed over standard web sockets, and rendered at 120 FPS inside headsets.
- **Dynamic 4D Splatting:** Extending Gaussians with temporal deformation fields to capture non-rigid moving human bodies and fluids.
- **Physics Simulation:** Treating Gaussians as physical particle proxies to model soft-body collisions, gravity, and fluid dynamics directly on photorealistic scans.
