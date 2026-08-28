---
title: Structure from Motion (SfM) and Multi-View Stereo
description: Master classical 3D reconstruction principles, epipolar geometry, fundamental and essential matrices, feature triangulation, and dense Multi-View Stereo (MVS).
---

Before neural networks, Neural Radiance Fields (NeRF), or 3D Gaussian Splatting can reconstruct a 3D scene from an unstructured collection of photos, they require accurate **camera poses** (extrinsic rotation and translation) and camera intrinsic parameters for every image.

This foundational problem is solved by **Structure from Motion (SfM)** and **Multi-View Stereo (MVS)**. Given an unordered set of 2D photographs of a static scene taken from multiple vantage points, SfM recovers both the 3D geometry of the scene (structure) and the camera viewpoints (motion) simultaneously.

---

## The SfM Pipeline Overview

```
Unordered Photos [I_1, I_2, ..., I_n]
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Feature Detection & Description (SIFT / SuperPoint)                      │
│    Identifies scale- and rotation-invariant keypoints in every photo        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Feature Matching & Geometric Verification (LightGlue / RANSAC)           │
│    Matches keypoints across pairs; filters outliers via Epipolar Constraints│
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Incremental Reconstruction & Triangulation (COLMAP)                      │
│    Initializes two-view seed -> Registers new images via PnP -> Triangulates│
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Global Bundle Adjustment (Levenberg-Marquardt)                           │
│    Joint non-linear minimization of all 2D reprojection errors              │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
  Sparse 3D Point Cloud + Precise Camera Poses [R_i | t_i]
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Dense Multi-View Stereo (MVS)                                            │
│    Computes per-pixel depth maps and fuses into dense point clouds / meshes │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Epipolar Geometry: The Foundation of Multi-View Vision

When two pinhole cameras observe the same 3D point $\mathbf{X} \in \mathbb{R}^3$, the geometry of their projection centers $\mathbf{C}_1, \mathbf{C}_2$ and image points $\mathbf{x}_1, \mathbf{x}_2$ forms the **Epipolar Plane**:

```
           3D World Point X
             /          \
            /            \
           /              \
  Image 1 /                \ Image 2
   ┌─────/────┐       ┌─────\────┐
   │    x1    │       │     x2   │
   │   /      │       │      \   │
   └──/───────┘       └───────\──┘
     /                         \
Camera Center C1 ────────────► Camera Center C2
              Baseline Vector b
```

### The Epipolar Constraint
The rays passing through $\mathbf{C}_1, \mathbf{x}_1$ and $\mathbf{C}_2, \mathbf{x}_2$ and the baseline $\mathbf{b}$ are coplanar. This geometric relationship is encoded algebraically by the **Essential Matrix ($\mathbf{E}$)** for calibrated cameras, or the **Fundamental Matrix ($\mathbf{F}$)** for uncalibrated cameras:

$$\mathbf{x}_2^\top \mathbf{F} \mathbf{x}_1 = 0$$

where:
- $\mathbf{x}_1, \mathbf{x}_2$ are homogeneous 2D pixel coordinates.
- $\mathbf{F} = \mathbf{K}_2^{-\top} [\mathbf{t}]_\times \mathbf{R} \mathbf{K}_1^{-1}$
- $\mathbf{K}_1, \mathbf{K}_2$ are the $3 \times 3$ camera intrinsic matrices.
- $[\mathbf{t}]_\times$ is the skew-symmetric cross-product matrix of translation vector $\mathbf{t}$.
- $\mathbf{R}$ is the relative rotation matrix.

Given point $\mathbf{x}_1$ in the first image, its corresponding match in the second image **must lie along the 1D epipolar line** $\mathbf{l}_2 = \mathbf{F} \mathbf{x}_1$. This reduces a 2D image search to a 1D line search, dramatically improving feature matching accuracy.

---

## Solving for Pose: The 8-Point Algorithm & RANSAC

Given at least 8 matched point correspondences across two images:
1. Each match $(\mathbf{x}_1, \mathbf{x}_2)$ yields a linear constraint on the entries of $\mathbf{F}$.
2. Singular Value Decomposition (SVD) solves the homogeneous linear system $\mathbf{A} \mathbf{f} = 0$.
3. Singularity is enforced by setting the smallest singular value of $\mathbf{F}$ to zero ($\det(\mathbf{F}) = 0$).
4. **RANSAC (Random Sample Consensus)** iteratively samples random 8-point subsets to identify inlier matches and reject false correspondence outliers.

---

## Triangulation and Bundle Adjustment

### 1. Triangulation
Once camera poses $\mathbf{P}_1 = \mathbf{K}_1 [\mathbf{I} \mid \mathbf{0}]$ and $\mathbf{P}_2 = \mathbf{K}_2 [\mathbf{R} \mid \mathbf{t}]$ are known, the 3D position of landmark point $\mathbf{X}$ is computed by finding the intersection of the two projection rays using direct linear transformation (DLT).

### 2. Bundle Adjustment (BA)
Because sensor noise and measurement approximations cause rays not to intersect perfectly, **Bundle Adjustment** optimizes all camera poses and 3D points simultaneously by minimizing the total non-linear **reprojection error**:

$$\min_{\{\mathbf{P}_j\}, \{\mathbf{X}_i\}} \sum_{j=1}^M \sum_{i=1}^N v_{ij} \, \rho\left( \| \mathbf{x}_{ij} - \pi(\mathbf{P}_j, \mathbf{X}_i) \|_2^2 \right)$$

where $\pi(\mathbf{P}_j, \mathbf{X}_i)$ is the predicted 2D projection, $\mathbf{x}_{ij}$ is the observed 2D feature, $v_{ij} \in \{0, 1\}$ indicates visibility, and $\rho$ is a robust Cauchy or Huber loss function. Solved using the **Levenberg-Marquardt algorithm** with sparse Schur complement decomposition.

---

## Multi-View Stereo (MVS)

While SfM outputs a sparse point cloud (typically thousands of points at distinct corners), **Multi-View Stereo (MVS)** reconstructs dense surfaces (millions of points, complete depth maps per pixel):
- **PatchMatch Stereo:** Propagates random 3D plane hypotheses across neighboring pixels, iteratively optimizing photometric consistency.
- **Depth Map Fusion:** Fuses individual depth maps into a watertight, textured 3D mesh via Poisson surface reconstruction or Truncated Signed Distance Functions (TSDF).

---

## SfM's Essential Role in Modern Generative AI

Modern novel-view synthesis systems—such as **Neural Radiance Fields (NeRF)** and **3D Gaussian Splatting (3DGS)**—do not replace SfM; **they rely directly on it**:
1. **COLMAP (the standard open-source SfM tool)** first processes user photos to recover precise camera poses $(\mathbf{R}_i, \mathbf{t}_i)$ and camera field-of-view angles.
2. The **sparse point cloud** generated by COLMAP provides the critical spatial initialization seeds for 3D Gaussians.

Without SfM camera registration, training modern 3D representations is impossible.

---

## Key Takeaways

- Structure from Motion simultaneously estimates camera poses and 3D scene landmarks from 2D image collections.
- Epipolar geometry and the Fundamental matrix $\mathbf{x}_2^\top \mathbf{F} \mathbf{x}_1 = 0$ reduce correspondence search from 2D space to a 1D line constraint.
- Bundle Adjustment utilizes Levenberg-Marquardt non-linear optimization to minimize reprojection errors across all views.
- Robust SfM tools like COLMAP serve as the mandatory data preparation backbone for modern NeRF and 3D Gaussian Splatting pipelines.
