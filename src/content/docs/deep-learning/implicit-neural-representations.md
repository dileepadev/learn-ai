---
title: Implicit Neural Representations
description: Explore Implicit Neural Representations (INRs) — neural networks that encode continuous signals as functions of coordinates — covering SIREN with sinusoidal activations, occupancy networks, signed distance functions, applications in 3D shape representation, image compression, and physics simulation.
---

**Implicit Neural Representations (INRs)**, also called **coordinate networks** or **neural fields**, represent signals not as discrete arrays of values (pixels, voxels, mesh vertices) but as **continuous functions parameterized by a neural network**. The network maps coordinates to signal values:

$$f_\theta : \mathbf{x} \in \mathbb{R}^n \mapsto v \in \mathbb{R}^m$$

For a 2D image, $\mathbf{x}$ would be pixel coordinates $(u, v)$ and $v$ would be RGB values. For a 3D shape, $\mathbf{x}$ would be a 3D point and $v$ might be an occupancy probability (inside/outside) or a signed distance value. For a video, $\mathbf{x}$ would be a spatio-temporal coordinate $(x, y, t)$ and $v$ would be an RGB value at that location and time.

The neural network acts as a **continuous, differentiable, and implicitly compressed** representation of the signal — "implicit" because the shape or signal is not stored explicitly but is implicitly defined by the neural network's input-output function.

## Motivation: Why Not Just Arrays?

Traditional discrete representations have fundamental limitations:

**Resolution rigidity**: A 512×512 image contains fixed-resolution information; upsampling introduces artifacts. An INR represents the image at arbitrary resolution — query any coordinate, get a value.

**Memory scaling**: A high-resolution 3D voxel grid at $1024^3$ resolution requires 4 billion voxels. An INR of fixed network size can represent complex 3D shapes regardless of their geometric complexity.

**Differentiability**: INRs are differentiable with respect to coordinates — enabling gradient-based operations like computing surface normals, curvature, or solving partial differential equations by differentiating the network.

**Continuous domains**: Physical phenomena are naturally continuous. Representing them as continuous functions (rather than discrete samples) is often more natural and enables exact evaluation at any point.

## The Activation Function Problem

A naive MLP with ReLU activations is a poor INR. The key issue: **spectral bias** (the "F-principle") — networks with smooth, non-oscillatory activations strongly prefer learning low-frequency components of a signal and require vast depth/width to represent high-frequency details.

For a typical signal (natural image, 3D surface), the model fits smooth low-frequency content quickly but fails to represent sharp edges, fine texture, or high-frequency geometry.

## SIREN: Sinusoidal Representation Networks

**SIREN** (Sitzmann et al., 2020) solves the spectral bias problem by replacing all activations with **sine functions**:

$$\text{SIREN}(\mathbf{x}) = \mathbf{W}_n \circ \phi_{n-1} \circ \cdots \circ \phi_0(\mathbf{x}), \quad \phi_i(\mathbf{x}_i) = \sin(\omega_0 \mathbf{W}_i \mathbf{x}_i + \mathbf{b}_i)$$

where $\omega_0$ is a frequency hyperparameter (typically 30 for images) that scales the input to the sine functions.

Critical to SIREN's performance is its **initialization scheme**: weights are initialized from $\mathcal{U}(-\sqrt{6/n}, \sqrt{6/n})$ for the first layer and $\mathcal{U}(-\sqrt{6/n}/\omega_0, \sqrt{6/n}/\omega_0)$ for subsequent layers — ensuring the distribution of activations stays in a regime where the sine function is expressive across layers.

```python
import torch
import torch.nn as nn
import numpy as np

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30., is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0=30.):
        super().__init__()
        layers = [SirenLayer(in_features, hidden_features, omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_features, hidden_features, omega_0))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords):
        return self.net(coords)

# Fit a SIREN to a 2D image
def fit_image_siren(image_tensor, num_epochs=1000, lr=1e-4):
    H, W = image_tensor.shape[:2]
    
    # Create coordinate grid in [-1, 1]
    y_coords = torch.linspace(-1, 1, H)
    x_coords = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [H*W, 2]
    pixels = image_tensor.reshape(-1, 3)  # [H*W, 3]
    
    model = SIREN(in_features=2, hidden_features=256, hidden_layers=3, out_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        pred = model(coords)
        loss = ((pred - pixels) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model
```

SIREN's derivatives are themselves sinusoidal — meaning its gradients and higher-order derivatives are analytically available and well-behaved, making it ideal for solving differential equations.

## Positional Encodings: Fourier Features

An alternative to sinusoidal activations is **Fourier feature mapping** (Tancik et al., 2020) — projecting input coordinates through a fixed random Fourier feature matrix before passing through a standard MLP:

$$\gamma(\mathbf{x}) = [\cos(2\pi \mathbf{B}\mathbf{x}), \sin(2\pi \mathbf{B}\mathbf{x})]$$

where $\mathbf{B}$ is a matrix of frequencies sampled from a Gaussian $\mathcal{N}(0, \sigma^2)$. The bandwidth $\sigma$ controls the frequency range of representable signals — higher $\sigma$ enables finer detail.

This is essentially the same mechanism used for positional encodings in **Neural Radiance Fields (NeRF)** — applying Fourier features to $(x,y,z,\theta,\phi)$ coordinates to enable the MLP to represent high-frequency view-dependent appearance.

## 3D Shape Representations

INRs enable powerful 3D representations without the memory cost and resolution limitations of voxel grids or meshes.

### Occupancy Networks

**Occupancy networks** (Mescheder et al., 2019) learn a function:

$$f_\theta : \mathbb{R}^3 \times \mathcal{Z} \to [0,1]$$

that predicts the probability that a 3D point $\mathbf{p}$ is inside a shape, conditioned on a latent code $\mathbf{z}$ (from an encoder). The 3D surface is the level set $\{\mathbf{p} : f_\theta(\mathbf{p}, \mathbf{z}) = 0.5\}$, extracted with **Marching Cubes**.

Occupancy networks represent shapes at arbitrary resolution from a fixed-size neural network — and can generalize across shape categories via the latent code.

### Signed Distance Functions (SDFs)

**Neural SDFs** represent 3D geometry as the signed distance to the nearest surface — positive outside, negative inside, zero on the surface:

$$f_\theta(\mathbf{x}) = d(\mathbf{x}, \partial\mathcal{S}) \cdot \text{sign}$$

**DeepSDF** (Park et al., 2019) learns a neural SDF conditioned on a latent shape code, enabling shape completion, interpolation between shapes, and high-quality surface reconstruction. SDFs are differentiable and the gradient $\nabla f$ gives the surface normal direction — enabling efficient sphere-tracing for rendering.

**Eikonal loss** enforces the SDF property (unit gradient norm on the surface):

$$\mathcal{L}_{eikonal} = \mathbb{E}_{\mathbf{x}}[(|\nabla f_\theta(\mathbf{x})| - 1)^2]$$

## Physics-Informed INRs

SIREN's analytically available derivatives make it ideal for **Physics-Informed Neural Networks (PINNs)** — where the loss function includes residuals of partial differential equations:

```python
def physics_loss(model, coords, laplacian_target):
    coords.requires_grad_(True)
    u = model(coords)
    
    # Compute first-order gradients
    grad_u = torch.autograd.grad(
        u, coords, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Compute Laplacian (sum of second-order partial derivatives)
    laplacian = sum(
        torch.autograd.grad(
            grad_u[:, i], coords,
            grad_outputs=torch.ones_like(grad_u[:, i]),
            create_graph=True
        )[0][:, i]
        for i in range(coords.shape[1])
    )
    
    # PDE loss: force Laplacian to match target (e.g., Poisson equation)
    pde_residual = ((laplacian - laplacian_target) ** 2).mean()
    return pde_residual
```

SIREN + PINNs have been applied to solving Poisson equations, Navier-Stokes equations, wave equations, and Schrödinger equations — outperforming ReLU-based PINNs on problems requiring accurate second-order derivatives.

## Applications

### Neural Radiance Fields (NeRF)

**NeRF** (Mildenhall et al., 2020) represents a scene as an INR mapping 5D coordinates $(x,y,z,\theta,\phi)$ to volume density and view-dependent color — enabling novel view synthesis from a sparse set of input photographs. NeRF is perhaps the highest-profile application of INRs, and its success sparked enormous interest in coordinate networks for 3D scene representations.

### Image Compression

INRs can be used as compressed image representations: fit an INR to an image and store the network weights instead of pixel values. The compression ratio is determined by the ratio of network parameter count to pixel count.

**COIN** (Dupont et al., 2021) demonstrated INR-based image compression competitive with JPEG on low-resolution images. More advanced approaches using meta-learned initializations (fitting faster from a shared prior) bring INR compression to practical decode speeds.

### Video Representation

A video is a 3D signal (spatial + temporal). INRs can represent entire videos as space-time functions — enabling:

- **Continuous frame interpolation**: Query at arbitrary time coordinates for frame synthesis.
- **Compact storage**: Significant compression for videos with low motion.
- **Super-resolution**: Query at subpixel spatial coordinates.

**NeRV** (Neural Representations for Videos) demonstrated INRs for compact video encoding with fast decoding speeds.

### Medical Imaging and Scientific Fields

- **CT/MRI reconstruction**: INRs can represent volumetric medical scans in a continuous, queryable format — enabling arbitrary-resolution slice extraction and compressed storage.
- **Weather forecasting**: Neural fields representing atmospheric state as a continuous function of 4D space-time coordinates.
- **Molecular representations**: INRs encoding electron density fields for molecular property prediction.

## Limitations

- **Slow fitting**: Fitting an INR to a new signal typically requires hundreds or thousands of gradient steps — slow compared to a forward pass through a CNN encoder.
- **No generalization across instances**: A single INR fits one signal; generalizing across a dataset requires latent-conditioned architectures (occupancy networks, DeepSDF) or meta-learning approaches.
- **Training instability**: SIREN's sinusoidal activations require careful initialization and can diverge without it.
- **Inference cost**: Evaluating an INR requires a forward pass per query point — rendering a 1080p image requires 2 million forward passes unless batched efficiently.

Despite these challenges, implicit neural representations have become a foundational building block for 3D scene understanding, generative models, and physics simulation — demonstrating that continuous, differentiable functions can be a more natural representation for many signals than discrete grids.
