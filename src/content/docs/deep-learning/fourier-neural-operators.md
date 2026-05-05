---
title: Fourier Neural Operators
description: Learn how Fourier Neural Operators (FNOs) learn mappings between infinite-dimensional function spaces, enabling neural networks to solve partial differential equations at any resolution — covering the spectral convolution layer, discretization invariance, GeoFNO and U-FNO variants, and applications in weather forecasting, fluid simulation, and materials science.
---

Classical neural networks map vectors to vectors. **Fourier Neural Operators (FNOs)**, introduced by Zongyi Li et al. (2021), map functions to functions — they learn operators that act on infinite-dimensional function spaces. This enables a single trained model to make predictions at arbitrary spatial resolutions, solving families of partial differential equations (PDEs) far more efficiently than traditional numerical solvers or standard neural networks.

## The Operator Learning Problem

Consider a PDE parameterized by an initial condition or coefficient field $a \in \mathcal{A}$ with solution $u \in \mathcal{U}$:

$$\mathcal{L}(a; u) = 0 \quad \text{on } D \subset \mathbb{R}^d$$

Traditional numerical solvers (finite element, finite difference) solve this for each specific $a$ from scratch — expensive when many solutions are needed (uncertainty quantification, design optimization, real-time control). Operator learning trains a neural network $\mathcal{G}_\theta: \mathcal{A} \rightarrow \mathcal{U}$ to approximate the solution operator directly:

$$\mathcal{G}_\theta(a) \approx u = \mathcal{G}^\dagger(a)$$

Once trained, new solutions are computed in milliseconds rather than hours.

## The Spectral Convolution Layer

The core insight of FNO is that the integral kernel of a general linear operator can be parameterized efficiently in Fourier space:

$$(\mathcal{K}(a; \phi) v_t)(x) = \int_D \kappa(x, y, a(x), a(y)) v_t(y) \, dy$$

In Fourier space, global convolution becomes pointwise multiplication — making it both efficient and expressive. The **spectral convolution** operation:

$$\mathcal{F}(\mathcal{K} v)(k) = R_\phi(k) \cdot \mathcal{F}(v)(k)$$

where $R_\phi \in \mathbb{C}^{d_v \times d_v}$ are learned complex weight matrices, one per retained Fourier mode. Only the lowest $k_{\max}$ modes are kept (low-frequency structure dominates PDE solutions):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: lift input to frequency domain, apply learned weights
    to low-frequency modes, transform back to physical space.
    
    Discretization-invariant: the same weights apply regardless of the
    spatial grid resolution — a function evaluated on a coarse grid
    and a fine grid will produce the same low-frequency predictions.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        """
        Args:
            modes1: number of Fourier modes to retain along dimension 1
            modes2: number of Fourier modes to retain along dimension 2
            (keep the lowest modes, which capture large-scale structure)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Learned complex weights for positive and negative frequencies
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication: (batch, in_ch, x, y) × (in_ch, out_ch, x, y)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, height, width)
        """
        batch_size = x.shape[0]
        
        # 1. Compute 2D FFT of input
        x_ft = torch.fft.rfft2(x)  # shape: (B, C, H, W//2+1)
        
        # 2. Multiply retained low-frequency modes by learned weights
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Top-left corner: positive x and y frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        
        # Bottom-left corner: negative x frequencies
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # 3. Inverse FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNOBlock2d(nn.Module):
    """
    One FNO layer: spectral convolution + local linear transform (skip connection)
    followed by activation. The local linear transform W captures local information
    that the global spectral convolution might miss.
    """
    
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.local_linear = nn.Conv2d(width, width, 1)   # 1×1 conv = linear transform
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral_conv(x) + self.local_linear(x)))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for solving 2D time-dependent PDEs.
    
    Architecture:
    1. Lift: project input channels to high-dimensional latent space
    2. Iterate: apply L Fourier layers
    3. Project: reduce back to output channels
    
    Input: discretized function a(x,y) + grid coordinates (x,y)
            shape: (batch, T_in + 2, H, W) — T_in historical timesteps + x,y coords
    Output: predicted function u(x,y,t+1, ..., t+T_out)
             shape: (batch, T_out, H, W)
    
    Trained on: Navier-Stokes, Darcy flow, wave equation, etc.
    Inference: 1000× faster than traditional FEM solvers
    """
    
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 in_channels: int = 12, out_channels: int = 10, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        self.fno_blocks = nn.Sequential(
            *[FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        Returns: (batch, out_channels, H, W)
        """
        x = self.lift(x)
        x = self.fno_blocks(x)
        return self.project(x)
```

## Key Properties

### Discretization Invariance

Unlike standard CNNs whose convolutional filters are tied to a fixed grid, FNO learns operators. A model trained on 64×64 resolution data can run inference at 256×256 — the spectral weights represent low-frequency basis functions that apply at any resolution. This is because the Fourier basis functions are defined on the continuous domain $[0, 1]^2$, not on a specific grid.

### Computational Complexity

For an $n$-point 2D grid with $k_{\max}$ retained modes:

$$\text{Cost per FNO layer} = \mathcal{O}(n \log n + k_{\max}^2)$$

For $k_{\max} \ll n$, this is dominated by the FFT — linear in $n$ up to logarithmic factors, vs. $\mathcal{O}(n^2)$ for full attention over grid points.

## Benchmark Results

On the Navier-Stokes equation (turbulent flow) benchmark:

| Method | Error | Runtime per solve |
| --- | --- | --- |
| Classical FEM solver | 0% (reference) | 2.5 hours |
| DeepONet | 0.35% | 0.5 seconds |
| U-Net | 0.24% | 0.8 seconds |
| FNO | 0.008% | **0.05 seconds** |

## Variants and Extensions

**GeoFNO**: Handles irregular geometries by learning a mapping from physical to latent Cartesian space where standard FFT applies. Enables FNO on airfoil shapes, medical imaging volumes, and molecular structures.

**U-FNO**: Adds U-Net-style skip connections between Fourier layers at multiple resolutions, improving accuracy on problems with multi-scale structure.

**SFNO** (Spherical FNO): Uses spherical harmonic transforms instead of Fourier transforms — enables global weather forecasting on the sphere. The basis of NVIDIA's FourCastNet model, which produces 10-day global weather forecasts in 2 seconds.

**FNO-3D**: Extends to space-time cubes by applying 3D FFTs, treating time as a spatial dimension — effective for video-like PDE data.

## Applications

- **Weather forecasting**: NVIDIA FourCastNet, trained on ERA5 reanalysis data, outperforms IFS (ECMWF's operational model) at 1/10,000 the compute cost per forecast
- **Computational fluid dynamics**: Surrogate models for aerodynamic design optimization — each FNO forward pass replaces a Navier-Stokes CFD run
- **Materials science**: Learning elastic field operators for composite materials (stress/strain fields from microstructure images)
- **Seismic inversion**: Mapping seismogram recordings to subsurface velocity models
- **Molecular dynamics**: Learning force fields from quantum chemistry data

FNOs represent a paradigm shift from fitting functions to learning operators — a natural inductive bias for physics simulation where the same governing equations apply across infinitely many initial and boundary conditions.
