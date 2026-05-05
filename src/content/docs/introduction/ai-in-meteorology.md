---
title: AI in Meteorology
description: A comprehensive guide to AI applications in meteorology, covering neural weather prediction, nowcasting, extreme event detection, ensemble post-processing, and the shift from numerical to ML-based forecasting.
---

# AI in Meteorology

Weather forecasting has been transformed by AI — moving from purely physics-based **Numerical Weather Prediction (NWP)** models that require supercomputer-scale simulation to neural networks that achieve comparable or superior skill in seconds on a single GPU. Models like GraphCast, Pangu-Weather, and FourCastNet represent a paradigm shift in how humanity predicts the atmosphere.

## From NWP to Neural Weather Models

Traditional NWP solves the equations of fluid dynamics on a 3D atmospheric grid:

- **ECMWF IFS**: European Centre's flagship model, ~9 km horizontal resolution
- **GFS**: NOAA's Global Forecast System, ~13 km resolution
- **ICON**: Germany's operational model

These models take **hours** on thousand-core supercomputers to produce a 10-day forecast. AI models trained on decades of ERA5 reanalysis data can produce equivalent forecasts in **under one second** on a single GPU.

## GraphCast (DeepMind, 2023)

GraphCast uses a graph neural network to model the atmosphere as a multi-mesh graph spanning the globe:

- **Input**: two consecutive 6-hour ERA5 snapshots (237 variables × 37 pressure levels × 0.25° grid)
- **Architecture**: encoder (lat-lon grid → icosahedral mesh) → processor (18 GNN layers) → decoder (mesh → grid)
- **Output**: next 6-hour atmospheric state
- **Skill**: outperforms ECMWF IFS on 90% of test variables at 10-day lead time

```python
# GraphCast inference (conceptual — uses proprietary weights)
import jax
import numpy as np

def graphcast_forecast(
    current_state: np.ndarray,   # (lat, lon, levels, variables)
    prev_state: np.ndarray,
    n_steps: int = 40,           # 40 × 6h = 10 days
) -> list:
    states = [prev_state, current_state]
    for _ in range(n_steps):
        next_state = graphcast_model(states[-2], states[-1])
        states.append(next_state)
    return states[2:]            # return forecast steps only
```

## Pangu-Weather (Huawei, 2023)

Pangu-Weather uses a 3D Earth Transformer with hierarchical temporal resolution:

- **Architecture**: 3D Swin Transformer processing (pressure level, lat, lon) cubes
- **Trick**: separate models trained for 1h, 3h, 6h, 24h lead times; at inference, hierarchically combine them to minimize error accumulation
- **Key result**: first AI model to beat ECMWF IFS on all standard upper-air metrics

## FourCastNet (NVIDIA, 2022)

FourCastNet uses Fourier Neural Operators (FNO) for global forecasting:

```python
from torch_harmonics import RealSHT, InverseRealSHT

class SphericalFourierBlock(torch.nn.Module):
    """FourCastNet's spherical harmonic mixing block."""
    def __init__(self, hidden_dim: int, modes: int = 128):
        super().__init__()
        self.sht = RealSHT(720, 1440, grid="equiangular")
        self.isht = InverseRealSHT(720, 1440, grid="equiangular")
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (batch, channels, lat, lon)
        x_sht = self.sht(x)
        x_sht = torch.einsum("bcmn,cdmn->bdmn", x_sht, self.weight)
        return self.isht(x_sht)
```

FourCastNet is particularly fast — 45,000× faster than IFS — enabling large ensemble runs that were previously computationally infeasible.

## Nowcasting: Precipitation at Minutes Scale

Nowcasting predicts precipitation at 0–6 hour timescales using radar data, where traditional NWP provides poor skill.

### DeepMind's DGMR (Deep Generative Model of Rain)

```python
import torch
import torch.nn as nn

class DGMRGenerator(nn.Module):
    """Conditional GAN-based radar nowcasting."""
    def __init__(self, context_frames: int = 4, forecast_frames: int = 18):
        super().__init__()
        self.encoder = RadarEncoder(context_frames)
        self.sampler = SpatiotemporalSampler()
        self.decoder = ConvLSTMDecoder(forecast_frames)

    def forward(self, radar_context: torch.Tensor, z: torch.Tensor = None):
        # radar_context: (batch, time, H, W) — past 20 mins of radar
        if z is None:
            z = torch.randn(radar_context.shape[0], 8, *radar_context.shape[2:])
        h = self.encoder(radar_context)
        h = self.sampler(h, z)
        return self.decoder(h)          # (batch, 18, H, W) — 90 mins forecast
```

DGMR produces **probabilistic** forecasts — generating multiple plausible futures rather than a single deterministic prediction, better capturing convective uncertainty.

### MetNet-3 (Google, 2023)

MetNet-3 uses a large context window (2048 km) and predicts 0–24 hour precipitation at 1 km / 2-minute resolution — the most detailed precipitation nowcast model deployed operationally.

## Ensemble Post-Processing

Raw NWP ensembles are biased and uncalibrated. ML models correct these deficiencies:

```python
from sklearn.isotonic import IsotonicRegression
import numpy as np

def ensemble_model_output_statistics(ensemble_members: np.ndarray, obs: np.ndarray):
    """EMOS: fit NGR correction to ensemble forecasts."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    def crps_loss(params):
        a, b, c, d = params
        mu = a + b * ensemble_members.mean(axis=1)
        sigma = np.sqrt(c + d * ensemble_members.var(axis=1))
        # Continuous Ranked Probability Score
        z = (obs - mu) / sigma
        crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))
        return crps.mean()

    result = minimize(crps_loss, x0=[0, 1, 1, 0.1], method="Nelder-Mead")
    return result.x
```

**IMPROVER** (Met Office) uses U-Net architectures to correct gridded ensemble output, reducing systematic biases in precipitation and temperature forecasts.

## Extreme Event Detection and Attribution

AI classifies and attributes extreme weather events:

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

# Tropical cyclone intensity estimation from satellite
processor = AutoImageProcessor.from_pretrained("noaa/tc-intensity-classifier")
model = AutoModelForImageClassification.from_pretrained("noaa/tc-intensity-classifier")

def estimate_tc_intensity(satellite_image):
    inputs = processor(images=satellite_image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    # Returns Dvorak classification (T-number → wind speed)
    return logits.argmax(-1)
```

**Climate attribution**: counterfactual ML models estimate how much climate change increased the probability or intensity of a specific extreme event by comparing observed conditions against factual/counterfactual climate simulations.

## AI for Seasonal and Sub-Seasonal Forecasting

Standard NWP loses skill beyond ~2 weeks. AI approaches to extend range:

- **S2S forecasting**: LSTM and Transformer models predicting MJO (Madden-Julian Oscillation) phase — a key source of predictability at 2–8 week range
- **ENSO prediction**: CNN and hybrid models predicting El Niño events 12–18 months in advance with skill exceeding dynamical models
- **Teleconnection learning**: graph neural networks learning atmospheric teleconnection patterns that transmit climate signals across hemispheres

## Comparison of AI Weather Models

| Model | Architecture | Resolution | Max Lead Time | Open Weights |
|---|---|---|---|---|
| GraphCast | GNN (multi-mesh) | 0.25° | 10 days | ✅ |
| Pangu-Weather | 3D Swin Transformer | 0.25° | 7 days | ✅ |
| FourCastNet v2 | Spherical FNO | 0.25° | 10 days | ✅ |
| MetNet-3 | Axial Transformer | 1 km | 24 hours | ❌ |
| NeuralGCM | Hybrid NWP+ML | 1.4° | 10 days | ✅ |

## Challenges

- **Rare extremes**: training data contains few examples of record-breaking events; models may underestimate tails of distributions
- **Physical consistency**: neural forecasts can violate conservation of mass/energy; hybrid physics-ML models (NeuralGCM) address this
- **Uncertainty quantification**: deterministic AI models require post-processing for calibrated probabilistic output
- **Operational trust**: meteorological agencies require extensive verification before replacing operational NWP

## Summary

AI has fundamentally disrupted meteorology — enabling global weather prediction at NWP quality in milliseconds, probabilistic nowcasting at kilometer scales, and extended-range seasonal forecasts beyond the traditional deterministic horizon. GraphCast, Pangu-Weather, and FourCastNet demonstrate that neural networks trained on atmospheric reanalysis data can match or exceed decades of NWP development. The future lies in hybrid physics-ML models that combine the physical consistency of dynamical cores with the pattern-recognition power of large neural networks.
