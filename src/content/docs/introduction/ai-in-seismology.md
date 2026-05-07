---
title: AI in Seismology
description: A comprehensive overview of AI applications in seismology, covering earthquake detection, phase picking, early warning systems, seismic tomography, ground motion prediction, and induced seismicity monitoring.
---

# AI in Seismology

Seismology has been transformed by deep learning — replacing rule-based signal processing pipelines with neural networks that detect and characterize earthquakes with greater sensitivity and speed. From automated phase picking that processes terabytes of continuous waveform data to real-time earthquake early warning systems, AI enables seismologists to extract information from seismic records at scales previously impossible.

## Traditional vs. AI-Based Seismic Processing

Classical seismology relies on:
- **STA/LTA detectors**: short-term/long-term average ratio for event detection
- **AIC pickers**: Akaike Information Criterion for P- and S-wave arrival times
- **Template matching**: cross-correlation against known event waveforms

These methods struggle with: low signal-to-noise events, overlapping phases, and the volume of continuous data from dense seismic networks. Deep learning addresses all three.

## Phase Picking with PhaseNet

Phase picking — identifying the precise arrival times of P-waves and S-waves — is the most labor-intensive step in seismic analysis. PhaseNet uses a U-Net architecture operating on 3-component waveform data:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PhaseNet(nn.Module):
    """U-Net for seismic phase picking.
    Input: (B, 3, T) — 3-component waveform, T samples at 100 Hz
    Output: (B, 3, T) — probability of P-arrival, S-arrival, noise at each sample
    """
    def __init__(self, channels=(8, 16, 32, 64, 128)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        in_ch = 3
        for out_ch in channels:
            self.encoders.append(PhaseNetBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool1d(2))
            in_ch = out_ch

        for i, out_ch in enumerate(reversed(channels[:-1])):
            self.upsamples.append(nn.ConvTranspose1d(channels[-1-i], out_ch, kernel_size=2, stride=2))
            self.decoders.append(PhaseNetBlock(out_ch * 2, out_ch))

        self.final = nn.Conv1d(channels[0], 3, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        skips = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            x = up(x)
            x = dec(torch.cat([x, skips[i+1]], dim=1))
        return torch.softmax(self.final(x), dim=1)
```

PhaseNet achieves F1 > 0.98 on standard benchmarks (INSTANCE, STEAD) — far exceeding manual picking consistency.

## Earthquake Detection with EQTransformer

EQTransformer combines a Transformer encoder with a PhaseNet-style U-Net decoder, adding a dedicated **detection head** alongside the phase picking heads:

```python
from seisbench.models import EQTransformer

# Load pretrained model (SeisBench model hub)
model = EQTransformer.from_pretrained("original")
model.eval()

# Annotate a 60-second waveform
import obspy
import seisbench.util as sbu

stream = obspy.read("waveform.mseed")   # 3-component miniseed
annotations = model.annotate(stream)
# Returns: stream with P-probability, S-probability, detection-probability traces

# Extract picks
picks, detections = model.classify(stream, P_threshold=0.3, S_threshold=0.3)
for pick in picks:
    print(f"{pick.phase}: {pick.start_time}, confidence={pick.peak_value:.3f}")
```

EQTransformer generalizes across tectonic environments with minimal fine-tuning — a property called **transfer learning in seismology**.

## Continuous Waveform Processing with SeisBench

SeisBench provides a unified framework for seismic ML:

```python
import seisbench.data as sbd
import seisbench.models as sbm
from torch.utils.data import DataLoader

# Download STEAD dataset (1.2M seismic traces)
dataset = sbd.STEAD()
train, dev, test = dataset.train_dev_test()

# Fine-tune PhaseNet on local data
model = sbm.PhaseNet.from_pretrained("stead")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loader = DataLoader(train, batch_size=256, shuffle=True)
for batch in train_loader:
    pred = model(batch["X"].to("cuda"))
    loss = F.binary_cross_entropy(pred, batch["y"].to("cuda"))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Earthquake Early Warning

Earthquake early warning (EEW) systems must estimate earthquake magnitude and location within seconds of P-wave detection — before destructive S-waves and surface waves arrive. ML models outperform classical magnitude scaling relations:

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Features extracted from first 3 seconds of P-wave
# (peak displacement, predominant period, Pd, tau_c, etc.)
def extract_ew_features(p_wave_snippet: np.ndarray, dt: float = 0.01) -> np.ndarray:
    pd = np.max(np.abs(p_wave_snippet))
    velocity = np.diff(p_wave_snippet) / dt
    acc = np.diff(velocity) / dt
    tau_c = 2 * np.pi * np.sqrt(
        np.sum(p_wave_snippet**2) / (np.sum(velocity**2) + 1e-10)
    )
    return np.array([np.log10(pd + 1e-10), tau_c, np.std(acc), np.max(np.abs(acc))])

# Train on historical catalog
mag_model = GradientBoostingRegressor(n_estimators=300, max_depth=4)
mag_model.fit(X_train_features, y_train_magnitudes)
```

Deep learning EEW systems (PLUM, GPD, MyShake) operate on smartphones and IoT seismometers, creating crowd-sourced early warning networks.

## Seismic Tomography with Neural Networks

Seismic tomography inverts arrival time residuals to image 3D subsurface velocity structure. Neural network approaches:

- **Physics-informed neural networks (PINNs)**: solve the eikonal equation for wave travel times with neural networks, enabling gradient-based tomographic inversion
- **Variational inference**: Bayesian neural networks provide uncertainty estimates on velocity models
- **Full Waveform Inversion (FWI) with ML**: surrogate models replace expensive forward simulations in iterative inversion

```python
import torch
import torch.nn as nn

class EikonalPINN(nn.Module):
    """Physics-informed NN for eikonal equation: |∇T|² = s(x)²"""
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)   # predicted travel time T(x, y)

    def eikonal_residual(self, xy: torch.Tensor, slowness: torch.Tensor) -> torch.Tensor:
        xy.requires_grad_(True)
        T = self.forward(xy)
        grads = torch.autograd.grad(T.sum(), xy, create_graph=True)[0]
        grad_norm_sq = (grads ** 2).sum(dim=-1, keepdim=True)
        return (grad_norm_sq - slowness**2) ** 2
```

## Ground Motion Prediction

Predicting peak ground acceleration (PGA) and spectral acceleration from source parameters (magnitude, depth, distance) and site conditions (Vs30) uses neural network ground motion models (NGMMs):

```python
import torch.nn as nn

class NeuralGMM(nn.Module):
    """Neural network ground motion prediction model."""
    def __init__(self, n_features: int = 8):
        super().__init__()
        # Inputs: Mw, Rrup, Rjb, depth, Vs30, fault_type, Ztor, dip
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        # Predict mean and aleatory uncertainty jointly
        self.mean_head = nn.Linear(64, 1)
        self.sigma_head = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def forward(self, x):
        h = self.backbone(x)
        return self.mean_head(h), self.sigma_head(h)  # log(PGA), sigma
```

NGMMs trained on NGA-West2 and KNET databases match or exceed the performance of empirical GMPEs on held-out data.

## Induced Seismicity Monitoring

Wastewater injection, geothermal energy, and hydraulic fracturing induce seismicity. ML models:

- Classify natural vs. induced earthquakes from waveform features and tectonic context
- Predict seismicity rates from injection parameters using recurrent neural networks
- Detect fault activation using distributed acoustic sensing (DAS) + 1D CNNs on fiber optic data

## Applications Summary

| Application | Model | Benchmark Metric |
|---|---|---|
| Phase picking | PhaseNet (U-Net) | F1 > 0.98 on STEAD |
| Earthquake detection | EQTransformer | AUC > 0.99 |
| Magnitude estimation (EEW) | Gradient boosting / CNN | MAE < 0.3 Mw units |
| Seismic tomography | PINN + adjoint | Comparable to FWI |
| Ground motion prediction | Neural GMM | Lower residuals than GMPEs |
| Induced vs. natural | Random forest / CNN | Accuracy > 90% |

## Summary

AI has become indispensable in modern seismology — enabling detection of earthquakes at signal-to-noise ratios previously invisible to classical detectors, processing continental-scale seismic networks in real time, and advancing earthquake early warning systems that could save thousands of lives. Libraries like SeisBench and pretrained models (PhaseNet, EQTransformer, GPD) provide accessible entry points for seismologists, while physics-informed neural networks bridge data-driven learning with the governing equations of wave propagation. As global seismic networks densify and DAS arrays expand, the role of AI in monitoring Earth's seismic activity will continue to grow.
