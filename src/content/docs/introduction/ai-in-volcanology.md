---
title: AI in Volcanology
description: An exploration of how artificial intelligence and machine learning are transforming volcanology — from seismic event detection and eruption forecasting to satellite-based deformation monitoring and lava flow simulation.
---

# AI in Volcanology

Volcanic systems are among the most complex and dynamic geological phenomena on Earth — a cascade of interacting processes spanning magma ascent, ground deformation, seismicity, gas flux, and surface changes. **Artificial intelligence** is transforming every facet of volcanic monitoring, enabling faster detection of precursory signals, more accurate eruption forecasts, and real-time hazard mapping that can save lives.

## Seismic Event Classification

Volcanoes generate distinctive seismic signals: volcano-tectonic (VT) earthquakes, long-period (LP) events, tremor, hybrid events, and explosion quakes. Classifying these automatically is critical for real-time monitoring of thousands of daily events.

```python
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class VolcanoSeismicCNN(nn.Module):
    """1D CNN for seismic waveform classification."""

    def __init__(self, num_classes: int = 6, sample_rate: int = 100):
        super().__init__()
        self.spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=256,
            hop_length=64,
            n_mels=64,
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) — raw seismic trace
        spec = self.spec(waveform).unsqueeze(1)     # (B, 1, F, T)
        features = self.encoder(spec).flatten(1)
        return self.classifier(features)


# Classes: VT, LP, tremor, hybrid, explosion, noise
model = VolcanoSeismicCNN(num_classes=6)
```

Deep learning classifiers achieve >95% accuracy on benchmark datasets (e.g., STEAD, Etna seismic catalog), far exceeding human experts in throughput.

## Tremor and Harmonic Analysis

Harmonic tremor — sustained periodic ground vibration — often precedes eruptions. Recurrent models can detect subtle tremor onset in continuous waveform data:

```python
from torch.nn import LSTM


class TremorDetector(nn.Module):
    def __init__(self, input_features: int = 64, hidden: int = 128):
        super().__init__()
        self.lstm = LSTM(input_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)   # binary: tremor / no-tremor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) — spectrogram frames
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)  # (B, T) — framewise probability
```

## InSAR Deformation Monitoring

Interferometric Synthetic Aperture Radar (InSAR) measures millimeter-scale surface deformation from satellite radar images. AI accelerates the processing pipeline:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def detect_deformation_anomalies(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    threshold: float = 0.4,
) -> np.ndarray:
    """
    Simple deformation anomaly detector using RF on pixel features.
    interferogram: (H, W) — unwrapped phase in radians (displacement proxy)
    coherence: (H, W) — InSAR coherence [0, 1]
    """
    H, W = interferogram.shape
    # Feature: local phase stats + coherence
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(interferogram, size=5)
    local_std = np.sqrt(uniform_filter(interferogram ** 2, size=5) - local_mean ** 2)
    gradient_mag = np.sqrt(
        np.gradient(interferogram, axis=0) ** 2 +
        np.gradient(interferogram, axis=1) ** 2
    )
    X = np.stack([interferogram, local_mean, local_std, gradient_mag, coherence], axis=-1)
    X = X.reshape(-1, 5)
    # In practice: use pre-trained model
    mask = (np.abs(interferogram) > threshold) & (coherence > 0.3)
    return mask
```

Convolutional networks applied to InSAR time series have detected pre-eruptive inflation at Kilauea, Etna, and Fagradalsfjall months before eruptions.

## SO₂ Flux Estimation

Volcanic SO₂ is a key eruption precursor. UV camera systems and satellite spectrometers (OMI, TROPOMI) measure SO₂ column density; ML models convert these to mass flux estimates:

```python
import numpy as np


def estimate_so2_flux(
    column_density_map: np.ndarray,   # (H, W) in Dobson units
    wind_speed: float,                 # m/s (from NWP model)
    plume_width_pixels: int,
    pixel_size_m: float,
) -> float:
    """Estimate SO2 flux (kg/s) from UV camera image."""
    # Integrate across plume cross-section
    cross_section = column_density_map[:, plume_width_pixels // 2]
    cross_section_kg_m2 = cross_section * 2.8577e-3   # DU -> kg/m^2
    flux = cross_section_kg_m2.sum() * pixel_size_m * wind_speed
    return float(flux)
```

Neural networks trained on labeled emission episodes can directly predict degassing style (passive, explosive, or effusive) from SO₂ spatial patterns.

## Eruption Forecasting

Eruption forecasting integrates multiple precursory signals using ensemble ML:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_eruption_forecast_model():
    """
    Feature vector per time window:
    [seismicity_rate, dominant_frequency, max_amplitude,
     deformation_mm, so2_flux_kg_s, thermal_anomaly_m2,
     days_since_last_eruption, ...]
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
        )),
    ])
    return pipeline
```

Probabilistic forecasting using Bayesian networks or Monte Carlo dropout provides calibrated uncertainty estimates essential for civil protection decisions.

## Lava Flow Simulation

AI accelerates physics-based lava flow simulators (e.g., MOLASSES, PyFLOWGO) by learning surrogate models:

```python
import torch
import torch.nn as nn


class LavaFlowSurrogate(nn.Module):
    """
    Surrogate model replacing computationally expensive CFD simulation.
    Input: DEM patch + eruption parameters (vent location, effusion rate, viscosity)
    Output: inundation probability map
    """

    def __init__(self, dem_size: int = 128, param_dim: int = 5):
        super().__init__()
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 * 16 * 16 + param_dim, 1024), nn.ReLU(),
            nn.Linear(1024, dem_size * dem_size), nn.Sigmoid(),
        )
        self.dem_size = dem_size

    def forward(self, dem: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        dem_feat = self.dem_encoder(dem).flatten(1)
        combined = torch.cat([dem_feat, params], dim=1)
        return self.decoder(combined).reshape(-1, self.dem_size, self.dem_size)
```

Surrogate models deliver probabilistic lava flow hazard maps in seconds versus hours for high-resolution CFD simulations.

## AI Applications in Volcanology

| Application | Method | Volcano | Outcome |
|---|---|---|---|
| Seismic classification | CNN + LSTM | Etna, Ruapehu | >95% accuracy |
| Eruption short-term forecast | LSTM + Bayesian | Kīlauea | 72-hr forecast |
| InSAR deformation detection | CNN time series | Santorini | Pre-eruption detection |
| Lava flow mapping | U-Net (satellite) | Nyiragongo | Real-time mapping |
| SO₂ flux estimation | Random Forest | Stromboli | ±15% accuracy |
| Pyroclastic density current | Physics-informed NN | Merapi | Runout prediction |

## Ethical Dimensions

AI-powered volcano monitoring raises several societal questions:

- **False alarms**: spurious eruption warnings cause costly evacuations and erode public trust
- **Equity**: high-tech monitoring concentrates in wealthy nations; active volcanoes in developing countries remain poorly monitored
- **Model accountability**: automated alert systems require clear chains of human authority for evacuation decisions

## Summary

AI is now embedded throughout the volcanological workflow — from raw seismic waveform classification and InSAR deformation detection to multi-parameter eruption forecasting and rapid lava flow hazard mapping. The combination of deep learning for pattern recognition in high-volume monitoring data, surrogate models for fast simulation, and probabilistic frameworks for uncertainty quantification is creating a new generation of near-real-time volcanic hazard systems capable of providing the hours-to-days advance warning that can make the difference between life and death.
