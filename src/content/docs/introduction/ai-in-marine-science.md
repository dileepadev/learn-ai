---
title: AI in Marine Science
description: Discover how artificial intelligence is transforming ocean science — from automated species identification in underwater imagery and acoustic detection of marine mammals to coral reef health monitoring, illegal fishing detection, fisheries stock assessment, and machine learning for ocean climate modeling.
---

Covering 71% of Earth's surface and containing 97% of its water, the ocean remains one of the least monitored environments on the planet. Traditional marine science relies on expensive ship-based surveys, limited acoustic sensors, and manual analysis of underwater footage — methods that can only sample a tiny fraction of the ocean at any time. AI is fundamentally changing this by enabling automated analysis of the growing torrent of data from satellite networks, autonomous vehicles, acoustic arrays, and remote sensors.

## Underwater Species Identification

Marine biodiversity surveys traditionally require expert taxonomists to manually review hours of underwater video — slow, expensive, and impossible to scale to the global ocean. Computer vision models now classify marine species with accuracy matching or exceeding experts.

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class MarineSpeciesClassifier:
    """
    Fine-tuned vision classifier for marine species identification.
    
    Key challenges vs. terrestrial vision:
    - Color distortion from water column absorption (blues/greens dominant)
    - Low contrast and backscatter from particles
    - Camouflage and highly variable orientation
    - Long-tailed distribution: thousands of rare species, few common ones
    - Domain shift between geographic regions and depths
    """
    
    COMMON_SPECIES = [
        "Acanthaster planci",      # Crown-of-thorns starfish (coral predator)
        "Acropora cervicornis",    # Staghorn coral
        "Amphiprion ocellaris",    # Clownfish
        "Chelonia mydas",          # Green sea turtle
        "Mobula birostris",        # Giant oceanic manta ray
        "Thunnus thynnus",         # Atlantic bluefin tuna
        # ... 1000+ additional species
    ]
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # EfficientNet-B4 fine-tuned on FishNet / iNaturalist marine datasets
        self.model = models.efficientnet_b4(pretrained=False)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, len(self.COMMON_SPECIES)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()
        
        # Underwater-specific preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(380),
            transforms.CenterCrop(380),
            # Underwater images often have blue/green cast — channel normalization
            # compensates for wavelength-dependent water absorption
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3, 0.4, 0.45],   # underwater-tuned means
                std=[0.18, 0.19, 0.20]
            )
        ])

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 5) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        return [
            {
                "species": self.COMMON_SPECIES[idx],
                "confidence": prob.item(),
                "is_invasive": self._check_invasive(self.COMMON_SPECIES[idx])
            }
            for prob, idx in zip(top_probs[0], top_indices[0])
        ]

    def _check_invasive(self, species: str) -> bool:
        INVASIVE = {"Pterois volitans", "Carcinus maenas", "Mnemiopsis leidyi"}
        return species in INVASIVE
```

**FishNet**, the largest labeled underwater fish dataset, contains over 86,000 images of 17,357 fish categories. The **CoralNet** platform automates coral cover estimation from benthic survey photos, processing millions of quadrats per year across global coral monitoring networks.

## Acoustic Detection of Marine Mammals

Sound travels efficiently through water — whales, dolphins, and fish communicate across ocean basins with acoustic signals. Passive acoustic monitoring (PAM) networks deploy hydrophones that record continuously, generating terabytes of audio data that must be searched for biological signals:

```python
import numpy as np
import librosa
import torch
import torch.nn as nn

def extract_spectrogram_features(
    audio_path: str,
    sample_rate: int = 2000,    # sufficient for most cetacean calls (0-1000 Hz)
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
    duration: float = 5.0       # 5-second analysis window
) -> np.ndarray:
    """
    Convert hydrophone audio to mel spectrogram for CNN classification.
    
    Marine mammal vocalizations:
    - Blue whale: 10-40 Hz infrasound, detectable 1000+ km
    - Humpback whale: complex songs 20 Hz–24 kHz, seasonal
    - Sperm whale: clicks 0.1–30 kHz, used for echolocation
    - Dolphin: whistles 2–20 kHz, clicks up to 150 kHz
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    
    # Mel spectrogram (dB scale)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db  # (n_mels, time_frames)


class WhaleCallDetector(nn.Module):
    """
    CNN for detecting and classifying whale vocalizations in hydrophone data.
    
    Trained on annotated PAM recordings from NOAA, MBARI, and academic 
    monitoring networks. Deployed in real-time on autonomous gliders
    and fixed hydrophone arrays (e.g., NEPTUNE, DMON).
    """
    CALL_TYPES = [
        "background_noise",
        "blue_whale_A",        # 17-22 Hz tonal call
        "blue_whale_B",        # 80-90 Hz FM sweep
        "fin_whale_20hz",      # 20 Hz pulse
        "humpback_song",       # complex broadband
        "sperm_whale_click",   # echolocation click
        "orca_call",           # pod-specific calls
        "vessel_noise"         # anthropogenic (important to distinguish)
    ]
    
    def __init__(self, n_mels: int = 64, n_classes: int = 8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, time_frames) — spectrogram input"""
        return self.classifier(self.backbone(x))
```

## Coral Reef Health Assessment

Coral reefs cover less than 0.1% of the ocean floor but host 25% of all marine species. They are under severe threat from bleaching events caused by thermal stress. AI enables large-scale monitoring:

```python
import torch
import segmentation_models_pytorch as smp

def build_coral_segmentation_model(num_classes: int = 15) -> nn.Module:
    """
    Semantic segmentation model for coral reef benthic surveys.
    
    Classes typically include:
    - Hard coral (multiple morphologies: branching, massive, encrusting, tabular)
    - Soft coral
    - Bleached coral (critical for bleaching event monitoring)
    - Dead coral (rubble, turf algae on dead skeleton)
    - Macroalgae (indicator of reef decline)
    - Coralline algae (indicator of healthy reef)
    - Sand, rubble, rock (substrate types)
    
    Input: underwater photos from towed cameras, AUVs, or diver surveys
    Training data: CoralNet annotations, AIMS LTMP database, XL Catlin Survey
    """
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b5",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model


def estimate_bleaching_severity(
    coral_mask: np.ndarray,
    rgb_image: np.ndarray
) -> dict[str, float]:
    """
    Estimate bleaching severity from segmented coral regions.
    
    Bleached coral appears pale white/yellow due to loss of symbiotic algae.
    Uses color-based analysis of segmented coral pixels.
    
    Returns: bleaching percentage and severity category per the NOAA scale
    (0-10%: Watch, 10-30%: Warning, 30-60%: Alert 1, >60%: Alert 2)
    """
    # Extract pixels classified as coral (hard coral classes)
    coral_pixels = rgb_image[coral_mask > 0]
    
    if len(coral_pixels) == 0:
        return {"coral_cover_pct": 0.0, "bleaching_pct": 0.0, "severity": "No coral"}
    
    # Bleaching heuristic: low saturation + high brightness = bleached
    # Convert to HSV for easier saturation analysis
    import colorsys
    saturations = []
    for r, g, b in coral_pixels:
        _, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        saturations.append(s)
    
    mean_saturation = np.mean(saturations)
    # Low saturation (< 0.25) indicates bleaching for healthy coral color range
    bleaching_pct = (np.array(saturations) < 0.25).mean() * 100
    
    severity_map = [(10, "Watch"), (30, "Warning"), (60, "Alert 1"), (100, "Alert 2")]
    severity = next((s for t, s in severity_map if bleaching_pct <= t), "Alert 2")
    
    total_pixels = np.prod(rgb_image.shape[:2])
    coral_cover_pct = len(coral_pixels) / total_pixels * 100
    
    return {
        "coral_cover_pct": round(coral_cover_pct, 1),
        "bleaching_pct": round(bleaching_pct, 1),
        "mean_saturation": round(mean_saturation, 3),
        "severity": severity
    }
```

## Illegal Fishing Detection

**Global Fishing Watch** uses AIS (Automatic Identification System) vessel tracking data combined with ML to detect illegal, unreported, and unregulated (IUU) fishing:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def detect_dark_vessels(ais_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify 'dark vessel' patterns: vessels that disable AIS transmitters
    (often indicating IUU fishing or sanctions evasion).
    
    Features derived from vessel movement and AIS gaps:
    - AIS gap duration and location
    - Speed and heading changes around gap
    - Proximity to known fishing grounds
    - Historical compliance record
    - Flag state risk score
    """
    features = [
        "gap_duration_hours",          # AIS signal gap length
        "gap_distance_from_port_km",   # how far from port during gap
        "pre_gap_speed_knots",         # slowing before gap = turning off AIS
        "post_gap_speed_knots",        # speed resuming after gap
        "gap_lat",                     # latitude (EEZ membership matters)
        "gap_lon",
        "vessel_length_m",
        "flag_state_risk_score",       # 0-1, from PSMA Port State Index
        "fishing_vessel_type_encoded",
        "days_at_sea_this_trip",
        "historical_ais_compliance_rate"
    ]
    
    # Model trained on labeled dark vessel events from enforcement actions
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                    random_state=42)
    # model.fit(X_train, y_train)  # in practice, pre-trained
    
    X = ais_df[features].fillna(0)
    ais_df["iuu_risk_score"] = model.predict_proba(X)[:, 1]
    ais_df["high_risk"] = ais_df["iuu_risk_score"] > 0.7
    
    return ais_df.sort_values("iuu_risk_score", ascending=False)
```

## Ocean Climate Modeling

Physical ocean models (like MOM6, NEMO) resolve dynamics at 1/12° spatial resolution — expensive to run globally at higher resolution. ML surrogates learn to approximate fine-scale ocean dynamics from coarse-resolution inputs:

- **Ocean eddy detection**: CNNs identify mesoscale eddies (200-300 km diameter) from sea surface height satellite data — critical for understanding heat transport and biological productivity
- **Sea surface temperature forecasting**: LSTMs and transformers forecast SST 7–30 days ahead, outperforming persistence baselines for El Niño prediction
- **Deep ocean current prediction**: Graph Neural Networks model the global thermohaline circulation from Argo float profiles
- **Plastic drift tracking**: ML-enhanced particle tracking models predict where ocean plastic accumulates given historical current patterns and wind data

Marine AI deployments face unique challenges: sensor biofouling degrades data quality over time, underwater acoustic noise varies with shipping traffic and weather, and the taxonomic expertise needed to curate labeled datasets is in short supply. Citizen science platforms like **iNaturalist** and the **Zooniverse** are helping address the annotation bottleneck by engaging recreational divers in labeling marine life observations — creating training datasets that span geographic ranges and seasons far beyond what scientific surveys can achieve alone.
