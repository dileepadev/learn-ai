---
title: AI for Biodiversity Conservation
description: Explore how artificial intelligence is transforming biodiversity conservation — from automated species identification and population monitoring to habitat mapping, invasive species detection, anti-poaching surveillance, and ecological forecasting under climate change.
---

Earth is experiencing its sixth mass extinction event, with species disappearing at rates 100–1,000 times the natural background rate. Monitoring and protecting biodiversity at the scale needed requires processing satellite imagery, acoustic recordings, camera trap photos, environmental DNA samples, and citizen science observations across millions of hectares — far beyond human capacity. AI is transforming conservation biology by automating species identification, enabling continuous large-area monitoring, predicting threats before they manifest, and optimizing where scarce conservation resources have the greatest impact.

## Species Identification

### Computer Vision for Camera Traps

Camera traps generate millions of images per year. Manual review by ecologists is a severe bottleneck — a single large-scale deployment can produce more images than a research team can process in years. Deep learning automates this at high accuracy.

**MegaDetector** (Microsoft AI for Earth) is a YOLO-based model detecting animals, humans, and vehicles in camera trap images regardless of species — providing a preprocessing filter that eliminates empty frames and separates animals from humans for privacy-preserving processing:

```python
from PIL import Image
from megadetector.detection.run_detector import load_and_run_detector

# Run MegaDetector on a batch of camera trap images
results = load_and_run_detector(
    model_file="md_v5a.0.0.pt",
    image_file_names=["camera_trap_001.jpg", "camera_trap_002.jpg"],
    output_dir="detections/",
    threshold=0.2,
)

for result in results:
    for detection in result["detections"]:
        print(f"Category: {detection['category']}, Confidence: {detection['conf']:.2f}")
```

Species-level classification follows detection: EfficientNet and Vision Transformer models trained on labeled camera trap datasets achieve >90% top-1 accuracy across hundreds of species when fine-tuned on regional data.

### Acoustic Species Identification

Bioacoustics monitoring uses microphones and hydrophones to capture calls from birds, bats, whales, frogs, and insects. Neural networks classify acoustic signals from passive recorders deployed across ecosystems.

**BirdNET** (Cornell Lab / Chemnitz University) identifies bird species from audio recordings using a custom ResNet architecture trained on millions of labeled recordings:

```python
from birdnet_analyzer import analyze

results = analyze.analyze_file(
    input_path="forest_recording_dawn.wav",
    min_confidence=0.7,
    sensitivity=1.0,
)

for detection in results:
    print(f"{detection['common_name']}: {detection['confidence']:.2%} at {detection['start_time']}s")
```

Acoustic monitoring detects cryptic species that cameras miss, enables 24/7 monitoring without human observers, and captures species active at night or in dense vegetation.

### Environmental DNA (eDNA) Metabarcoding

Species shed DNA into their environment — water, soil, air. Collecting environmental samples and sequencing DNA enables non-invasive biodiversity surveys. ML classifies sequences against reference databases:

- **Taxonomic classification**: transformer models trained on DNA sequences classify reads to species level, handling novel sequences with uncertainty quantification
- **Community composition**: ordination models (UMAP, PCA on count matrices) reveal ecosystem structure and detect changes in species assemblages
- **Occupancy estimation**: hierarchical Bayesian models combine eDNA detection probability with occupancy likelihood to estimate true presence from imperfect sampling

## Population Monitoring and Abundance Estimation

### Individual Re-Identification

Many species can be individually identified from natural markings — whale flukes, jaguar rosettes, elephant ear venation, lion whisker spot patterns. Deep metric learning enables large-scale re-ID without human annotation of every individual.

The approach mirrors face recognition: a CNN backbone extracts embeddings, contrastive or triplet loss trains embeddings to cluster by individual, and nearest-neighbor search matches new photos against a database of known individuals.

```python
import torch
import torch.nn as nn
from torchvision import models

class WildlifeReID(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.efficientnet_b3(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embedder = nn.Linear(backbone.classifier[1].in_features, embedding_dim)

    def forward(self, x):
        features = self.features(x).flatten(1)
        return nn.functional.normalize(self.embedder(features), dim=1)

# Training with triplet loss for individual re-identification
triplet_loss = nn.TripletMarginLoss(margin=0.2)
```

**HotSpotter** and **Wildlife Insights** (Google) deploy these approaches for cheetahs, sharks, dolphins, and other species, enabling population size estimation (mark-recapture statistics) from photo-ID data contributed by tourists, researchers, and citizen scientists.

### Aerial and Satellite Counting

Drone and satellite imagery enable population counts across entire habitats without disturbing animals:

- **Penguin colonies**: CNNs count individual penguins in satellite imagery of Antarctic rookeries, enabling global population monitoring without field expeditions
- **Elephant censuses**: object detection models on aerial drone footage count elephants across savannas faster and more accurately than aerial survey teams
- **Coral reef health**: semantic segmentation maps live coral, algae, bleached coral, and substrate from drone imagery, tracking reef health across vast areas

```python
# Fine-tuning YOLO for wildlife counting from drone imagery
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="wildlife_aerial.yaml",
    epochs=100,
    imgsz=1280,        # High resolution for small objects
    batch=16,
    augment=True,
    degrees=15,        # Rotation augmentation for aerial perspective
    flipud=0.5,
    mosaic=1.0,
)
```

## Habitat Mapping and Landscape Analysis

### Remote Sensing Classification

Multispectral and hyperspectral satellite imagery enables landscape-scale habitat mapping. Transformer models (SegFormer, Prithvi foundation model by IBM/NASA) process multi-band imagery to produce land cover and habitat quality maps:

- **Land cover classification**: forests, wetlands, grasslands, degraded habitats, urban areas
- **Habitat connectivity**: graph neural networks model wildlife corridor permeability between habitat patches
- **Deforestation detection**: change detection models compare temporal image pairs to identify forest loss within days of occurrence

The **Prithvi** geospatial foundation model is pre-trained on Harmonized Landsat Sentinel-2 imagery and fine-tuned for biodiversity-relevant tasks:

```python
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

# Prithvi geospatial foundation model for habitat mapping
processor = AutoImageProcessor.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")
model = AutoModelForSemanticSegmentation.from_pretrained(
    "ibm-nasa-geospatial/Prithvi-100M-sen1floods11",
    num_labels=len(habitat_classes),
    ignore_mismatched_sizes=True,
)
```

### Species Distribution Modeling

MaxEnt and its neural network successors predict where species can occur based on environmental variables (temperature, precipitation, vegetation indices, elevation):

- **BioCLIM / MaxEnt**: classical presence-only SDMs using environmental envelopes
- **Deep SDMs**: neural networks that learn non-linear relationships between environmental variables and occurrence probability, handling spatial autocorrelation explicitly
- **Climate change projections**: SDMs applied to future climate scenarios project habitat shifts under 1.5°C, 2°C, and 4°C warming pathways — identifying climate refugia and range contractions

## Invasive Species Detection

Early detection of invasive species before populations establish is critical. ML enables rapid screening:

- **Hyperspectral detection**: invasive plant species have distinct spectral signatures detectable from airborne or satellite sensors — CNN classifiers distinguish invasives from native vegetation at landscape scale
- **Acoustic detection**: invasive frogs (coqui in Hawaii) and insects (spotted lanternfly) have identifiable calls and sounds detectable with BirdNET-style classifiers
- **eDNA early warning**: eDNA sampling of waterways combined with ML classification detects invasive fish and amphibians before visual surveys can confirm presence

## Anti-Poaching and Wildlife Crime Detection

### PAWS: Protection Assistant for Wildlife Security

**PAWS** uses game-theoretic AI and historical patrol data to predict where poaching is most likely and optimize ranger patrol routes:

- Input: historical snare/poaching detection locations, terrain, patrol coverage maps, wildlife density
- Model: feature-based prediction of poaching risk using gradient boosting or random forests
- Output: patrol routes that maximize coverage of high-risk areas subject to ranger time and terrain constraints

Field deployments in Uganda, Cambodia, and Malaysia have demonstrated significant increases in snare detection rates compared to systematic or intuition-based patrol strategies.

### Real-Time Camera Surveillance

AI-powered camera networks at reserve boundaries detect human intrusion at night:

- Person detection and tracking (MegaDetector-style models) trigger alerts when humans are detected in restricted areas
- Gunshot detection from acoustic sensors uses CNN classifiers on spectrogram images to alert rangers within seconds
- Vehicle detection on access roads identifies suspicious nighttime activity

## Ecological Forecasting

### Phenology Prediction

Phenology — the timing of biological events (flowering, migration, breeding) — is shifting under climate change. ML models predict phenological events from satellite-derived vegetation indices (NDVI, EVI) and meteorological data:

- Bloom date prediction for pollinators — critical for scheduling conservation interventions
- Migration timing models for birds — predicting when peak counts will occur at key stopover sites
- Breeding season onset prediction for amphibians and insects — informing survey timing

### Extinction Risk Prediction

Graph neural networks on species trait databases predict extinction risk from life history traits, range size, habitat specialization, and threat exposure:

$$p(\text{threatened} \mid \text{traits}, \text{threats}) = \sigma(f_\theta(\mathbf{x}_{\text{traits}}, \mathbf{x}_{\text{threats}}))$$

These models identify data-deficient species likely to be threatened — prioritizing them for field assessments — and project how climate and land use change will shift extinction risk distributions across taxa.

## Citizen Science and Community Monitoring

Platforms like iNaturalist, eBird, and Merlin aggregate millions of biodiversity observations from volunteers worldwide. ML makes these platforms functional:

- **Computer vision identification**: iNaturalist's CV model identifies species from user-submitted photos, enabling non-experts to contribute quality data
- **Data quality filtering**: ML classifiers flag misidentifications, geographic outliers, and duplicate observations
- **Effort correction**: occupancy models account for uneven sampling effort across locations, times, and observers to produce unbiased abundance indices

## Summary

AI is enabling biodiversity conservation at scales and speeds previously impossible:

- **Species identification**: computer vision and acoustic models automate processing of millions of camera trap images and audio recordings
- **Population monitoring**: individual re-ID with metric learning and aerial counting with object detection track populations continuously
- **Habitat mapping**: satellite foundation models produce landscape-scale habitat quality and connectivity maps
- **Invasive species detection**: hyperspectral, acoustic, and eDNA approaches provide early warning before populations establish
- **Anti-poaching**: game-theoretic patrol optimization and real-time intrusion detection improve ranger effectiveness
- **Ecological forecasting**: phenology and extinction risk models anticipate threats before they manifest

As AI tools become more accessible and field sensor networks expand, the bottleneck in conservation is shifting from data collection to ecological interpretation and on-the-ground action — precisely where AI frees up human expertise to be most impactful.
