---
title: AI in Paleontology
description: An exploration of how artificial intelligence is revolutionizing paleontology — from automated fossil identification and CT scan analysis to phylogenetic inference, geometric morphometrics, and ancient DNA reconstruction.
---

# AI in Paleontology

Paleontology is experiencing a digital revolution. CT scanners, photogrammetry, and high-throughput sequencing are generating terabytes of fossil data that overwhelm traditional manual analysis. **Artificial intelligence** enables paleontologists to process this data at unprecedented scale — automating species identification, reconstructing evolutionary trees, extracting functional information from bone morphology, and even inferring the appearance of extinct creatures from fragmentary remains.

## Automated Fossil Identification

### Image-Based Species Classification

Convolutional neural networks trained on curated fossil image databases can match and often exceed expert taxonomic accuracy:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class FossilClassifier(nn.Module):
    def __init__(self, num_species: int = 500, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b4(pretrained=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_species)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# Inference pipeline
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = FossilClassifier(num_species=500)
image = Image.open("ammonite_specimen.jpg")
tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    logits = model(tensor)
    species_idx = logits.argmax().item()
```

### Open Datasets for Fossil AI

- **iDigBio**: 130M+ digitized natural history specimen records with images
- **GBIF** (Global Biodiversity Information Facility): museum specimen data
- **PaleoBioDB** (Paleobiology Database): occurrence records for >150,000 taxa
- **MorphoSource**: 3D scan repository for biological specimens

## CT Scan Analysis

Computed tomography allows non-destructive virtual dissection of fossils still embedded in matrix. AI processes the resulting volumetric data:

```python
import numpy as np
import torch
import torch.nn as nn


class FossilSegmentation3D(nn.Module):
    """3D U-Net for segmenting fossil bone from surrounding matrix in CT volumes."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_filters: int = 16):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1), nn.BatchNorm3d(out_c), nn.ReLU(),
                nn.Conv3d(out_c, out_c, 3, padding=1), nn.BatchNorm3d(out_c), nn.ReLU(),
            )

        f = base_filters
        self.enc1 = conv_block(in_channels, f)
        self.enc2 = conv_block(f, f * 2)
        self.enc3 = conv_block(f * 2, f * 4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = conv_block(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = conv_block(f * 2, f)
        self.out_conv = nn.Conv3d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
```

This enables virtual preparation (removing matrix), internal anatomy visualization (pneumaticity in dinosaur bones), and taphonomic analysis (crack vs. real feature).

## Geometric Morphometrics and Landmark Analysis

Geometric morphometrics studies shape variation by placing homologous landmarks on specimens. ML now automates landmark detection:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def procrustes_align(configurations: np.ndarray) -> np.ndarray:
    """
    Generalized Procrustes Analysis — remove scale, location, rotation.
    configurations: (N, K, D) — N specimens, K landmarks, D dimensions
    """
    from scipy.spatial import procrustes
    reference = configurations[0].copy()
    aligned = [reference]
    for i in range(1, len(configurations)):
        _, aligned_i, _ = procrustes(reference, configurations[i])
        aligned.append(aligned_i)
    return np.array(aligned)


def shape_space_pca(aligned: np.ndarray, n_components: int = 10):
    """Project Procrustes-aligned shapes into principal component space."""
    N, K, D = aligned.shape
    X = aligned.reshape(N, K * D)  # (N, K*D)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return scores, pca


# Classify species from shape
aligned = procrustes_align(landmark_data)
scores, pca = shape_space_pca(aligned)
lda = LinearDiscriminantAnalysis()
lda.fit(scores, species_labels)
```

## Phylogenetic Tree Inference

AI accelerates Bayesian phylogenetic inference, which can take weeks for large datasets:

```python
# PhyloGFN — GFlowNets for phylogenetic tree posterior sampling
# Conceptual example of sampling trees from approximate posterior

class PhyloSampler:
    """
    Approximate posterior sampler over phylogenetic trees.
    Replaces MCMC (e.g., BEAST, MrBayes) with faster neural approximation.
    """

    def __init__(self, num_taxa: int, embedding_dim: int = 128):
        self.num_taxa = num_taxa
        # GFlowNet policy network builds trees step by step (merge pairs)
        self.policy = nn.Sequential(
            nn.Linear(num_taxa * embedding_dim, 256), nn.ReLU(),
            nn.Linear(256, num_taxa * (num_taxa - 1) // 2),  # log-prob of each merge
        )

    def sample_tree(self, sequence_embeddings: torch.Tensor) -> list:
        """Sequentially merge taxa according to policy logits."""
        active = list(range(self.num_taxa))
        merges = []
        h = sequence_embeddings.flatten()
        while len(active) > 1:
            logits = self.policy(h.unsqueeze(0)).squeeze()
            pair_idx = torch.multinomial(logits.softmax(dim=0), 1).item()
            i, j = self._idx_to_pair(pair_idx, len(active))
            merges.append((active[i], active[j]))
            active.pop(max(i, j))
            active.pop(min(i, j))
            active.append(f"node_{len(merges)}")
        return merges

    def _idx_to_pair(self, idx: int, n: int):
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if k == idx:
                    return i, j
                k += 1
```

## Ancient DNA Analysis

ML interprets degraded ancient DNA (aDNA) — short fragments with characteristic damage patterns (C→T deamination at termini):

```python
def authenticate_adna(read_sequences: list, reference: str) -> dict:
    """
    Authenticate ancient DNA by checking for characteristic damage patterns.
    Returns damage score and estimated age proxy.
    """
    ct_mismatches_5prime = 0
    ga_mismatches_3prime = 0
    total_reads = len(read_sequences)

    for read in read_sequences:
        # Check first 3 positions for C→T transitions
        for pos in range(min(3, len(read))):
            if read[pos] == 'T' and pos < len(reference) and reference[pos] == 'C':
                ct_mismatches_5prime += 1
        # Check last 3 positions for G→A transitions
        for pos in range(-3, 0):
            if read[pos] == 'A' and reference[pos] == 'G':
                ga_mismatches_3prime += 1

    damage_5prime = ct_mismatches_5prime / (3 * total_reads)
    damage_3prime = ga_mismatches_3prime / (3 * total_reads)
    return {"damage_5prime": damage_5prime, "damage_3prime": damage_3prime}
```

## Applications Summary

| Application | Method | Key Tool/Model |
|---|---|---|
| Fossil image classification | EfficientNet / ViT | iDigBio + custom training |
| CT bone segmentation | 3D U-Net | MedSAM, nnU-Net |
| Landmark detection | CNN heatmap regression | DeepLabCut adaptation |
| Phylogenetic inference | GFlowNet / BNNs | PhyloGFN, VariPhy |
| aDNA authentication | ML damage scoring | mapDamage, PyDamage |
| Morphological gap filling | Generative models | GANs for fossil reconstruction |

## Ethical and Practical Considerations

- **Sampling bias**: museum collections over-represent certain taxa, geographies, and time periods — AI models trained on these may perpetuate biases in taxonomic coverage
- **Reproducibility**: models trained on proprietary museum databases are hard to reproduce; open-weight models and public datasets are essential
- **Indigenous heritage**: many fossil sites are on Indigenous lands; AI-driven prospecting must respect sovereignty and benefit-sharing agreements

## Summary

AI is reshaping paleontology from a largely descriptive science into one where pattern recognition across millions of specimens, probabilistic evolutionary inference, and 3D virtual morphology analysis are computationally tractable. As museum digitization accelerates and ancient DNA databases grow, the combination of computer vision, geometric deep learning, and probabilistic inference will unlock evolutionary insights locked in rocks for hundreds of millions of years.
