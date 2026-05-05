---
title: AI in Art Conservation
description: Explore how AI is transforming art conservation and cultural heritage preservation — from multispectral imaging that reveals hidden underdrawings to crack detection with semantic segmentation, AI-assisted authentication, virtual restoration with diffusion models, provenance research with NLP, and the Rijksmuseum Night Watch restoration as a landmark case study.
---

Art conservation — the science of preserving, analyzing, and restoring cultural heritage objects — has traditionally been one of the most labor-intensive and expert-dependent disciplines. A conservator studying a single painting might spend months with specialized cameras, X-ray equipment, and chemical sampling before making restoration decisions. AI is dramatically accelerating this process: what once required months of expert analysis can now be completed in hours, revealing layers of history invisible to the naked eye and enabling data-driven conservation decisions for the world's millions of at-risk artworks.

## Imaging Modalities for Non-Destructive Analysis

Before AI can act, specialized imaging must capture information beyond the visible spectrum. The digital imaging stack in modern conservation:

**Multispectral imaging (MSI)** captures images at 8-20 discrete wavelength bands, including near-infrared (NIR, 800–1000nm) and ultraviolet (UV, 320–400nm). Infrared radiation penetrates paint layers, revealing underdrawings made in charcoal or chalk beneath the final painted surface — the artist's original composition plan, often significantly different from the finished work.

**Hyperspectral imaging (HSI)** extends this to hundreds of contiguous bands, enabling chemical fingerprinting of individual pigments. Prussian blue, natural ultramarine, and synthetic ultramarine have distinct spectral signatures that can be mapped spatially — revealing where a painting has been restored, dated by pigment availability (Prussian blue was unavailable before 1704).

**Macro X-ray fluorescence (MA-XRF)** scanning maps elemental composition (lead, mercury, copper, cobalt) across the entire canvas, revealing compositional changes and pentimenti (painted-over earlier versions).

AI operates on these imaging datasets to automate pattern recognition that previously required expert eyes.

## Crack Detection with Semantic Segmentation

Cracking (craquelure) is the network of fine cracks that develops in paint films as materials age. The pattern of craquelure is diagnostic: old master paintings develop characteristic age cracks, while forgeries may show mechanically induced cracking or anachronistic patterns. Automated crack detection also quantifies deterioration for condition monitoring.

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class CrackDetectionUNet(nn.Module):
    """
    U-Net for crack/damage segmentation in high-resolution artwork scans.
    
    Semantic segmentation labels each pixel as:
    - Crack (fine linear fractures in paint film)
    - Flaking (areas where paint is actively lifting or missing)
    - Cupping (raised, curved areas from support deformation)
    - Varnish yellowing (darkened coating over paint)
    - Healthy paint
    
    Input: RGB scan at 300+ DPI, typically processed in tiles (512×512 or 1024×1024)
    Output: Per-pixel class probability map, shape (B, n_classes, H, W)
    
    Training data: conservation institute databases (Rijksmuseum, Getty, Fraunhofer)
    augmented with synthetic crack patterns applied to undamaged artworks.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 5,
                 features: list[int] = [64, 128, 256, 512]):
        super().__init__()
        
        # Encoder (downsampling path) — typically pretrained ResNet or EfficientNet
        self.encoder1 = self._double_conv(in_channels, features[0])
        self.encoder2 = self._double_conv(features[0], features[1])
        self.encoder3 = self._double_conv(features[1], features[2])
        self.encoder4 = self._double_conv(features[2], features[3])
        self.bottleneck = self._double_conv(features[3], features[3] * 2)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder (upsampling path) with skip connections from encoder
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
        self.decoder4 = self._double_conv(features[3] * 2, features[3])
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.decoder3 = self._double_conv(features[2] * 2, features[2])
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.decoder2 = self._double_conv(features[1] * 2, features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.decoder1 = self._double_conv(features[0] * 2, features[0])
        
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def _double_conv(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        return self.final_conv(d1)   # (B, n_classes, H, W) logits
```

## Artwork Style Embeddings for Attribution

Visual similarity models can assist in attribution — grouping works by the same hand, identifying copies and forgeries, and studying workshop practices where paintings were made by masters and their assistants collaboratively. CLIP and DINOv2 provide strong foundational embeddings for artwork images:

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ArtworkAttributionEmbedder:
    """
    Extract semantic-visual embeddings from artwork images for:
    - Stylometric attribution (which artist made this?)
    - Iconographic classification (scenes, subjects, genres)
    - Provenance linkage (find related works in museum databases)
    - Forgery detection (does this painting's style match its claimed attribution?)
    
    CLIP is well-suited because:
    - Pre-trained on diverse internet images including artworks and museum photos
    - Language-vision alignment allows text queries like "impressionist landscape"
    - The image embedding captures style, composition, and content simultaneously
    
    For higher accuracy on artworks specifically, fine-tune on WikiArt (81K images,
    27 artists, 27 styles) or the Rijksmuseum challenge dataset.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def embed_artwork(self, image: Image.Image) -> torch.Tensor:
        """Extract normalized image embedding for an artwork."""
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        return embedding / embedding.norm(dim=-1, keepdim=True)   # L2 normalize

    def find_similar_works(
        self,
        query_image: Image.Image,
        database_images: list[Image.Image],
        top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Find the most visually similar artworks in a database."""
        query_emb = self.embed_artwork(query_image)
        
        db_embeddings = torch.cat([
            self.embed_artwork(img) for img in database_images
        ], dim=0)
        
        similarities = (query_emb @ db_embeddings.T).squeeze(0)
        top_indices = similarities.topk(top_k).indices.tolist()
        top_scores = similarities.topk(top_k).values.tolist()
        
        return list(zip(top_indices, top_scores))

    def text_guided_search(
        self,
        query_text: str,
        database_images: list[Image.Image],
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Find artworks matching a natural language description.
        
        Examples:
        - "Rembrandt self-portrait with chiaroscuro lighting"
        - "17th century Dutch still life with flowers"
        - "cracked paint surface showing underdrawing"
        """
        text_inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_emb = self.model.get_text_features(**text_inputs)
        
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        db_embeddings = torch.cat([
            self.embed_artwork(img) for img in database_images
        ], dim=0)
        
        similarities = (text_emb @ db_embeddings.T).squeeze(0)
        top_indices = similarities.topk(top_k).indices.tolist()
        top_scores = similarities.topk(top_k).values.tolist()
        
        return list(zip(top_indices, top_scores))
```

## Virtual Restoration with Diffusion Models

Diffusion models trained on artwork datasets can propose virtual restorations — generating plausible completions of damaged or missing areas. This is strictly a **visualization tool** for conservators: virtual restoration helps stakeholders and the public understand what a work may have looked like, but physical conservation decisions remain the domain of human experts with direct physical access to the object.

The Rijksmuseum's **Night Watch restoration project** (2019–2021) combined X-ray imaging, high-resolution photography, and deep learning to extend the painting beyond its cropped borders. Johannes Cornelis Looten copied the original 1715 composition before it was cut, and convolutional neural networks trained on Rembrandt's style synthesized the missing portions, guided by the Looten copy and X-ray data.

## Provenance Research with NLP

Establishing an artwork's chain of ownership (provenance) is critical for identifying stolen or looted works and authenticating attribution claims. Provenance records are scattered across:

- Auction house catalogs (Christie's, Sotheby's archives dating to 1766)
- Exhibition catalogs and loan records
- Historical inventories and estate records in multiple languages
- Dealer correspondence and private sale records

LLMs and NER models extract structured provenance chains from these unstructured text sources, linking mentions of specific works across centuries of records. The Art Loss Register (world's largest private database of stolen art) and Cultural Property Investigative Unit use NLP-based entity resolution to match recovered works against records of looted art.

## The Salvator Mundi Attribution Case

The 2017 sale of the *Salvator Mundi* attributed to Leonardo da Vinci for $450 million at Christie's prompted intense technical analysis. AI-based stylometric analysis compared brushwork patterns, sfumato technique characteristics, and underdrawing style against securely attributed Leonardo works using multispectral imaging and high-resolution digital scans. The case illustrates both the promise and the limits of computational attribution: technical evidence can narrow possibilities but cannot definitively resolve contested attributions without convergent documentary and technical evidence.

## 3D Digitization and NeRF

**Neural Radiance Fields** (NeRF) and **photogrammetry** enable millimeter-accurate 3D reconstructions of sculptures, reliefs, and architectural elements from photographs — creating permanent digital records before degradation, enabling remote study, and generating the geometry needed for 3D-printed replicas that make collections accessible without risking originals.

The British Museum's photogrammetry program has digitized over 1,400 objects. Google Arts & Culture's Art Camera has captured more than 3,000 high-resolution gigapixel images enabling virtual close examination.

AI in art conservation exemplifies a pattern repeated across domains: the technology does not replace expertise but changes where expertise is most needed — from mechanical pattern recognition toward higher-order judgment about interpretation, context, and ethical decision-making.
