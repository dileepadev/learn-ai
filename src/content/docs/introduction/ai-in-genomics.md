---
title: AI in Genomics
description: Discover how AI is transforming genomics and computational biology — from DNA foundation models and variant effect prediction to single-cell RNA sequencing analysis and polygenic risk scores. Learn how large-scale sequence models like HyenaDNA and Enformer predict gene expression, how scRNA-seq enables cell type discovery, and how federated learning addresses multi-hospital genomics data challenges.
---

The human genome contains approximately 3.2 billion base pairs, with over 4 million regulatory elements and 20,000 protein-coding genes. Understanding how this sequence determines biological function — and how variants in it cause disease — is one of the most complex pattern recognition problems in science. AI has become indispensable in genomics, not by replacing biological intuition, but by extracting signals from datasets at scales that are impossible to interpret manually.

## Genomics Data Types

Genomics AI operates on diverse molecular data modalities, each capturing different aspects of genome function:

| Data Type | Measures | Common Formats | Scale |
| --- | --- | --- | --- |
| DNA sequence | Primary sequence of base pairs (A, T, C, G) | FASTA, VCF | 3.2 billion bp / human |
| Bulk RNA-seq | Gene expression levels across cell populations | Count matrices | ~20,000 genes |
| scRNA-seq | Gene expression per single cell | Sparse matrices | 10K–1M cells × 20K genes |
| ChIP-seq | Transcription factor binding, histone marks | BED, bigWig | Peak calls along genome |
| ATAC-seq | Open chromatin / accessible regulatory regions | BED, bigWig | ~200,000 peaks / cell type |
| Hi-C | 3D chromatin conformation, TAD boundaries | Contact matrices | Genome-wide contacts |

## DNA Sequence Encoding

The first step in any sequence-based model is converting the nucleotide alphabet to numerical representations:

```python
import numpy as np
import torch
from typing import Optional

# Standard one-hot encoding
BASE_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}

def one_hot_encode_dna(sequence: str, max_length: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    
    Standard representation for CNNs and classical models.
    Each base becomes a 4-dimensional binary vector:
    A = [1,0,0,0], T = [0,1,0,0], C = [0,0,1,0], G = [0,0,0,1]
    N (ambiguous) = [0.25, 0.25, 0.25, 0.25]  (soft encoding for unknown bases)
    
    Returns: (sequence_length, 4) float array
    """
    seq = sequence.upper()
    if max_length:
        seq = seq[:max_length].ljust(max_length, "N")
    
    encoding = np.zeros((len(seq), 4), dtype=np.float32)
    
    for i, base in enumerate(seq):
        if base == "A":
            encoding[i, 0] = 1.0
        elif base == "T":
            encoding[i, 1] = 1.0
        elif base == "C":
            encoding[i, 2] = 1.0
        elif base == "G":
            encoding[i, 3] = 1.0
        elif base == "N":
            encoding[i, :] = 0.25
    
    return encoding


class SimpleCNNVariantClassifier(torch.nn.Module):
    """
    Simple CNN for variant effect prediction.
    
    Given a short DNA sequence around a single nucleotide variant (SNV),
    predicts whether the variant is likely pathogenic (disease-causing)
    or benign. This is the core task of DeepSEA (Zhou & Troyanskaya, 2015)
    and subsequent deep learning-based variant effect predictors.
    
    Input: one-hot encoded DNA sequences, shape (B, seq_len, 4)
    Output: binary classification logits (B, 1)
    
    More sophisticated models (Enformer, Sei) predict hundreds of
    epigenetic tracks (histone modifications, TF binding, DNase accessibility)
    which then serve as features for variant interpretation.
    """
    
    def __init__(self, seq_length: int = 1000, n_filters: int = 64):
        super().__init__()
        
        # Input: (B, 4, seq_len) — channels first for Conv1d
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(4, n_filters, kernel_size=8, padding=4),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(n_filters, n_filters * 2, kernel_size=8, padding=4),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=8, padding=4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_filters * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, 4) one-hot encoded DNA
        Returns: (B, 1) logits for pathogenic/benign classification
        """
        x = x.permute(0, 2, 1)   # (B, 4, seq_len) for Conv1d
        features = self.conv_layers(x)
        return self.classifier(features)
```

## DNA Foundation Models

Modern DNA foundation models adapt the transformer paradigm to genomic sequence modeling, learning representations that transfer to downstream tasks just as BERT transfers to NLP tasks.

**DNABERT-2**: Tokenizes DNA using BPE (Byte Pair Encoding) rather than k-mers, enabling efficient processing of diverse genomes. Pre-trained on multi-species DNA with masked language modeling.

**HyenaDNA** (Nguyen et al., 2023): Based on the Hyena operator (a subquadratic alternative to attention), HyenaDNA processes DNA sequences up to 1 million base pairs — far beyond BERT-style models' context limits. This long context is critical for capturing long-range regulatory interactions that can span hundreds of kilobases.

```python
from transformers import AutoTokenizer, AutoModel
import torch

def get_dna_embeddings(sequences: list[str],
                        model_name: str = "zhihan1996/DNABERT-2-117M") -> torch.Tensor:
    """
    Extract contextual DNA sequence embeddings using DNABERT-2.
    
    Useful for:
    - Variant effect prediction (compare ref vs alt embeddings)
    - Regulatory element classification (enhancer, promoter, silencer)
    - Species-agnostic genome annotation
    - Sequence similarity search in embedding space
    
    The [CLS] token embedding is a sequence-level representation.
    Token embeddings can be used for nucleotide-level predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    all_embeddings = []
    
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True,
                          max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling over all token embeddings for sequence-level representation
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        token_embeddings = outputs.last_hidden_state
        embedding = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        
        all_embeddings.append(embedding.cpu())
    
    return torch.cat(all_embeddings, dim=0)   # (n_sequences, hidden_dim)
```

## Enformer: Predicting Gene Expression from Sequence

**Enformer** (Avsec et al., 2021, DeepMind) predicts histone modifications, TF binding, and gene expression directly from 200,000 base pairs of DNA sequence context. It uses a transformer with dilated convolutions to process the long genomic window, then predicts 5,313 genomic tracks simultaneously.

The key advance over DeepSEA: Enformer's 200kb receptive field can capture distal enhancers and their regulatory interactions with gene promoters, making it able to explain how distant sequence variants affect gene expression — directly actionable for interpreting GWAS hits.

## Single-Cell RNA Sequencing Analysis

**scRNA-seq** measures gene expression in individual cells rather than bulk populations, revealing cell-type heterogeneity invisible to bulk methods. A typical scRNA-seq experiment produces a sparse matrix of ~10,000 cells × ~20,000 genes.

```python
import numpy as np

def basic_scrna_pipeline(
    count_matrix: np.ndarray,   # (n_cells, n_genes) raw UMI counts
    min_cells: int = 3,
    min_genes: int = 200,
    max_genes: int = 5000,
    max_pct_mito: float = 0.2   # filter out likely dying cells
) -> np.ndarray:
    """
    Basic scRNA-seq preprocessing pipeline.
    
    Quality control removes:
    - Cells with too few detected genes (empty droplets or low-quality cells)
    - Cells with too many detected genes (doublets — two cells in one droplet)
    - Cells with high mitochondrial gene expression (dying/apoptotic cells)
    
    Normalization and log transformation prepare data for downstream analysis:
    PCA, UMAP visualization, cell type clustering, trajectory inference.
    
    Production pipelines use Scanpy (Python) or Seurat (R).
    Foundation models (Geneformer, scGPT) skip these manual steps by
    operating on raw counts with learned normalization.
    """
    # QC filtering
    n_genes_per_cell = (count_matrix > 0).sum(axis=1)
    n_cells_per_gene = (count_matrix > 0).sum(axis=0)
    
    cell_mask = (n_genes_per_cell >= min_genes) & (n_genes_per_cell <= max_genes)
    gene_mask = n_cells_per_gene >= min_cells
    
    filtered = count_matrix[cell_mask][:, gene_mask]
    
    # Normalize: library size normalization to 10,000 counts per cell
    lib_sizes = filtered.sum(axis=1, keepdims=True)
    normalized = filtered / lib_sizes * 10_000
    
    # Log1p transform: compress dynamic range
    log_normalized = np.log1p(normalized)
    
    return log_normalized
```

**Geneformer** (Theodoris et al., 2023): A transformer pre-trained on 29.9 million single-cell transcriptomes from the Human Cell Atlas. Each cell is represented as a sequence of genes ranked by their expression level — the most highly expressed genes come first. Fine-tuned Geneformer achieves state-of-the-art on cell type classification, disease gene prioritization, and virtual drug perturbation prediction.

## Variant Effect Prediction and Polygenic Risk Scores

**Polygenic Risk Scores (PRS)** aggregate the effects of many common genetic variants to estimate an individual's risk for a complex disease. PRS calculations traditionally sum variant effect sizes from GWAS (Genome-Wide Association Studies), but AI is improving PRS by:

- Learning non-linear interaction effects between variants
- Incorporating functional annotations (is the variant in a regulatory region?)
- Using transfer learning from model organisms
- Generalizing across diverse ancestries (historically a major weakness of PRS)

**AlphaMissense** (Google DeepMind, 2023): Classifies all 71 million possible single amino acid substitutions in human proteins as likely pathogenic, benign, or uncertain — using a variant of AlphaFold's sequence representations fine-tuned on ClinVar and population frequency data.

## Federated Learning for Multi-Hospital Genomics

Genomic data is among the most privacy-sensitive human data — it is uniquely identifying and reveals information about relatives who never consented to participate. Multi-hospital genomics studies require federated learning:

- Each hospital trains locally on its patient population
- Model updates (gradients or parameters) are aggregated centrally without sharing raw sequence data
- Differential privacy mechanisms prevent gradient inversion attacks that could reconstruct individual sequences

Projects like the Global Alliance for Genomics and Health (GA4GH) and federated GWAS frameworks (MetaAnalysis) use these techniques to enable multi-million-participant studies while maintaining GDPR and HIPAA compliance.

AI in genomics is accelerating the virtuous cycle between sequence data and biological understanding — each new model architecture reveals previously invisible patterns in genome organization, which in turn motivates collection of new data types to validate and extend the discoveries.
