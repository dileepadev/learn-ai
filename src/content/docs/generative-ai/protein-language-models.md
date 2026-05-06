---
title: Protein Language Models
description: A deep dive into protein language models (pLMs) — large transformer-based models trained on protein sequences that learn evolutionary, structural, and functional information without explicit structural supervision.
---

# Protein Language Models

**Protein Language Models (pLMs)** apply the transformer architecture to protein sequences, treating amino acids as tokens in a biological "language" shaped by billions of years of evolution. Trained on hundreds of millions of sequences from databases like UniRef, these models learn rich representations capturing evolutionary relationships, secondary and tertiary structure, and functional properties — enabling zero-shot fitness prediction, structure prediction, and protein design without labeled data.

## Why Sequences Encode Structure

The central hypothesis is that the statistics of evolutionary co-variation in protein multiple sequence alignments (MSAs) implicitly encode 3D structural constraints. If two residues co-evolve, they are likely in contact. A model trained on enough sequences can learn these constraints from single sequences without ever seeing structural data.

Formally, the log-likelihood of a sequence $p(x_1, \ldots, x_L)$ under a well-trained pLM approximates the log-probability of the sequence being a naturally occurring, functional protein.

## Tokenization

Proteins are sequences of 20 standard amino acids (plus special tokens):

```python
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ESM tokenizer example
from esm import Alphabet

alphabet = Alphabet.from_architecture("ESM-1b")
batch_converter = alphabet.get_batch_converter()

data = [("protein1", "MKTIIALSYIFCLVFA")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
print(batch_tokens.shape)  # (1, L+2) — includes <cls> and <eos>
```

## ESM-2: Evolutionary Scale Modeling

ESM-2 (Lin et al., 2023) is Meta's flagship pLM, trained with masked language modeling on 250 million UniRef50 sequences:

```python
import torch
import esm

# Load ESM-2 650M parameter model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

data = [
    ("insulin_B_chain", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

# Per-residue embeddings (last layer)
token_representations = results["representations"][33]
# Predicted contact map
contacts = results["contacts"]
print(contacts.shape)  # (L, L) — symmetric probability matrix
```

ESM-2 models are available at multiple scales: 8M, 35M, 150M, 650M, 3B, 15B parameters. Larger models predict contacts and structure more accurately.

## Zero-Shot Fitness Prediction

Mutation effects can be predicted by comparing the log-likelihood of mutant vs. wild-type sequences:

$$\Delta \log p = \log p(\text{mutant}) - \log p(\text{wild\text{-}type})$$

Positive values suggest the mutation is tolerated; negative values suggest it disrupts function:

```python
def score_mutation(
    model,
    alphabet,
    sequence: str,
    position: int,
    mutant_aa: str,
) -> float:
    """Zero-shot mutation effect prediction."""
    batch_converter = alphabet.get_batch_converter()

    wt_aa = sequence[position]
    mutant_seq = sequence[:position] + mutant_aa + sequence[position + 1:]

    sequences = [("wt", sequence), ("mut", mutant_seq)]
    _, _, tokens = batch_converter(sequences)

    with torch.no_grad():
        log_probs = model(tokens, repr_layers=[], return_contacts=False)
        # Get masked marginals
        logits = log_probs["logits"]  # (2, L+2, vocab)
        log_p = logits.log_softmax(dim=-1)

    pos = position + 1  # offset for <cls>
    wt_idx = alphabet.get_idx(wt_aa)
    mut_idx = alphabet.get_idx(mutant_aa)

    delta = log_p[1, pos, mut_idx] - log_p[0, pos, wt_idx]
    return delta.item()
```

This approach achieves Spearman correlations of 0.6–0.8 with experimental fitness measurements across diverse protein families.

## ESMFold: Structure Prediction from Language Models

ESMFold (Lin et al., 2023) wraps ESM-2 with a folding head that predicts 3D coordinates directly from a single sequence — achieving AlphaFold2-comparable accuracy at 60× higher speed:

```python
import esm

# ESMFold — single-sequence structure prediction
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

sequence = "MKTIIALSYIFCLVFAQKIPGAHMVLSLDFSQRTA"
with torch.no_grad():
    output = model.infer_pdb(sequence)

# Save as PDB
with open("structure.pdb", "w") as f:
    f.write(output)
```

The key insight is that ESM-2 representations already contain sufficient structural information — the folding head simply reads off coordinates.

## Inverse Folding: Sequence Design from Structure

Given a 3D backbone, inverse folding models design amino acid sequences likely to fold into that structure. ESM-IF1 accomplishes this with a geometric encoder over the backbone $C_\alpha$ atoms:

```python
import esm

model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

# Load structure and design sequences
from esm.inverse_folding.util import load_structure, extract_coords_from_structure

structure = load_structure("protein.pdb", chain_id="A")
coords, native_seq = extract_coords_from_structure(structure)

# Sample diverse designed sequences
sampled_seqs = model.sample(
    coords,
    temperature=1.0,
    num_samples=10,
)
```

## ProtTrans: HuggingFace-Compatible pLMs

ProtTrans (Elnaggar et al., 2022) provides T5 and BERT-based protein models compatible with `transformers`:

```python
from transformers import T5Tokenizer, T5EncoderModel

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

sequence = "MKTIIALSYIFCLVFAQKIPGAHM"
# Spaces required between residues for T5 tokenizer
sequence_spaced = " ".join(list(sequence))

ids = tokenizer(sequence_spaced, return_tensors="pt", padding=True)
with torch.no_grad():
    embedding = model(**ids).last_hidden_state  # (1, L, 1024)
```

## Fine-Tuning for Downstream Tasks

pLM embeddings serve as powerful features for supervised tasks:

```python
import torch.nn as nn
from torch.utils.data import DataLoader


class ProteinClassifier(nn.Module):
    def __init__(self, plm, embed_dim: int = 1280, num_classes: int = 2):
        super().__init__()
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = False   # freeze pLM
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, tokens):
        with torch.no_grad():
            representations = self.plm(tokens, repr_layers=[33])["representations"][33]
        # Mean pool over sequence length (excluding special tokens)
        embeddings = representations[:, 1:-1].mean(dim=1)
        return self.head(embeddings)
```

## Key Protein Language Models

| Model | Params | Training Data | Key Capability |
|---|---|---|---|
| ESM-2 | 8M–15B | UniRef50 (250M) | Contact prediction, embeddings |
| ESMFold | 690M | UniRef50 | Direct structure prediction |
| ProtT5-XL | 3B | UniRef50/100 | General embeddings |
| ProtBERT | 420M | UniRef100 | Classification tasks |
| ESM-IF1 | 142M | CATH structures | Inverse folding |
| ProGen2 | 151M–6.4B | UniRef90 + PDB | Protein generation |
| EvoDiff | 640M | UniRef50 | Discrete diffusion generation |

## Summary

Protein language models have transformed computational biology by enabling structure, function, and fitness prediction from sequence alone. ESM-2 and ESMFold demonstrate that billion-scale pretraining on evolutionary data encodes almost all the information needed for structure prediction — previously thought to require explicit co-evolutionary modeling or expensive multiple sequence alignment. As these models scale and improve, they are becoming foundational infrastructure for drug discovery, enzyme engineering, and de novo protein design.
