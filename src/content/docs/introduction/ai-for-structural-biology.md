---
title: AI for Structural Biology
description: Explore how AI is transforming structural biology — from AlphaFold 3's diffusion-based multi-molecule prediction and RoseTTAFold All-Atom to ESMFold's language model approach, Boltz-1's open-source contributions, and AI-accelerated protein-ligand docking for drug discovery.
---

Structural biology — the study of the three-dimensional shapes of biological molecules — is foundational to understanding how life works at the molecular level. The 3D structure of a protein determines its function, its interactions with other molecules, and how drugs can modulate its activity. Determining structures experimentally, through X-ray crystallography, cryo-electron microscopy, or NMR spectroscopy, is time-consuming and expensive. AI has transformed this landscape dramatically, enabling structure prediction at genome scale and extending to multi-molecule complexes critical for drug discovery.

## AlphaFold 2: The Foundation

**AlphaFold 2** (Jumper et al., DeepMind, 2021) achieved near-experimental accuracy for single-chain protein structure prediction, solving one of biology's 50-year grand challenges. Its architecture — combining multiple sequence alignments (MSAs), attention-based pair representations, and an equivariant structure module with invariant point attention — demonstrated that co-evolutionary information from aligned sequences across species encodes structural constraints.

AlphaFold 2's key insight was representing the protein as a pairwise residue-residue distance and orientation map, then iteratively refining both sequence and structure representations through a recycling mechanism. At CASP14, it achieved a median GDT_TS score of 92.4 on the hardest Free Modelling targets, far surpassing all prior methods.

## AlphaFold 3: Multi-Molecule Prediction

**AlphaFold 3** (Abramowitz et al., DeepMind, 2024) extends prediction from individual proteins to **joint multi-molecule complexes**: proteins, DNA, RNA, small molecule ligands, ions, and covalent modifications. This is the critical capability for drug discovery, where the relevant question is not "what does this protein look like?" but "how does this protein bind to this drug candidate, in the presence of these cofactors, on this DNA strand?"

### Diffusion-Based Architecture

The most significant architectural change in AlphaFold 3 is the replacement of the equivariant structure module with a **diffusion model**:

- The trunk network (Pairformer, similar to AlphaFold 2's Evoformer) produces pair and single representations from the input sequences and MSAs.
- A **diffusion module** then generates 3D atomic coordinates by denoising from Gaussian noise, conditioned on the trunk representations.
- The diffusion process operates directly on all-atom coordinates of every molecule in the complex simultaneously.

This design handles heterogeneous chemistry naturally: proteins, nucleic acids, and small molecules all have different atom types and bonding patterns, but the diffusion denoiser operates on all of them in a unified atomic representation.

### Sequence-Level Input

AlphaFold 3 accepts a **joint sequence input** that concatenates:

- Amino acid sequences for each protein chain.
- Nucleotide sequences for DNA/RNA chains.
- SMILES strings (converted to atomic graphs) for small molecules.
- Ion identities for metal ions.

A cross-chain attention mechanism in the Pairformer learns inter-chain interactions — how the protein surface contacts the ligand, how the DNA groove is recognized by the protein — that are absent from single-chain models.

### Protein-Ligand Docking Accuracy

On the **Posebusters benchmark** (a standard for physically valid protein-ligand binding pose prediction), AlphaFold 3 achieves 76% of poses passing all validity checks, compared to 52% for the best prior physics-based docking methods (Glide SP). Crucially, AlphaFold 3 does not require an initial ligand pose or pocket location — it predicts the complex from scratch.

## RoseTTAFold All-Atom

**RoseTTAFold All-Atom** (Krishna et al., University of Washington / IPD, 2024) takes a similar multi-molecule approach but uses a transformer-based architecture rather than diffusion:

- **3-track architecture**: simultaneously processes 1D (sequence), 2D (pairwise distances), and 3D (coordinate) representations, with information flowing across all three tracks.
- **Chemical graph input**: small molecules are represented as graphs with atom types and bond orders, processed by a chemical graph neural network before being incorporated into the 1D and 2D tracks.
- **Confidence estimates**: outputs per-residue and per-atom pLDDT and PAE scores, enabling automatic quality assessment.

RoseTTAFold All-Atom achieves competitive accuracy with AlphaFold 3 on protein-ligand docking while also supporting **covalent modifications** (phosphorylation, glycosylation, disulfide bonds) and **metal coordination** — important for enzyme active sites.

RoseTTAFold All-Atom is **fully open-source**, including model weights, enabling academic researchers to run it locally without API access.

## ESMFold: The Language Model Approach

**ESMFold** (Lin et al., Meta AI, 2022) takes an entirely different approach: it uses a protein language model (ESM-2, trained on 250 million protein sequences from UniRef50) to predict structure directly from sequence, without any MSA alignment step.

### Single-Sequence Prediction

ESMFold's key contribution is demonstrating that a large enough protein language model internalizes structural information from sequence alone:

- The ESM-2 encoder produces per-residue representations that implicitly encode secondary structure and inter-residue contacts.
- A lightweight Folding Trunk (structure module similar to AlphaFold 2) converts these representations to 3D coordinates.
- **No MSA required**: prediction takes ~1 second per protein on a GPU, vs. 10–100 seconds for AlphaFold 2 (which must compute MSAs from large sequence databases).

This speed advantage makes ESMFold practical for genome-scale structural proteomics: ESMFold was used to predict structures for ~617 million metagenomic proteins from the ESM Metagenomic Atlas, representing an order-of-magnitude expansion of structural knowledge.

### Accuracy vs. Speed Tradeoff

On CAMEO targets (continuous automated model evaluation), ESMFold achieves ~90% of AlphaFold 2's accuracy for well-folded proteins, but falls further behind on intrinsically disordered regions and multi-chain complexes. The accuracy gap reflects the information content of MSAs: evolutionary co-variation encodes pairwise structural constraints that a single-sequence LLM cannot fully recover.

## Boltz-1 and Open-Source Advances

**Boltz-1** (MIT / independent researchers, 2024) is a fully open-source reimplementation and extension of AlphaFold 3's multi-molecule prediction capability. It reproduces AlphaFold 3's performance on standard benchmarks and adds:

- **Open weights and training code**: enabling the research community to fine-tune on custom data (e.g., specific protein families, proprietary compound datasets).
- **Covalent ligand support**: predicts structures with covalent inhibitors (irreversible drugs), which require special handling in the diffusion process.
- **MSA server integration**: connects to ColabFold's MSA server for fast alignment computation.

Boltz-1 represents the democratization of AlphaFold 3-level capabilities outside Google's infrastructure.

## RNA Structure Prediction

RNA structure prediction — 2D (secondary structure) and 3D (tertiary structure) — lags behind protein structure prediction but has seen major advances:

- **RhoFold** and **EternaFold** predict RNA secondary structure (base-pairing patterns) with high accuracy using deep learning, outperforming thermodynamic methods (ViennaRNA, RNAfold).
- **RoseTTAFold Nucleic Acids** extends the 3-track architecture to RNA tertiary structure prediction.
- **AlphaFold 3** includes RNA as a first-class molecular type, predicting protein-RNA complexes (ribosomes, spliceosomes) as well as pure RNA tertiary structures.

RNA is more conformationally flexible than proteins and has a much smaller experimental structure database, making high-accuracy prediction harder. Performance on riboswitch and aptamer structures — where RNA folds specifically to bind small molecules — is an active research frontier.

## Drug Discovery Applications

The primary application of multi-molecule structure prediction in industry is **structure-based drug design**:

### Virtual Screening

Instead of docking millions of compounds against a protein target using classical docking (AutoDock Vina, Glide), AI structure predictors enable:

1. Predict the protein-ligand complex for each candidate compound in a library.
1. Score by predicted binding pose quality (pLDDT, PoseScore).
1. Shortlist compounds for experimental validation.

This workflow scales to **ultra-large libraries** (billions of compounds) that are inaccessible to physics-based docking due to computational cost.

### Hit-to-Lead Optimization

Once a promising hit is identified, AI structure prediction enables rapid optimization:

- Predict structures for a series of analogs (modified versions of the hit).
- Identify which functional groups make favorable contacts with the binding pocket.
- Guide synthesis toward analogs with predicted better binding without expensive experimental assays for each analog.

### Induced Fit and Conformational Change

Classical docking assumes a rigid protein. Real protein-ligand binding often involves **induced fit**: the protein changes conformation to accommodate the ligand. AlphaFold 3's joint diffusion of all atoms simultaneously allows it to model induced fit, generating structures where the protein and ligand co-adapt. This is particularly important for kinase inhibitors, where the DFG loop conformation changes upon ligand binding.

## Limitations and Open Problems

- **Confidence calibration**: pLDDT scores are well-calibrated for proteins but less so for ligands and novel RNA structures.
- **Intrinsically disordered proteins (IDPs)**: proteins that lack a stable 3D structure cannot be "predicted" because no single structure exists. AlphaFold correctly predicts low pLDDT for these regions but the biological interpretation requires additional analysis.
- **Conformational ensembles**: biology requires not just the most stable conformation but the ensemble of accessible conformations. Single-structure predictors miss the dynamic flexibility critical for understanding allosteric regulation.
- **Water and ions**: explicit water molecules and their positions in binding sites are important for docking accuracy; current AI models provide limited information about solvent structure.
- **Experimental validation**: AI predictions accelerate hypothesis generation but still require wet-lab validation; predicted structures contain errors that can mislead downstream analysis.

## Summary

AI for structural biology has progressed from single-chain protein prediction (AlphaFold 2) to joint multi-molecule complex prediction (AlphaFold 3, RoseTTAFold All-Atom), enabling direct computation of protein-ligand, protein-DNA, and protein-RNA structures from sequence alone. ESMFold demonstrates that protein language models can bypass MSA computation for genome-scale prediction. Boltz-1 makes these capabilities open-source. These advances are transforming drug discovery by enabling structure-based virtual screening at billion-compound scale and accelerating hit-to-lead optimization. Open problems — conformational ensembles, intrinsically disordered proteins, water network prediction — define the next frontier for the field.
