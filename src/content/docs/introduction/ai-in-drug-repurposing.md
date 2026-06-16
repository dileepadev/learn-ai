---
title: AI in Drug Repurposing
description: How AI and machine learning accelerate drug repurposing — finding new therapeutic uses for existing approved drugs — through graph networks, embedding models, and clinical data mining.
---

Drug repurposing (also called drug repositioning) identifies new therapeutic uses for drugs that are already approved or in clinical development. Because these drugs have known safety profiles, repurposing dramatically reduces the time and cost to bring a new treatment to patients. AI is transforming this field by systematically analyzing biological, chemical, and clinical data at scales impossible for human researchers.

## Why Drug Repurposing?

Developing a new drug from scratch takes 10–15 years and costs over $1 billion on average. Most candidates fail in clinical trials. Drug repurposing sidesteps much of this:
- **Known safety profile:** Phase I/II safety trials may be skippable or shortened.
- **Faster regulatory path:** Approved drugs have an existing regulatory dossier.
- **Immediate patient benefit:** Promising repurposed drugs can sometimes be used off-label while trials proceed.

Classic examples: aspirin's use as a cardiovascular preventive, sildenafil (Viagra) repurposed from a cardiovascular drug, thalidomide repurposed for multiple myeloma despite its tragic early history.

## AI Approaches to Drug Repurposing

### Knowledge Graph Embeddings
Biomedical knowledge can be represented as a heterogeneous graph: nodes are drugs, diseases, proteins, genes, and pathways; edges represent relationships (drug-targets-protein, protein-involved-in-pathway, gene-associated-with-disease).

Graph neural networks and embedding models (TransE, RotatE, BioKG) learn vector representations for each entity. By completing missing edges — predicting which drug-disease links are likely to exist — these models surface repurposing candidates. OpenKE, PrimeKG, and Hetionet are widely used biomedical knowledge graphs.

### Drug-Target Interaction Prediction
If we know a drug's mechanisms (which proteins it binds), and we know which proteins are implicated in a disease, we can infer potential efficacy. ML models (GraphDTA, DeepDTA, MolBERT) predict drug-target binding affinity from molecular structure and protein sequence, flagging known drugs that bind disease-relevant targets.

### Clinical Data Mining (EHR Analysis)
Electronic health records contain millions of patient treatment histories. By mining EHR data, AI can identify:
- Patients on Drug X for Condition A who unexpectedly have lower rates of Condition B (suggesting a protective effect).
- Adverse events that paradoxically correlate with improved outcomes for certain diseases.

UCSF, Stanford, and other academic medical centers have used this approach to identify potential repurposing candidates for COVID-19, Alzheimer's disease, and rare cancers.

### Transcriptomic Signature Matching (CMap/LINCS)
The Connectivity Map (CMap) and LINCS databases contain gene expression profiles for thousands of drugs in human cell lines. Each drug has a "transcriptomic signature" — how it perturbs gene expression.

A disease also has a signature — the pattern of dysregulated genes. Repurposing candidates are drugs whose signatures are **anti-correlated** with the disease signature (they reverse the disease's gene expression changes). AI models improve on simple correlation by learning non-linear matching and integrating multi-omics data.

### Molecular Generative Models and Virtual Screening
Generative AI can propose new molecular variants of existing drugs with improved properties for a new indication. Reinforcement learning-based molecular design (GuacaMol, REINVENT) optimizes molecules for target binding, solubility, and low toxicity simultaneously.

### LLMs for Literature Mining
The biomedical literature contains millions of papers. LLMs fine-tuned on scientific text (BioGPT, PubMedBERT, BioMistral) can extract drug-disease relationships, summarize evidence across papers, and flag inconsistencies — giving researchers a prioritized reading list rather than an unsearchable mass of text.

## COVID-19: A Case Study

COVID-19 became a proving ground for AI-based repurposing. Within weeks of the pandemic's start:
- Baricitinib (a JAB inhibitor for rheumatoid arthritis) was identified by BenevolentAI's knowledge graph as a potential COVID-19 treatment. It was subsequently validated in clinical trials and received FDA emergency use authorization.
- Multiple academic groups used transcriptomic signature matching and drug-target interaction models to identify candidates, several of which entered trials.

This demonstrated AI's ability to identify and prioritize candidates in days rather than years.

## Challenges

- **Validation gap:** AI models generate many candidates; experimental validation (cell assays, animal studies, trials) remains expensive and slow. Prioritization quality determines how much work is saved.
- **Data quality:** Knowledge graphs contain errors and gaps. Models trained on incomplete data will miss true repurposing opportunities and generate false positives.
- **Polypharmacology:** Drugs have many targets beyond their primary one. Correctly accounting for off-target effects is essential for safety prediction.
- **Regulatory acceptance:** Regulators are beginning to engage with AI-generated evidence, but clear frameworks for AI-assisted repurposing submissions are still developing.

## Outlook

AI drug repurposing is shifting from research curiosity to mainstream pharmaceutical practice. Major pharma companies (Pfizer, AstraZeneca, Johnson & Johnson) have internal AI repurposing programs, and startups like BenevolentAI, Exscientia, and Insilico Medicine have raised substantial funding in this space.
