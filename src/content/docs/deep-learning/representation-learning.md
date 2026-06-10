---
title: Representation Learning and Embeddings
description: Learning meaningful vector representations — word embeddings, contextual embeddings, and measuring semantic similarity.
---

**Representation learning** is the foundation of modern deep learning: learn to represent data (text, images, graphs) as dense vectors that capture semantic meaning.

High-quality representations enable downstream tasks (classification, retrieval, reasoning) to learn efficiently. Poor representations force models to compensate, requiring more capacity and data.

## Why Representations Matter

### Example: Word Embeddings

**One-hot encoding**: Each word as a sparse vector with one 1 and rest 0s.

```
"cat" = [0, 0, 0, ..., 1, 0, 0]  (4000 dimensions, most zeros)
```

Provides no information about semantic relationships. "cat" and "dog" are orthogonal — same distance as "cat" and "carburetor."

**Word embeddings** (e.g., Word2Vec): Dense vectors where semantically similar words are close:

```
cat = [0.1, -0.5, 0.3, 0.2, ...]  (300 dimensions)
dog = [0.15, -0.48, 0.32, 0.19, ...]  (300 dimensions)
carburetor = [-0.6, 0.2, -0.1, 0.7, ...]  (300 dimensions)
```

cosine_sim(cat, dog) ≈ 0.95, cosine_sim(cat, carburetor) ≈ 0.1. Similarity reflects semantic relationships.

## Word Embeddings

### Word2Vec

Learn embeddings by predicting context (surrounding words) from a word:

$$\max \log P(w_{\text{context}} | w_{\text{target}}) = \max \log \frac{\exp(u_{\text{context}} \cdot v_{\text{target}})}{\sum_{w'} \exp(u_{w'} \cdot v_{\text{target}})}$$

Train on large corpora (billions of words). Each word's embedding is optimized to predict its context.

**Properties**:
- Semantic relationships emerge: embeddings capture analogy structure. $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$.
- One embedding per word (non-contextual).

### GloVe

Factorizes word co-occurrence matrix:

$$v_i^T v_j + b_i + b_j = \log X_{ij}$$

where $X_{ij}$ is co-occurrence count of words $i, j$. Learns embeddings that reconstruct the co-occurrence structure.

**Advantage over Word2Vec**: Combines global statistical information (co-occurrence) with local context.

## Contextual Embeddings

**Limitation of Word2Vec**: One embedding per word, regardless of context. "Bank" in "river bank" and "financial bank" get the same embedding.

**Contextual embeddings** compute embeddings conditioned on surrounding context.

### ELMo

Bidirectional LSTM layers:

$$h_t = \text{LSTM}(\text{word}_t, h_{t-1})$$

Embedding of word $t$ depends on all surrounding words (via bidirectional LSTM). Different embedding for "bank" in different contexts.

### BERT

Masked language modeling:

```
Input: [CLS] The quick brown fox [SEP]
Masked: [CLS] The quick [MASK] fox [SEP]
Task: Predict "brown"
```

Bidirectional transformer predictions are context-dependent. Embedding of [MASK] changes based on neighbors.

**Advantages**:
- Captures context at multiple granularities (multiple layers).
- Transferable: pre-trained embeddings work for diverse tasks.
- Bidirectional (looks left and right).

## Sentence and Document Embeddings

Extend from words to longer sequences.

### Averaging + Weighting

Average word embeddings (Simple baseline):

$$\text{sent}_{\text{emb}} = \frac{1}{n} \sum_i v_{\text{word}_i}$$

or weighted by importance (TF-IDF):

$$\text{sent}_{\text{emb}} = \frac{1}{Z} \sum_i w_i v_{\text{word}_i}$$

where $w_i$ is TF-IDF weight.

**Limitation**: Loses order and composition; "dog bites man" and "man bites dog" get similar embeddings.

### InferSent and Universal Sentence Encoder

Train sentence encoders on tasks requiring semantic understanding (entailment, similarity):

- **InferSent**: Bilinear pooling of LSTM hidden states + classification of semantic similarity.
- **USE**: Multi-task learning on sentence-level objectives (entailment, semantic similarity, paraphrase).

Resulting embeddings capture sentence meaning; nearby embeddings often have similar semantics.

### Dense Passage Retrieval (DPR)

For retrieval tasks, learn embeddings such that query and relevant document embeddings are similar:

$$\mathcal{L} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{d}^+)}{\sum_{d'} \exp(\mathbf{q} \cdot \mathbf{d}')}$$

Contrastive objective: relevant pairs similar, irrelevant pairs dissimilar. Enables fast retrieval via nearest-neighbor search.

## Vision Embeddings

### CNN Features

Early layers of trained CNNs learn low-level features (edges, textures). Later layers learn high-level features (parts, objects).

Using layer-$k$ activations as embeddings for downstream tasks is effective (transfer learning).

### Vision Transformers

ViT divides images into patches, treats each patch as a token. Embeddings from transformer layers capture image semantics.

Early layers: low-level spatial structure.
Later layers: object categories, scenes.

## Multimodal Embeddings

Align representations across modalities (vision + language, audio + text).

### CLIP

Train image encoder and text encoder jointly:

- Image "cat" and text "a photo of a cat" should have similar embeddings.
- Image "cat" and text "a dog" should have dissimilar embeddings.

Contrastive loss aligns multimodal embeddings. Enables zero-shot classification and image-text retrieval.

### ALIGN and Florence

Similar approaches with scale. Larger datasets and models improve alignment quality.

## Measuring Representation Quality

### Intrinsic Evaluation

**Semantic similarity**: Do embeddings capture semantic relationships?

- Evaluate on word similarity benchmarks (RareWord, SimLex-999): Compare embedding similarity to human-judged similarity.
- Correlation (Spearman, Pearson) measures how well embeddings match human judgments.

**Analogy tasks**: Do embeddings support analogies?

- "king" - "man" + "woman" ≈ "queen"?
- Accuracy on analogy completion tasks.

### Extrinsic Evaluation

Downstream task performance:

- Use embeddings as features for classification, clustering, or retrieval.
- Better embeddings → better downstream task performance.

## Challenges

### Curse of Dimensionality

High-dimensional embeddings can hurt nearest-neighbor retrieval and storage. Dimensionality reduction (PCA, quantization) helps but may lose information.

### Distributional Bias

Embeddings capture biases in training data. "Doctor" closer to "male" than "female" in many embeddings, reflecting historical gender bias in medical profession.

Debiasing techniques reduce but don't eliminate bias.

### Out-of-Vocabulary Words

For rare or new words not in training, embeddings are unavailable. Handle via:
- Character-level models (compute embeddings from characters).
- Subword tokenization (BPE, WordPiece): break into known subwords.

## Recent Trends

- **Large-scale pretraining**: Embeddings from models trained on billions of tokens/images generalize widely.
- **Multimodal alignment**: Bridging vision, language, audio in unified embedding spaces.
- **Sparse embeddings**: Complement dense embeddings with sparse indicators for interpretability.
- **Efficient retrieval**: Trade-offs between embedding quality and retrieval speed.

Representation learning is the bedrock of modern AI — good representations make downstream tasks dramatically easier, while poor representations doom even sophisticated models. The focus on learning representations (rather than hand-engineering features) is a defining characteristic of deep learning's success.
