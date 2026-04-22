---
title: Transformer Pretraining Objectives
description: Understand the pretraining objectives that give transformers their power — including masked language modeling (BERT), causal language modeling (GPT), span corruption (T5), denoising, contrastive learning, and next sentence prediction.
---

**Transformer pretraining objectives** are the self-supervised learning tasks used to train large language and vision models on unlabeled data before fine-tuning on specific downstream tasks. These objectives define *what* the model must learn to predict during pretraining — and consequently determine what representations the model develops, what capabilities it acquires, and what tasks it transfers well to.

The fundamental idea is **creating a supervisory signal from the data itself**: given a large corpus of text or images, define a task that can be solved only by developing a deep understanding of language or visual structure — without any human annotation.

## Why Pretraining Objectives Matter

The choice of pretraining objective has profound effects:

- **What the model learns**: Masked language modeling develops rich bidirectional representations; causal language modeling develops strong generative and in-context learning abilities.
- **Architecture constraints**: Some objectives require bidirectional attention (BERT), others unidirectional (GPT), others encoder-decoder (T5).
- **Downstream transfer**: A model pretrained with generation objectives transfers more naturally to text generation tasks; models pretrained with discrimination objectives transfer better to classification.
- **Sample efficiency**: Some objectives are more informative per token, training faster or reaching higher downstream performance with the same compute.

## Causal Language Modeling (CLM)

**Causal language modeling** — also called **autoregressive language modeling** — is the original transformer pretraining objective, used by GPT and its successors.

### Formulation

Given a sequence of tokens $x_1, x_2, \ldots, x_T$, the model is trained to predict each token given all previous tokens:

$$\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

This is a **next-token prediction** objective — the simplest possible language modeling task. The training signal is efficient: every token in the sequence contributes a loss term.

### Architectural Constraint

CLM requires a **decoder-only (causal) transformer** with a causal attention mask that prevents each position from attending to future positions. This ensures the model cannot "cheat" by looking ahead.

### Why CLM Produces Powerful Generalists

The GPT series demonstrated that large-scale CLM pretraining produces models with remarkable in-context learning abilities: given a few examples of a task in the prompt, the model extrapolates to new examples without any weight updates. This **few-shot prompting** capability emerges from CLM at sufficient scale and is not observed in masked objectives.

The current dominant paradigm — GPT-4, Claude, Gemini, Llama, Mistral — uses CLM with decoder-only architectures, trained at trillion-token scale.

## Masked Language Modeling (MLM)

**Masked language modeling** is the pretraining objective introduced by **BERT** (Devlin et al., 2018). It produces strong bidirectional representations that were state of the art for language understanding tasks for years after publication.

### Formulation

A random 15% of input tokens are selected and processed as follows:

- 80% are replaced with a special `[MASK]` token.
- 10% are replaced with a random token from the vocabulary.
- 10% are kept unchanged.

The model is trained to predict the original token at masked positions:

$$\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i \mid \tilde{x})$$

where $\tilde{x}$ is the masked sequence.

### Bidirectional Attention

Unlike CLM, MLM allows each position to attend to all other positions (bidirectional attention). This is essential for understanding: predicting a masked word correctly often requires context from both before and after it.

> "The bank was steep and covered in \_\_\_ ."

Predicting the masked word requires understanding "bank" (riverbank, not financial), which requires the following word "steep" — impossible with unidirectional attention.

### BERT's Full Pretraining Objective

BERT combined MLM with **Next Sentence Prediction (NSP)**: given two text segments, predict whether the second is the actual continuation of the first (50% positive, 50% random). The `[CLS]` token representation is used for this binary classification.

NSP was intended to help tasks requiring sentence-pair understanding (question answering, entailment). However, subsequent work (RoBERTa, ALBERT) found NSP adds little or no benefit and was dropped.

**RoBERTa** (Liu et al., 2019) showed that simply training BERT longer, on more data, with larger batches, dynamic masking, and without NSP substantially improved performance — demonstrating that BERT was significantly undertrained.

### Limitation: Training-Inference Mismatch

The `[MASK]` token appears during training but never during fine-tuning or inference. This distribution shift is an inherent limitation of MLM. The 10% unchanged and 10% random token replacement was a partial mitigation by BERT's original authors.

## Span Corruption (T5)

**T5** (Raffel et al., 2020) introduced **span corruption** — corrupting contiguous spans of tokens rather than individual tokens, and training the model to reconstruct the corrupted spans.

### Formulation

- Randomly select 15% of tokens as corruption targets.
- Group consecutive selected tokens into **spans** (average span length: 3 tokens).
- Replace each span with a unique sentinel token (`<extra_id_0>`, `<extra_id_1>`, ...).
- The decoder must produce the original content of all sentinel spans.

For example:

- Input: `Thank you for <extra_id_0> me to your <extra_id_1> party.`
- Target: `<extra_id_0> inviting <extra_id_1> neighborhood <extra_id_2>`

### Encoder-Decoder Architecture

Span corruption is designed for **encoder-decoder transformers**. The corrupted sequence is fed to the encoder; the decoder generates the original spans autoregressively. This architecture is naturally suited for sequence-to-sequence tasks (translation, summarization, question answering).

T5's **"Text-to-Text"** framework unifies all NLP tasks into a single text generation format — every task is framed as producing a text output from a text input — enabling a single model to be fine-tuned on any task.

## Permutation Language Modeling (XLNet)

**XLNet** introduced **permutation language modeling** — training the model to predict tokens in all possible orderings of the sequence, not just left-to-right.

For a sequence of length $T$, there are $T!$ possible orderings. XLNet samples a random permutation $z$ and trains:

$$\mathcal{L}_{PLM} = -\mathbb{E}_{z \sim \mathcal{Z}_T} \left[ \sum_{t=1}^{T} \log P(x_{z_t} \mid x_{z_{<t}}) \right]$$

This gives XLNet the benefits of MLM (bidirectional context, since predicting $z_t$ can use tokens from both left and right of the original position) while maintaining an autoregressive factorization (avoiding the training-inference mismatch of masking).

XLNet uses a **two-stream self-attention** mechanism to handle the fact that the model must predict each token without seeing the token itself.

## Replaced Token Detection (ELECTRA)

**ELECTRA** (Clark et al., 2020) addresses the inefficiency of MLM: only 15% of tokens contribute loss. ELECTRA uses a **generator-discriminator** setup inspired by GANs.

### Architecture

- **Generator**: A small BERT-like MLM model that fills in `[MASK]` tokens.
- **Discriminator** (the main model): Receives the generator's output and must determine, **for every token**, whether it is the original token or a generated replacement.

The discriminator is trained with binary cross-entropy at every token position — 100% token utilization vs. 15% for MLM.

$$\mathcal{L}_{RTD} = -\sum_{t=1}^{T} \left[ \mathbb{1}[x_t = \hat{x}_t] \log P(\text{original} \mid \hat{x}) + \mathbb{1}[x_t \neq \hat{x}_t] \log P(\text{replaced} \mid \hat{x}) \right]$$

ELECTRA achieves substantially better downstream performance per FLOP than BERT — with the same compute budget, the discriminator significantly outperforms equivalently-sized BERT models.

## Contrastive Objectives

**Contrastive learning** defines a pretraining objective in terms of similarity: representations of similar examples should be close in embedding space; representations of dissimilar examples should be far apart.

### SimCLR (Vision)

**SimCLR** (Chen et al., 2020) applies two random augmentations to the same image, creating two views. The **NT-Xent** (normalized temperature-scaled cross-entropy) loss trains the encoder to map the two views of the same image to similar representations, while pushing apart representations of different images:

$$\mathcal{L}_{SimCLR} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where $z_i, z_j$ are the representations of the two views of the same image, and $\tau$ is a temperature hyperparameter.

### CLIP (Contrastive Language-Image Pretraining)

**CLIP** trains an image encoder and text encoder contrastively: for a batch of $(image, text)$ pairs, maximize similarity between matched pairs and minimize similarity between unmatched pairs. This produces a joint vision-language embedding space enabling zero-shot classification and cross-modal retrieval.

### SimCSE (Sentence Embeddings)

**SimCSE** applies contrastive learning to sentence embeddings by using **dropout as augmentation**: pass the same sentence through the encoder twice with different dropout masks to create two views. The same sentence is its own positive; different sentences in the batch are negatives. This dramatically improves sentence embedding quality.

## Masked Image Modeling (MIM)

Vision transformers can be pretrained with analogues to MLM:

**BEiT** (Bao et al., 2021): Tokenize images into discrete visual tokens using a pre-trained VQVAE; randomly mask patches and predict the discrete visual tokens of masked patches.

**MAE (Masked Autoencoders)** (He et al., 2022): Randomly mask a very high fraction of image patches (75–80%) and train the model to reconstruct raw pixel values at masked positions. The high masking rate creates a difficult enough task that the model must learn rich representations — unlike text MLM, predicting image pixels from context does not have shortcut solutions because adjacent pixels are highly correlated.

**SimMIM**: A simplified MIM that predicts raw pixels (like MAE) but uses a full ViT without the encoder-only trick of MAE, with competitive performance at lower complexity.

## Denoising Objectives

**Denoising autoencoders** corrupt input data and train the model to reconstruct the original. This generalizes both MLM (masking is a form of corruption) and span corruption.

**BART** (Lewis et al., 2020) applies multiple types of noise to text and trains the encoder-decoder to reconstruct the original:

- **Token masking**: Replace tokens with `[MASK]` (like BERT).
- **Token deletion**: Remove tokens entirely.
- **Text infilling**: Replace a span of tokens with a single `[MASK]`.
- **Sentence permutation**: Shuffle the order of sentences.
- **Document rotation**: Rotate the document so it begins at a random token.

BART found that text infilling was the most effective single corruption strategy. The denoising objective with an encoder-decoder architecture makes BART excel at abstractive summarization, translation, and dialogue generation.

## Prefix Language Modeling

**Prefix LM** is a hybrid between CLM and MLM: a prefix of the sequence attends bidirectionally (like an encoder), and the suffix is generated autoregressively (like a decoder).

**UniLM** trained a single model on three objectives simultaneously: unidirectional LM, bidirectional LM, and sequence-to-sequence LM — controlled by attention masks. This enables both understanding and generation from a single model.

**T5** with a prefix architecture (PrefixLM): the input is treated as a prefix with bidirectional attention; the output is generated autoregressively. This is the natural fit for encoder-decoder architectures processing (input, output) pairs.

## Choosing a Pretraining Objective

| Objective | Architecture | Strengths | Weaknesses |
|-----------|-------------|-----------|------------|
| **CLM (GPT)** | Decoder-only | Generation, in-context learning, versatile | Unidirectional, less efficient for understanding |
| **MLM (BERT)** | Encoder-only | Bidirectional, strong understanding | No generation, masking mismatch |
| **Span Corruption (T5)** | Encoder-decoder | Seq2seq, unified text-to-text | Larger architecture, slower inference |
| **RTD (ELECTRA)** | Encoder-only | Very efficient per-FLOP | Complex training setup |
| **Contrastive (CLIP)** | Dual encoder | Cross-modal, zero-shot transfer | No generation capability |
| **MIM (MAE)** | Encoder (vision) | High-quality visual representations | Image-only |

The field has largely converged on **CLM with decoder-only transformers** as the dominant paradigm for general-purpose language models, with encoder-based and encoder-decoder models remaining important for specific applications requiring efficient encoding (retrieval, classification) or structured generation (translation, summarization).

Understanding these objectives is foundational for evaluating published models, designing pretraining recipes for new domains, and debugging transfer learning failures.
