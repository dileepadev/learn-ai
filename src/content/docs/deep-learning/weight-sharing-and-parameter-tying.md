---
title: Weight Sharing and Parameter Tying in Neural Networks
description: Learn how weight sharing and parameter tying reduce model size while preserving capacity — covering ALBERT's cross-layer sharing, Universal Transformers, tied input-output embeddings, convolutional weight sharing, Siamese networks, and the relationship between parameter efficiency and generalization.
---

Modern neural networks can contain billions of parameters — but many of those parameters are redundant. **Weight sharing** (also called **parameter tying**) constrains multiple connections or layers to use the same learned values, drastically reducing model size while often preserving — or even improving — generalization. This technique underpins some of the most important architectures in NLP and computer vision: BERT's tied embedding matrices, ALBERT's cross-layer attention sharing, Universal Transformers, and Siamese networks for similarity learning.

## What Is Weight Sharing?

**Weight sharing** means that two or more components in a neural network use the **same parameter tensor**, so when parameters are updated via backpropagation, the gradient flows back to a single shared parameter rather than separate copies. The update to the shared parameter is the **sum of gradients** from all components that use it.

Formally, if parameters $\theta_A = \theta_B = \theta$ (tied), the gradient of the loss $\mathcal{L}$ with respect to $\theta$ is:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \theta_A} + \frac{\partial \mathcal{L}}{\partial \theta_B}$$

This contrasts with **untied** parameters, where $\theta_A$ and $\theta_B$ are initialized identically but updated independently. Weight sharing enforces a permanent structural constraint — not just an initialization choice.

## Tied Input-Output Embeddings

One of the oldest and most widely adopted forms of weight sharing in NLP is **tying the input embedding matrix to the output projection matrix** (Press and Wolf, 2017).

A language model maps:

- Input tokens $\to$ embedding vectors via matrix $E \in \mathbb{R}^{V \times d}$ (vocabulary size $V$, hidden dim $d$).
- Hidden state $h \in \mathbb{R}^d \to$ logits over vocabulary via output projection $W \in \mathbb{R}^{d \times V}$.

With tying, $W = E^\top$ — the output projection is the transpose of the input embedding. This:

- Reduces parameters by $V \times d$ (e.g., 25,000 parameters saved for $V=50,\!000$, $d=768$).
- Ensures that token representations are consistent between input processing and output prediction.
- Improves perplexity on language modeling benchmarks — Press and Wolf showed ~3 perplexity points improvement on PTB and WikiText.

This technique is used in GPT-2, T5, BERT, and virtually all modern transformer language models.

## ALBERT: Cross-Layer Attention Sharing

**ALBERT** (A Lite BERT, Lan et al., 2020) introduced two parameter reduction strategies:

### Factorized Embedding Parameterization

Instead of tying the embedding dimension to the hidden dimension ($E = d$), ALBERT uses a smaller embedding dimension $E = 128$ and projects to the hidden dimension $d = 1024$ with a learned matrix $P \in \mathbb{R}^{E \times d}$. This reduces the embedding parameters from $V \times d$ to $V \times E + E \times d$:

$$\text{Parameters saved} = V \times d - (V \times E + E \times d) = E(V - d)$$

For BERT-Large with $V = 30,\!000$, $d = 1024$, $E = 128$: saves $\approx 27$M parameters.

### Cross-Layer Parameter Sharing

All $L$ transformer layers share the **same set of parameters** — every layer uses the same attention weight matrices and feed-forward weight matrices. Instead of $L \times P$ parameters (one set per layer), the model uses $P$ parameters regardless of depth.

Cross-layer sharing in ALBERT-xxlarge reduces parameters from ~340M (BERT-Large) to ~235M while using more layers (24 vs. 12). Empirically, ALBERT maintains 96–98% of BERT's downstream task performance with this drastic reduction.

The surprising implication is that **depth provides representational benefit independent of parameter count** — iteratively applying the same transformation $L$ times extracts richer representations than applying $L$ different transformations once each, at the cost of depth without parameter growth.

## Universal Transformers

**Universal Transformers** (Dehghani et al., 2019) take cross-layer sharing to its logical extreme: the transformer is formulated as a **recurrent** computation over a fixed set of shared parameters applied repeatedly across depth:

$$h^t = \text{Transformer-Block}(h^{t-1}; \theta)$$

where $h^0$ is the input and $\theta$ is the single shared parameter set applied at each "step" $t$. The number of steps is either fixed or determined by an **Adaptive Computation Time (ACT)** mechanism that halts processing of individual tokens when they are "confident enough."

Universal Transformers are:

- **Turing complete** under the ACT mechanism — they can simulate any algorithm given sufficient steps.
- More **data-efficient** than standard transformers on small datasets (LAMBADA, bAbI).
- Weaker on large-scale pretraining benchmarks where non-shared transformers benefit from specializing each layer.

## Siamese Networks and Contrastive Learning

**Siamese networks** (Bromley et al., 1993) use two or more identical subnetworks with **tied weights** to encode pairs of inputs into a shared representation space. The key property is that tied weights guarantee that the distance metric in the output space is consistent — if the same input is fed to both branches, they produce the same output.

### Applications of Siamese Architectures

- **Face verification**: two face images are encoded by the same CNN; a distance metric classifies them as the same/different person.
- **Sentence similarity**: two sentences are encoded by the same transformer; cosine similarity measures semantic overlap.
- **Contrastive learning** (SimCLR, MoCo): two augmented views of the same image are encoded by the same backbone; the model is trained to bring their representations close.
- **Self-supervised knowledge distillation** (BYOL, SimSiam): Siamese structures enable self-supervised learning without negative samples.

### Self-Supervised Collapse and Stop-Gradient

A surprising finding from **SimSiam** (Chen and He, 2021) is that Siamese networks can be trained for self-supervised learning **without negative pairs** by using a stop-gradient on one branch:

$$\mathcal{L} = -\frac{1}{2} \left[ D(p_1, \text{sg}(z_2)) + D(p_2, \text{sg}(z_1)) \right]$$

where $D$ is cosine similarity, $p$ is a prediction head output, $z$ is the encoder output, and $\text{sg}(\cdot)$ denotes stop-gradient. Without stop-gradient, the network collapses to a constant output (trivial solution). With stop-gradient, weight sharing creates an implicit expectation-maximization structure that prevents collapse.

## Weight Sharing in Convolutional Networks

**Convolutional neural networks** are fundamentally built on weight sharing: the same filter is applied at every spatial location in the input. This is the original and most successful instantiation of weight sharing:

- A CNN layer with filter size $k \times k$, $C_{in}$ input channels, and $C_{out}$ output channels has $k^2 \times C_{in} \times C_{out}$ parameters regardless of the spatial dimensions of the input.
- Without weight sharing, each spatial location would need its own filter weights, requiring $H \times W \times k^2 \times C_{in} \times C_{out}$ parameters.
- The weight sharing encodes **translation equivariance**: the same feature detector is applied everywhere, allowing features learned from one part of the image to transfer to all other parts.

### Group Equivariant CNNs

**G-CNNs** (Cohen and Welling, 2016) extend this by sharing weights not only across spatial locations but also across **rotation and reflection symmetries**. A filter is simultaneously applied at multiple orientations, with tied weights enforcing that the same underlying feature is detected regardless of orientation.

## Tied Weights in Autoencoders

Autoencoders can tie the encoder and decoder weights: if the encoder is a matrix $W \in \mathbb{R}^{d \times n}$ (compressing from $n$ to $d$ dimensions), the decoder is $W^\top \in \mathbb{R}^{n \times d}$. **Tied autoencoders** enforce that the reconstruction operation is the transpose of the encoding operation:

- **Reduces parameters** by half (one matrix instead of two).
- **Regularizes** the encoder-decoder pair: the decoder cannot learn arbitrary invertions of the encoder, only the transpose.
- Encourages **symmetric feature detectors**: a feature that fires on a particular pattern in the input will also reconstruct that pattern in the output.

Tied weights in autoencoders were found to improve generalization in early deep learning work (Vincent et al., 2010) but are less commonly used in modern VAEs where encoder and decoder may have different architectures.

## Theoretical Perspective: Generalization and Inductive Bias

Weight sharing injects **inductive bias** by constraining the function class the model can represent. From a PAC learning perspective, a model with $P$ effective parameters (after sharing) has a generalization bound of approximately:

$$\text{Generalization error} \lesssim O\!\left(\sqrt{\frac{P \log(1/\delta)}{n}}\right)$$

Sharing reduces $P$, tightening the bound. But the constraint must match the true structure of the problem:

- **CNNs work** because natural images are approximately translation-invariant.
- **Cross-layer sharing in ALBERT works** because different layers in a transformer implement similar computations (attention + FFN) with different contexts.
- **Cross-layer sharing fails** in early language models where different layers must learn qualitatively different representations (syntax vs. semantics).

The success of weight sharing depends on the alignment between the sharing structure and the true symmetries of the task.

## Practical Considerations

### Memory vs. Compute Trade-off

Weight sharing reduces **memory** (fewer parameters to store and update) but does not necessarily reduce **compute** — a model with $L$ shared layers still performs $L$ forward passes. Universal Transformers trade parameter efficiency for computational depth.

### Gradient Accumulation

Shared parameters receive gradients from multiple components. For $L$ layers sharing parameters, the gradient magnitude is approximately $L$ times larger than for untied parameters. This requires adjusting the learning rate (typically scaling down by $\sqrt{L}$) to prevent instability.

### Fine-Tuning Behavior

Models with shared weights (ALBERT, Universal Transformers) fine-tune differently from unshared models: because all layers must change coherently (the same parameters are updated), fine-tuning may require lower learning rates and more careful hyperparameter tuning.

## Summary

Weight sharing and parameter tying reduce neural network parameter counts by enforcing structural constraints — the same parameters are used in multiple components. Tied input-output embeddings improve language model perplexity with zero accuracy cost. ALBERT's cross-layer sharing and factorized embeddings reduce BERT-scale models by ~4x with minimal performance loss. Universal Transformers generalize this to arbitrary depth with ACT. Siamese networks use weight tying to enforce consistent distance metrics for similarity and contrastive learning. CNNs demonstrate that translation equivariance via spatial weight sharing is a powerful inductive bias for visual data. The key insight across all these applications is that sharing encodes **structural symmetry** — it works when the shared parameters genuinely describe equivalent computations at different locations, steps, or roles.
