---
title: Capsule Networks Deep Dive
description: How capsule networks improve on CNNs by encoding pose information and part-whole relationships, dynamic routing, and why they haven't yet displaced convolutional networks.
---

Capsule networks (CapsNets), introduced by Geoffrey Hinton et al. in 2017, proposed a fundamentally different approach to visual representation. Rather than pooling features into scalar activations, capsules encode the pose, orientation, and properties of objects as vectors — enabling equivariance and part-whole relationship reasoning that standard CNNs lack.

## The Problem with CNNs

Convolutional neural networks achieve remarkable performance, but they have a structural weakness: **pooling is invariant, not equivariant**. Max pooling discards spatial information to gain translation invariance. This means:

- A CNN can recognize a face even if the eyes are where the mouth should be — it detects the presence of parts, not their spatial relationships.
- CNNs require massive amounts of training data with many viewpoints to learn that an object looks the same from different angles (viewpoint invariance by memorization rather than geometry).
- They struggle with "part-whole" relationships — understanding that a nose, eyes, and mouth in the right spatial configuration constitute a face.

## What Is a Capsule?

A **capsule** is a group of neurons whose collective output is a vector (rather than a scalar). The components of this vector encode:
- **Magnitude (length):** The probability that the entity the capsule represents is present.
- **Direction (orientation):** The instantiation parameters — pose, position, scale, velocity, color — of that entity.

By encoding pose in the vector's direction, capsules can represent **equivariance**: when the object moves or rotates, the capsule's output vector changes in a predictable, structured way rather than losing the information.

## Architecture: CapsNet

The original CapsNet for MNIST consists of:

### 1. Convolutional Layer
A standard convolutional layer detects basic features (edges, curves) in the input image. Output: feature maps.

### 2. Primary Capsules
The convolutional output is reshaped into a grid of 8-dimensional capsules. Each capsule represents a low-level visual entity (a detected edge or blob) with its instantiation parameters.

A **squashing function** normalizes capsule output vectors so their length (the probability) stays between 0 and 1:

$$\mathbf{v}_j = \frac{\|\mathbf{s}_j\|^2}{1 + \|\mathbf{s}_j\|^2} \cdot \frac{\mathbf{s}_j}{\|\mathbf{s}_j\|}$$

### 3. Dynamic Routing
This is the core innovation. Instead of fixed pooling, capsules in a lower layer **vote** for which higher-level capsule they should send their output to, based on agreement.

**Process (routing by agreement):**
1. Each lower capsule `i` predicts what the output of higher capsule `j` should be via a learned transformation matrix: $\hat{\mathbf{u}}_{j|i} = W_{ij}\mathbf{u}_i$
2. Coupling coefficients $c_{ij}$ (initialized uniformly) determine how much capsule `i` sends to capsule `j`.
3. The input to higher capsule `j` is a weighted sum: $\mathbf{s}_j = \sum_i c_{ij} \hat{\mathbf{u}}_{j|i}$
4. After squashing, agreement is measured: if $\hat{\mathbf{u}}_{j|i}$ is similar to the output $\mathbf{v}_j$, the coupling $c_{ij}$ increases (via softmax update).
5. Steps 3–4 iterate 3–5 times.

This is an EM-like iterative agreement process: capsules that agree on what higher-level entity is present reinforce each other, while disagreeing capsules reduce their coupling.

### 4. Digit Capsules
One 16-dimensional capsule per class. The length of each capsule vector is the probability of that digit being present. For classification, the class with the longest capsule vector wins.

### 5. Reconstruction Decoder
A regularization mechanism: the winning capsule's 16D vector is fed to a decoder network that reconstructs the input image. The reconstruction loss penalizes the network if the capsule's pose parameters don't accurately describe the actual digit's appearance.

## Advantages Over CNNs

- **Viewpoint equivariance:** Pose information is preserved, not discarded. Capsules can generalize to unseen viewpoints with less data.
- **Part-whole relationships:** Dynamic routing explicitly models which lower-level parts belong to which higher-level wholes.
- **Robustness to affine transformations:** CapsNets generalize better to transformed inputs (rotation, scaling) that weren't seen during training.
- **Better interpretability:** Capsule activation vectors have geometric meaning — you can inspect what instantiation parameters a capsule is encoding.

## Limitations and Why CapsNets Haven't Dominated

Despite theoretical elegance, CapsNets face practical challenges:

- **Computational cost:** Dynamic routing (iterative, non-parallelizable) is significantly slower than pooling.
- **Scaling difficulty:** CapsNets were demonstrated on MNIST (28×28, 10 classes). Scaling to ImageNet-class tasks (224×224, 1,000 classes) has proven difficult.
- **Memory:** The all-pairs coupling between capsule layers grows quadratically.
- **No clear SOTA results on large benchmarks:** Despite significant research effort, CapsNets have not exceeded CNNs or Vision Transformers on standard vision benchmarks.

## Later Developments

- **EM Routing (Matrix Capsules, 2018):** Hinton replaced dynamic routing with an Expectation-Maximization algorithm and matrix capsules, achieving better CIFAR-10 results.
- **Stacked Capsule Autoencoders (SCAE, 2019):** Unsupervised capsule learning that achieves stronger part-based decomposition.
- **CapsFormer:** Attempts to combine capsules with transformer architectures for scalability.

## Current Status

Capsule networks remain an active research topic with compelling theoretical properties — especially for tasks requiring geometric reasoning, few-shot generalization, or robust part-based recognition. However, practical adoption remains limited compared to CNNs and Vision Transformers, primarily due to scaling challenges and the dominance of data-driven approaches that work well even without explicit equivariance inductive biases.
