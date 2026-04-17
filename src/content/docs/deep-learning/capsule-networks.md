---
title: Capsule Networks
description: Understand Capsule Networks — Geoffrey Hinton's architectural alternative to CNNs that uses dynamic routing to preserve spatial hierarchies and overcome limitations of max-pooling.
---

**Capsule Networks (CapsNets)** are a neural network architecture proposed by Geoffrey Hinton, Sara Sabour, and Nicholas Frosst (2017) as a fundamental rethinking of how spatial information should be represented in vision models. They address one of the core weaknesses of Convolutional Neural Networks: the loss of positional and relational information caused by pooling.

## The Problem with CNNs and Pooling

CNNs use **max-pooling** to achieve translational invariance — the ability to recognize an object regardless of where it appears in an image. While effective, this comes at a cost:

- Pooling **discards spatial information** — exact positions, orientations, and relationships between features are thrown away.
- A CNN can correctly classify a "face" even if the eyes, nose, and mouth are in completely wrong relative positions — it detects the *presence* of parts, not their *arrangement*.
- They require large amounts of data to learn robust representations because they do not exploit the geometrical structure of scenes.

Hinton argued that the brain does not work this way. It maintains **equivariance** (tracking how an object's properties change with viewpoint) rather than just **invariance** (ignoring those changes).

## What Is a Capsule?

A **capsule** is a group of neurons whose activity vector represents both:

- **The probability** that a particular entity (part, object) is present — encoded as the **length** of the vector ($\|v\| \in [0, 1]$).
- **The instantiation parameters** of that entity — encoded as the **orientation** of the vector (pose, scale, velocity, color, etc.).

This is a key departure from CNNs, where a single scalar activation represents "how much" of a feature is present, discarding all spatial information.

## Squashing Function

The activation function for capsule outputs is a **squashing function** that preserves orientation but normalizes length to $[0, 1]$:

$$v_j = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \cdot \frac{s_j}{\|s_j\|}$$

where $s_j$ is the total input to capsule $j$ and $v_j$ is the output vector.

- Short vectors are squashed nearly to zero.
- Long vectors are squashed to just below 1.
- The orientation is preserved exactly.

## Dynamic Routing by Agreement

The core algorithmic innovation of CapsNets is **routing by agreement** — a mechanism that decides how strongly each lower-level capsule should "vote for" each higher-level capsule.

Instead of static pooling, the routing algorithm iteratively updates connection weights $c_{ij}$ between capsule $i$ (lower level) and capsule $j$ (higher level):

1. Lower capsule $i$ generates a **prediction vector** $\hat{u}_{j|i}$ for what higher capsule $j$'s output should be:
$$\hat{u}_{j|i} = W_{ij} u_i$$

2. If many lower capsules predict a similar output for capsule $j$, they are in **agreement** — their votes are amplified.

3. The routing coefficients $c_{ij}$ are updated based on the dot product (agreement) between predictions and the actual capsule $j$ output:

$$b_{ij} \leftarrow b_{ij} + \hat{u}_{j|i} \cdot v_j$$

$$c_{ij} = \text{softmax}(b_{ij})$$

This process iterates 2–3 times per forward pass. The result: parts that **spatially agree** on the presence and pose of a higher-level entity are strongly routed to it.

## Architecture of the Original CapsNet (MNIST)

```
Input (28×28×1)
    ↓
Conv1: 256 feature maps, 9×9 kernel → (20×20×256)
    ↓
PrimaryCaps: 32 capsule types, 8D vectors, 9×9 kernel → (6×6×32 capsules, each 8D)
    ↓
DigitCaps: 10 capsule types (one per digit class), 16D vectors
    ↓
Routing by Agreement (3 iterations)
    ↓
Output: 10 capsule vectors; length = class probability
```

A **reconstruction network** (decoder) regularizes training by trying to reconstruct the input from the active capsule's vector.

## Margin Loss

Instead of softmax cross-entropy, CapsNets use a **margin loss** that allows multiple classes to be present simultaneously:

$$L_k = T_k \max(0, m^+ - \|v_k\|)^2 + \lambda (1 - T_k) \max(0, \|v_k\| - m^-)^2$$

where:

- $T_k = 1$ if class $k$ is present.
- $m^+ = 0.9$, $m^- = 0.1$ are margin thresholds.
- $\lambda = 0.5$ down-weights the absent-class loss.

## Results and Capabilities

On MNIST, the original CapsNet achieved **~0.25% error** with 10× fewer parameters than a comparable CNN.

A remarkable capability: **equivariance to affine transformations**. When you add a small amount to a dimension of the DigitCaps vector, the decoded reconstruction smoothly changes a specific property (thickness, rotation, width) of the digit — evidence that the vector dimensions encode meaningful pose information.

## Limitations and Challenges

Despite the theoretical appeal, CapsNets face significant practical obstacles:

- **Computational cost** — Dynamic routing is much more expensive than pooling; the iterative procedure is slow to parallelize on GPUs.
- **Scaling difficulty** — Extending CapsNets to ImageNet-scale datasets and complex scenes has proven difficult. Most strong results remain on simple datasets (MNIST, SmallNORB).
- **Routing instability** — The routing algorithm can be sensitive to initialization and has convergence challenges.
- **Crowded scenes** — The "one capsule per entity" assumption breaks down in cluttered images with overlapping objects.

## Later Developments

Researchers have proposed several improvements:

- **EM Routing** (Hinton et al., 2018) — Uses Expectation-Maximization instead of agreement-based routing; achieved strong results on SmallNORB.
- **Self-Routing Capsules** — Replace dynamic routing with attention-based routing for efficiency.
- **Efficient-CapsNet** — Lightweight variant achieving competitive MNIST results with far fewer parameters.
- **CapsFormer** — Attempts to combine capsule ideas with Transformer architectures.

## Significance

Despite not yet achieving the CNNs' and ViTs' dominance on large-scale benchmarks, Capsule Networks remain a deeply influential idea. They crystallized important critiques of standard architectures and motivated a broader research agenda around:

- **Equivariant representations** in vision.
- **Part-whole hierarchies** in visual understanding.
- **Spatial reasoning** beyond what pooling-based nets support.

Hinton continued to advocate for moving beyond backpropagation itself — the Forward-Forward Algorithm (2022) represents a further evolution of his ideas about biologically-inspired learning without global gradient backprop.
