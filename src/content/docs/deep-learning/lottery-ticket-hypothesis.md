---
title: The Lottery Ticket Hypothesis
description: Understand the Lottery Ticket Hypothesis — the idea that dense neural networks contain small sparse subnetworks that can be trained in isolation to match the full network's accuracy — and its implications for pruning and efficient learning.
---

The Lottery Ticket Hypothesis (LTH), proposed by Frankle and Carlin (2019), states that a randomly initialized dense neural network contains a small sparse subnetwork — a "winning ticket" — such that when trained from its original initialization, this subnetwork can match the full network's accuracy in a comparable number of steps. This hypothesis reframed how researchers think about pruning, initialization, and the inductive biases built into neural network training.

## The Core Hypothesis

Formally, given a dense network $f(x; \theta)$ with initialization $\theta_0$, there exists a mask $m \in \{0,1\}^{|\theta|}$ such that the sparse subnetwork $f(x; m \odot \theta)$ trained from its original weights $m \odot \theta_0$ achieves accuracy comparable to the full network.

The mask $m$ is called a **ticket**, and the combination $(m, \theta_0)$ is a **winning ticket** if the subnetwork trains efficiently.

The key insight is that the **initial weights matter**. A random re-initialization of the sparse subnetwork (keeping the structure but discarding the original values) typically performs much worse — suggesting that the specific initialization $\theta_0$ captured by the winning ticket is essential to its trainability.

## Finding Winning Tickets: Iterative Magnitude Pruning

Frankle and Carlin proposed **Iterative Magnitude Pruning (IMP)** to find winning tickets:

1. Initialize a network with random weights $\theta_0$
2. Train the full network to convergence, obtaining $\theta_T$
3. Prune the $p$% of weights with the smallest absolute magnitude
4. **Reset** surviving weights to their values at $\theta_0$ (the "rewind")
5. Repeat from step 2 with the pruned network

This process is iterated until the target sparsity is reached. At each round, a fraction $p$ of remaining weights is pruned:

$$\text{Remaining weights after } k \text{ rounds} = |\theta| \cdot (1 - p)^k$$

The resulting sparse network — initialized with the rewound original values — is the winning ticket candidate.

### One-Shot vs. Iterative Pruning

One-shot pruning (prune once at the end) finds tickets that perform worse than iterative pruning. This suggests that gradual structure discovery, with periodic resets, is important for finding high-quality subnetworks.

## Why Do Winning Tickets Exist?

### Fortunate Initialization

Not all initializations are equally trainable. In a large network, by the birthday paradox of neural network weight space, some subsets of weights happen to be initialized favorably — with the right signs, magnitudes, and relative alignment — to learn particular features quickly. The large network provides enough "lottery tickets" that at least some will be winners.

### Implicit Regularization

The pruning-and-reset procedure implicitly regularizes the subnetwork. By working with fewer parameters, the winning ticket cannot memorize as easily and is biased toward learning robust features.

### Loss Landscape

Winning tickets tend to reside near flat minima of the loss landscape, which are associated with better generalization. The sparse structure may constrain the subnetwork to a region of weight space with favorable geometry.

## Matching the Full Network

The original paper defined winning tickets by three criteria (all must hold at the target sparsity):

1. **Early stopping step**: The subnetwork reaches peak test accuracy in $\leq$ the number of steps the full network needed
2. **Test accuracy**: The subnetwork matches or exceeds the full network's final test accuracy
3. **Sparsity**: The subnetwork is significantly smaller (e.g., 90%+ weights removed)

Winning tickets at high sparsity (up to ~90–95%) were found for MNIST and CIFAR-10 tasks with small networks. For larger networks like ResNets and VGGs on CIFAR-10/100, tickets at 70–80% sparsity could match full network accuracy.

## Stability and the Late Rewinding Fix

A major complication arose when researchers tried to apply LTH to large networks (ResNets, Transformers, BERT). Frankle et al. (2020) found that for larger models, rewinding to the true initialization ($\theta_0$) failed — tickets found this way did not match full network performance.

The fix was **late rewinding**: instead of rewinding to $\theta_0$, rewind to a checkpoint $\theta_k$ taken after a small number of early training steps:

$$\text{Rewind to } \theta_k, \quad k \ll T$$

This stabilized ticket finding for large networks and became the standard variant. The insight is that early training establishes a coarse structure (the "ticket") which is then refined; very early steps set this structure.

## Extensions and Variants

### Strong Lottery Tickets

Ramanujan et al. (2020) showed that a sufficiently overparameterized random network contains a subnetwork that — **without any training** — matches the accuracy of a trained smaller network. This is the **Strong Lottery Ticket Hypothesis**:

- No training needed, only weight masking
- Requires greater overparameterization than the standard LTH
- Proven theoretically for two-layer networks under certain conditions

### Linear Mode Connectivity

Frankle et al. (2020) introduced **linear mode connectivity** (LMC) as a lens for understanding winning tickets. Two solutions $\theta_A$ and $\theta_B$ are linearly mode connected if the loss along the linear interpolation $\alpha \theta_A + (1-\alpha) \theta_B$ is approximately flat. Stable winning tickets are linearly mode connected to the full network solution, indicating they reside in the same basin.

### Supermasks

Zhou et al. (2019) introduced **supermasks** — binary masks applied to a random, fixed network that achieve non-trivial accuracy. This demonstrates that the structure discovered by masking (independent of weight training) encodes task-relevant information.

### Network Tickets Across Tasks

Desai et al. (2019) found that tickets found for one task can **transfer** to related tasks. BERT winning tickets transfer well across NLP tasks, suggesting that tickets encode generalizable feature detectors rather than task-specific solutions.

## Implications for Pruning

Traditional neural network pruning proceeds as:

1. Train a large network
2. Prune
3. Fine-tune

The LTH suggests an alternative view:

1. The pruned subnetwork's **initial weights** matter as much as its structure
2. Finding the mask early (before full training) and resetting could yield better-performing sparse networks
3. This motivates **training-aware pruning** rather than post-hoc compression

### Structured vs. Unstructured Tickets

Most LTH work uses **unstructured pruning** (individual weights). This creates sparse weight matrices that are difficult to accelerate on GPUs without specialized hardware. **Structured pruning** (entire neurons, heads, or channels) yields hardware-friendly sparsity but finds weaker tickets.

## Relationship to Neural Architecture Search

The LTH can be viewed as a form of architecture search:

- The mask $m$ defines an architecture (a subgraph of the full network)
- IMP discovers this architecture by training, pruning, and iterating
- Unlike standard NAS, the architecture and initialization are found together

This connection has led to methods like **differentiable pruning** that treat the mask as a continuous variable:

$$m_i = \sigma\left(\frac{\log \alpha_i - \log(1 - \alpha_i) + \epsilon}{\tau}\right)$$

where $\alpha_i$ are learnable parameters and $\tau$ is a temperature controlling sparsity.

## Practical Challenges

### Computational Cost

IMP requires training the full network multiple times (once per pruning round), making it expensive for large models.

### Hyperparameter Sensitivity

The pruning fraction per round, learning rate schedule, and rewind point all significantly affect ticket quality.

### Theoretical Gaps

While the existence of winning tickets has been proven theoretically for simple models (two-layer networks), a complete theory for deep networks remains open.

## Summary Table

| Variant | Rewind Point | Training Required | Best For |
| --- | --- | --- | --- |
| Standard LTH | $\theta_0$ | Yes | Small networks |
| Late Rewinding | $\theta_k$, small $k$ | Yes | Large networks, Transformers |
| Strong LTH | N/A (random) | No | Theoretical analysis |
| Supermasks | Fixed random $\theta$ | No | Understanding structure |

## Impact and Legacy

The Lottery Ticket Hypothesis reoriented pruning research from a compression-only perspective to a question about learning dynamics and initialization. It demonstrated that:

- Large networks are not intrinsically necessary — they provide a rich search space for finding good subnetworks
- Initialization is a first-class concern in network design
- Sparse networks trained from scratch can match dense ones, if initialized correctly

These insights continue to influence work on sparse training, efficient neural architecture design, and the theoretical understanding of overparameterization.
