---
title: Deep Equilibrium Models
description: Understand Deep Equilibrium Models (DEQs) — a class of implicit-depth neural networks that find fixed points of a transformation rather than stacking explicit layers — covering fixed-point theory, implicit differentiation, and applications to sequence modeling and vision.
---

Deep Equilibrium Models (DEQs), introduced by Bai et al. (2019), are a radical departure from conventional feedforward networks. Instead of stacking $L$ explicit layers, a DEQ defines its output as the **fixed point** of a single transformation applied infinitely — or until convergence. This implicit depth makes DEQs memory-efficient, theoretically interesting, and surprisingly powerful for sequence modeling, vision, and scientific computing.

## The Core Idea

A standard $L$-layer network computes:

$$z^{(1)} = f_\theta(x, z^{(0)}), \quad z^{(2)} = f_\theta(x, z^{(1)}), \quad \ldots, \quad z^{(L)} = f_\theta(x, z^{(L-1)})$$

A DEQ instead finds the **fixed point** $z^*$ satisfying:

$$z^* = f_\theta(x, z^*)$$

The output is $z^*$ regardless of how many iterations were required to converge. Crucially, the transformation $f_\theta$ is the **same at every step** — equivalent to weight-tied infinite depth — but only one set of parameters is stored.

## Fixed-Point Iteration (Forward Pass)

The forward pass solves the fixed-point equation using a root-finding method. Writing $g(z) = f_\theta(x, z) - z$, we want $g(z^*) = 0$.

### Broyden's Method

DEQ's original forward solver is **Broyden's method** — a quasi-Newton method for systems of equations. It maintains an approximate Jacobian inverse and updates it via rank-1 updates at each step:

$$B_{k+1}^{-1} = B_k^{-1} + \frac{(\Delta z_k - B_k^{-1} \Delta g_k) \Delta z_k^T B_k^{-1}}{\Delta z_k^T B_k^{-1} \Delta g_k}$$

Broyden converges superlinearly and avoids computing the full Jacobian — critical for large hidden states.

### Anderson Mixing

**Anderson mixing** (Anderson, 1965) is another popular solver that accelerates fixed-point iteration by maintaining a history of the last $m$ iterates and solving for the optimal linear combination:

$$z_{k+1} = \sum_{i=0}^m \alpha_i f_\theta(x, z_{k-i}), \quad \sum_{i=0}^m \alpha_i = 1$$

The coefficients $\{\alpha_i\}$ are chosen to minimize the residual norm, making convergence significantly faster than naive fixed-point iteration.

### Convergence Criterion

Iteration continues until:

$$\frac{\|z_{k+1} - z_k\|_2}{\|z_k\|_2 + \epsilon} < \delta$$

for tolerance $\delta$ (typically $10^{-6}$). The number of iterations varies per input — a form of adaptive computation depth.

## Backward Pass via Implicit Differentiation

The key theoretical insight that makes DEQs trainable is that the gradient of the loss with respect to parameters can be computed **without backpropagating through the forward solver's iterations**. This is the **implicit function theorem** applied to the fixed-point equation.

Given $z^* = f_\theta(x, z^*)$, differentiating both sides with respect to $\theta$:

$$\frac{\partial z^*}{\partial \theta} = \frac{\partial f_\theta}{\partial z^*} \frac{\partial z^*}{\partial \theta} + \frac{\partial f_\theta}{\partial \theta}$$

Solving for $\frac{\partial z^*}{\partial \theta}$:

$$\frac{\partial z^*}{\partial \theta} = \left(I - \frac{\partial f_\theta}{\partial z^*}\right)^{-1} \frac{\partial f_\theta}{\partial \theta}$$

The gradient of the scalar loss $\mathcal{L}$ with respect to $\theta$ via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial z^*} \left(I - J_{f}\right)^{-1} \frac{\partial f_\theta}{\partial \theta}$$

where $J_f = \frac{\partial f_\theta}{\partial z^*}$ is the Jacobian of $f$ at the fixed point.

### Computing the Vector-Jacobian Product

Rather than materializing $\left(I - J_f\right)^{-1}$ explicitly, we solve another fixed-point problem for the adjoint vector $v^T$:

$$v^T = v_0^T + v^T J_f$$

where $v_0^T = \frac{\partial \mathcal{L}}{\partial z^*}$. This is itself a fixed-point equation that can be solved with the same solvers (Broyden, Anderson). The result $v^T$ is then used to compute $\frac{\partial \mathcal{L}}{\partial \theta} = v^T \frac{\partial f_\theta}{\partial \theta}$ via a single Jacobian-vector product.

### Memory Advantage

Standard backpropagation through $L$ layers requires storing all intermediate activations $\{z^{(1)}, \ldots, z^{(L-1)}\}$ — memory $O(L)$. DEQ backward pass requires only the fixed point $z^*$ and the loss gradient — memory **$O(1)$ in depth**, independent of the number of forward iterations.

## The Transformation Function $f_\theta$

The transformation $f_\theta$ can be any neural network block. Common choices:

### Transformer Block

A single Transformer encoder block (multi-head attention + feed-forward) used as the weight-tied transformation creates an infinitely deep Transformer:

$$z^* = \text{TransformerBlock}_\theta(x, z^*)$$

At convergence, $z^*$ is contextually rich — equivalent to an infinite-depth Transformer but stored with the parameters of just one block.

### Convolutional Block

For image tasks, a residual convolutional block:

$$z^* = \text{ResBlock}_\theta(x + z^*)$$

The fixed point captures multi-scale spatial features without increasing parameter count with depth.

### Multiscale DEQ

**Multiscale DEQs (MDEQ)** by Bai et al. (2020) apply the fixed-point framework across multiple spatial resolutions simultaneously, finding a joint fixed point over feature pyramids — achieving state-of-the-art on semantic segmentation and object detection with fewer parameters than explicit multiscale architectures.

## Stability and Convergence Guarantees

Fixed-point iteration $z_{k+1} = f_\theta(x, z_k)$ converges to a unique fixed point if and only if $f_\theta$ is a **contraction mapping** — i.e., the spectral radius of the Jacobian $\rho(J_f) < 1$.

### Spectral Normalization

To encourage contraction, DEQ implementations often apply spectral normalization to the weight matrices of $f_\theta$, ensuring the Lipschitz constant of $f_\theta$ (and thus its Jacobian's spectral radius) stays below 1.

### Jacobian Regularization

An explicit Jacobian regularization term penalizes large spectral radii:

$$\mathcal{L}_{\text{reg}} = \lambda \cdot \text{tr}(J_f^T J_f)$$

estimated via randomized SVD or power iteration. This encourages stability without explicit spectral normalization.

## Connection to Other Architectures

### Weight-Tied Recurrent Networks

A weight-tied RNN applied for infinite steps is a DEQ: $h_t = f_\theta(x_t, h_{t-1})$ run to convergence. DEQs generalize this by allowing arbitrary fixed-point transformations and using principled implicit differentiation rather than truncated BPTT.

### Implicit Layers

DEQs are a special case of **implicit layers** in neural networks — layers defined by equations rather than explicit computations. Other implicit layers include:

- **OptNet** (Amos & Kolter, 2017): layer defined as the solution to a quadratic program
- **Deep Implicit Layers** (general framework)
- **Neural ODEs**: continuous-depth networks where the hidden state evolves via an ODE

### Neural ODEs

DEQs and Neural ODEs (Chen et al., 2018) are related: both define outputs implicitly. Neural ODEs use ODE solvers to integrate $\frac{dz}{dt} = f(z, t)$ from $t=0$ to $t=1$, while DEQs find fixed points of $z = f(z)$. At equilibrium of the ODE (when $\frac{dz}{dt} = 0$), the steady state satisfies the DEQ fixed-point equation.

## Phantom Gradients

A practical challenge: the implicit differentiation backward pass requires solving the adjoint fixed-point problem to the same precision as the forward pass. If the forward solver terminates early (before true convergence), the implicit gradient may be inaccurate.

**Phantom gradients** (Geng et al., 2021) are a correction technique that adds a correction term to account for early termination:

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx v^T \frac{\partial f}{\partial \theta} + \lambda \cdot \text{correction}(z_k, z^*)$$

This improves gradient accuracy without requiring full convergence.

## Performance and Applications

### Sequence Modeling

DEQ Transformers achieve competitive perplexity with standard Transformers on WikiText-103 and enwik8 using $4\times$ fewer parameters — the weight-tying is essentially free regularization.

### Semantic Segmentation

MDEQ matches HRNet on Cityscapes segmentation while using significantly fewer parameters, demonstrating that fixed-point multiscale features can match explicitly trained deep feature pyramids.

### Protein Structure

DEQ-like fixed-point reasoning appears in Evoformer (AlphaFold2), which iterates a pair/residue representation update block (Evoformer trunk) for 48 steps with shared structure — conceptually related to finding equilibrium between sequence and structure representations.

## Practical Considerations

| Aspect | Standard Deep Network | DEQ |
| --- | --- | --- |
| Memory (training) | $O(L \cdot d)$ — stores all activations | $O(d)$ — stores only fixed point |
| Parameters | $O(L \cdot d^2)$ — one block per layer | $O(d^2)$ — one shared block |
| Forward compute | Fixed ($L$ steps) | Variable (until convergence) |
| Backward compute | BPTT through $L$ layers | Solve adjoint fixed-point (implicit) |
| Stability | Gradient vanishing/explosion | Requires Jacobian regularization |
| Convergence | Guaranteed (finite steps) | Not guaranteed without contraction |

## Summary

Deep Equilibrium Models define outputs as fixed points of a weight-tied transformation, solved via quasi-Newton or Anderson mixing methods during the forward pass. The backward pass uses implicit differentiation — computing gradients without backpropagating through solver iterations — yielding $O(1)$ memory in depth regardless of iteration count. DEQs provide a theoretically elegant framework for infinite-depth computation with finite parameters, competitive with explicit deep networks on sequence modeling and vision while using substantially fewer parameters. The core ideas — implicit layers, fixed-point reasoning, and memory-efficient backpropagation — have influenced AlphaFold's Evoformer, Neural ODEs, and the broader landscape of architecture-agnostic deep learning.
