---
title: Tensor Decomposition Networks
description: Learn how tensor decompositions — CP, Tucker, and Tensor Train — compress neural network weight tensors, enabling efficient inference on resource-constrained devices while preserving model accuracy.
---

Neural network layers are parameterized by high-dimensional weight tensors: convolution kernels are 4D ($C_\text{out} \times C_\text{in} \times H \times W$), fully-connected weights are 2D matrices, and attention projections span multiple heads. Tensor decomposition methods factorize these large tensors into products of smaller components, dramatically reducing parameter counts and computational costs while retaining representational capacity.

## Why Tensor Decompositions?

Convolutional and fully-connected layers dominate both the parameter count and inference latency of modern networks. For edge deployment — mobile devices, microcontrollers, embedded systems — compression is essential.

Standard approaches like pruning and quantization reduce individual weight values, but tensor decompositions impose a **structured low-rank** prior on entire weight tensors, enabling:

- **Faster inference**: decomposed layers use fewer FLOPs per forward pass
- **Smaller footprint**: fewer parameters mean smaller model files
- **Hardware-friendly structure**: regular matrix-vector products map efficiently to BLAS libraries and accelerators
- **Systematic compression**: decomposition rank is a single tunable parameter controlling the accuracy-efficiency tradeoff

## Tensor Fundamentals

A $d$-dimensional tensor $\mathcal{W} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$ generalizes matrices to arbitrary order. The **rank** of a tensor is the minimum number of rank-1 components needed to express it — but unlike matrices, computing tensor rank is NP-hard in general.

Practical decompositions find approximate low-rank representations that minimize reconstruction error while satisfying hardware constraints.

## CP Decomposition (Canonical Polyadic)

The **CP decomposition** (also called PARAFAC) expresses a tensor as a sum of rank-1 components:

$$\mathcal{W} \approx \sum_{r=1}^R \lambda_r \, a_r^{(1)} \otimes a_r^{(2)} \otimes \cdots \otimes a_r^{(d)}$$

where $a_r^{(k)} \in \mathbb{R}^{n_k}$ are factor vectors and $\lambda_r$ are scalar weights. For a 4D convolution kernel $\mathcal{W} \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times H \times W}$:

$$\mathcal{W} \approx \sum_{r=1}^R \lambda_r \, s_r \otimes t_r \otimes h_r \otimes w_r$$

The original convolution with $C_\text{out} \cdot C_\text{in} \cdot H \cdot W$ parameters is replaced by $R \cdot (C_\text{out} + C_\text{in} + H + W)$ parameters — a significant reduction when $R \ll C_\text{out}, C_\text{in}$.

### Four Sequential Convolutions

The CP decomposition of a convolutional layer is implemented as four successive 1D convolutions along each mode — a natural sequence of operations that standard deep learning frameworks support directly.

### Fitting CP Decomposition

CP decomposition is typically computed via **Alternating Least Squares (ALS)**: iteratively fix all factor matrices except one and solve for the remaining factor in closed form. ALS is not guaranteed to converge to the global optimum and can suffer from degeneracy (factors canceling each other at large magnitudes).

## Tucker Decomposition

The **Tucker decomposition** expresses a tensor as a core tensor multiplied by factor matrices along each mode:

$$\mathcal{W} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)} \times_4 U^{(4)}$$

where $\mathcal{G} \in \mathbb{R}^{r_1 \times r_2 \times r_3 \times r_4}$ is the **core tensor** and $U^{(k)} \in \mathbb{R}^{n_k \times r_k}$ are factor matrices. The multilinear rank $(r_1, r_2, r_3, r_4)$ controls compression along each mode independently.

For convolution kernels, Tucker decomposition reduces input and output channel dimensions:

$$\mathcal{W}_{C_\text{out} \times C_\text{in} \times H \times W} \approx \mathcal{G}_{r_\text{out} \times r_\text{in} \times H \times W} \times_1 U_\text{out} \times_2 U_\text{in}$$

This is implemented as three sequential convolutions:

1. $1 \times 1$ convolution reducing $C_\text{in}$ channels to $r_\text{in}$
1. $H \times W$ depthwise-separable convolution (full spatial filtering) on $r_\text{in}$ channels
1. $1 \times 1$ convolution expanding $r_\text{in}$ back to $C_\text{out}$ (via $r_\text{out}$)

Tucker compression is more flexible than CP because different ranks can be assigned per mode — allowing, for example, heavy compression in channel dimensions while preserving full spatial resolution.

### Higher-Order SVD (HOSVD)

The standard algorithm for computing Tucker decompositions is **HOSVD**: compute the SVD of each mode-$k$ unfolding of the tensor and truncate to rank $r_k$. The core tensor is then obtained by projecting onto the truncated factor matrices. HOSVD provides the best approximation among Tucker decompositions with independent mode truncations.

## Tensor Train (TT) Decomposition

The **Tensor Train** (TT) decomposition, also called **Matrix Product State (MPS)** in physics, expresses a tensor as a chain of 3-way core tensors:

$$\mathcal{W}_{i_1 i_2 \cdots i_d} = G^{(1)}_{i_1} G^{(2)}_{i_2} \cdots G^{(d)}_{i_d}$$

where $G^{(k)} \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}$ are the TT cores and $r_0 = r_d = 1$ are boundary conditions. The **TT-rank** $(r_1, \ldots, r_{d-1})$ controls expressivity.

### TT-Matrix (TT-FC Layers)

For compressing large fully-connected weight matrices (e.g., embedding tables, large MLP layers), the **TT-Matrix** format reshapes the weight matrix into a high-order tensor and applies TT decomposition:

$$W \in \mathbb{R}^{m \times n} \to \mathcal{W} \in \mathbb{R}^{m_1 \times \cdots \times m_d \times n_1 \times \cdots \times n_d}$$

then applies TT decomposition with small ranks $r_k$. The original $m \times n$ matrix with $mn$ parameters is replaced by $\sum_k r_{k-1} m_k n_k r_k$ parameters — exponential compression for large embedding matrices.

**Novikov et al. (2015)** demonstrated TT-Matrix compression of the fully-connected layers in AlexNet achieving 200,000× compression with minimal accuracy loss.

### TT-RNN

Recurrent networks with large hidden-to-hidden matrices ($H \times H$) benefit greatly from TT-Matrix compression. The weight matrix is expressed in TT format, and the hidden state is operated on as a TT-vector for efficient forward passes.

## Tensor Ring Decomposition

The **Tensor Ring (TR)** decomposition extends TT by closing the chain into a ring — allowing the first and last cores to interact:

$$\mathcal{W}_{i_1 \cdots i_d} = \text{Tr}\left(G^{(1)}_{i_1} G^{(2)}_{i_2} \cdots G^{(d)}_{i_d}\right)$$

TR has more expressive power than TT at the same rank due to the closed-loop structure, offering better accuracy at equivalent compression ratios for certain architectures.

## Post-Training vs. Training-Aware Decomposition

### Post-Training Decomposition

Apply decomposition to a pre-trained model's weight tensors, then fine-tune briefly to recover accuracy:

1. Compute HOSVD or ALS decomposition of each layer
1. Replace the original layer with the decomposed equivalent
1. Fine-tune for a few epochs with the original training data

This requires access to the original data (for fine-tuning) but not the original training pipeline. Popular tools: `tensorly`, `tltorch`.

### Training-Aware (Factorized Training)

Initialize layers in factorized form from scratch and train end-to-end. The factor matrices are the direct trainable parameters. This avoids the decomposition step and achieves better final accuracy but requires modifying the training pipeline.

**Automatic Rank Determination**: rather than hand-tuning ranks, differentiable rank selection methods (e.g., pruning small singular values, group Lasso on factor columns) automatically discover the optimal rank during training.

## Application to Transformers

Transformer attention and feed-forward layers contain large projection matrices. Tucker and TT compression are applied to:

- **Key/Query/Value projections**: compress $d_\text{model} \times d_\text{head}$ matrices across all heads
- **FFN layers**: the two large $d_\text{model} \times d_\text{ff}$ matrices account for most Transformer parameters
- **Embedding tables**: for NLP models with large vocabularies, TT-Matrix compresses the embedding table

**LoRA** (Low-Rank Adaptation) is a special case of Tucker decomposition restricted to rank-$(r, r)$ with $r \ll d$ — computing $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$.

## Comparison of Decomposition Methods

| Method | Structure | Ranks | Best For | Limitation |
| --- | --- | --- | --- | --- |
| CP | Sum of rank-1 tensors | Single $R$ | Uniform compression | ALS instability |
| Tucker | Core + factor matrices | Per-mode $(r_1,\ldots,r_d)$ | Convolutional kernels | Core tensor may still be large |
| Tensor Train | Chain of 3-way cores | Chain $(r_1,\ldots,r_{d-1})$ | High-order tensors, FC layers | Exponential in $d$ for global ops |
| Tensor Ring | Cyclic TT | Ring ranks | Higher expressivity vs TT | More complex implementation |

## Practical Workflow

```python
import tensorly as tl
from tensorly.decomposition import tucker, parafac

# Tucker decomposition of a conv weight tensor
weight = model.conv1.weight.detach().numpy()  # (C_out, C_in, H, W)
core, factors = tucker(weight, rank=[32, 16, 3, 3])

# Reconstruct to verify quality
reconstructed = tl.tucker_to_tensor((core, factors))
error = tl.norm(weight - reconstructed) / tl.norm(weight)
print(f"Relative reconstruction error: {error:.4f}")
```

After decomposition, replace the original layer with the factorized equivalent layers and fine-tune.

## Summary

Tensor decompositions offer a principled, structured approach to neural network compression:

- **CP** decomposes tensors into rank-1 components — simple but numerically unstable
- **Tucker** uses a core tensor with per-mode factor matrices — flexible and widely used for CNN compression
- **Tensor Train / Ring** factorize high-order tensors into chains of small cores — excellent for large FC layers and embedding tables

Combined with brief fine-tuning, these methods achieve 5–50× compression ratios with 1–3% accuracy degradation on image classification and NLP tasks, making them valuable tools for deploying powerful models on constrained hardware.
