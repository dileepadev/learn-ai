---
title: Kolmogorov-Arnold Networks (KAN)
description: Explore Kolmogorov-Arnold Networks, a new neural network paradigm that replaces fixed activation functions with learnable splines on edges, offering improved expressiveness and interpretability.
---

Kolmogorov-Arnold Networks (KAN) are a class of neural networks proposed in 2024 as a theoretically grounded alternative to the classical Multi-Layer Perceptron (MLP). Rather than placing fixed activation functions on **nodes**, KANs place **learnable activation functions on edges**, inspired directly by the Kolmogorov-Arnold representation theorem.

## The Mathematical Foundation

The **Kolmogorov-Arnold Representation Theorem** (1957) states that any continuous multivariate function $f: [0,1]^n \to \mathbb{R}$ can be written as:

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

where $\phi_{q,p}: [0,1] \to \mathbb{R}$ and $\Phi_q: \mathbb{R} \to \mathbb{R}$ are continuous univariate functions. The takeaway: any multivariate function can be decomposed into a composition of univariate functions.

KANs operationalize this idea into a trainable architecture by parameterizing those univariate functions using **B-splines**.

## KAN vs. MLP: Architectural Difference

| Property | MLP | KAN |
|---|---|---|
| Activation functions | Fixed, on nodes | Learnable, on edges |
| Parameterization | Weight matrices | Spline coefficients |
| Universal approximation | Yes (width/depth) | Yes (by theorem) |
| Interpretability | Low | High (visualizable) |
| Grid refinement | N/A | Supported (resolution increase) |

In an MLP layer: $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$

In a KAN layer: $h_j = \sum_{i} \phi_{i,j}(x_i)$ where each $\phi_{i,j}$ is a learned spline.

## B-Spline Activations

Each edge function $\phi(x)$ in a KAN is parameterized as a combination of a **residual connection** and a **B-spline basis**:

$$\phi(x) = w_b \cdot b(x) + w_s \cdot \text{spline}(x)$$

where:

- $b(x) = x / (1 + e^{-x})$ is a SiLU-like base activation
- $\text{spline}(x) = \sum_i c_i B_i(x)$ is a learnable sum of B-spline basis functions
- $w_b$ and $w_s$ are trainable scalar weights

The **grid** (number of spline knots) can be refined after training to increase function resolution without retraining from scratch — a unique advantage over MLPs.

## Key Properties

### Interpretability

KANs are inherently more interpretable than MLPs because each edge function can be **plotted as a 1D curve**. Practitioners can visualize exactly what transformation is applied to each input feature — enabling symbolic regression-like insights.

### Compositional Structure

KANs naturally decompose functions into nested univariate transformations. For many physics and mathematical tasks, this structure aligns directly with the underlying equations, making KANs excellent at **discovering symbolic formulas** from data.

### Fewer Parameters

For low-dimensional scientific problems, KANs often achieve better accuracy than comparably-sized MLPs. In benchmarks on function approximation tasks (e.g., PDEs, symbolic regression), KANs with 100 parameters have matched MLPs requiring thousands.

## Limitations

- **Training speed:** KAN training is currently slower than MLPs due to spline computation overhead. Batch-parallelized implementations have improved this but the gap remains.
- **High-dimensional data:** For tasks like image classification, KANs have not demonstrated clear advantages over CNNs or Vision Transformers.
- **Optimization landscape:** B-spline optimization is less well-understood than gradient descent on linear weights; convergence can be less reliable.
- **Tooling maturity:** Ecosystem support (hardware kernels, distributed training) lags behind MLP-based frameworks.

## Applications

KANs have shown particular promise in:

- **Scientific machine learning:** Solving PDEs, fitting physics equations
- **Symbolic regression:** Discovering mathematical formulas from data (e.g., recovering Feynman equations)
- **Interpretable AI:** Domain-sensitive applications where understanding the learned function matters
- **Time series:** Temporal KAN variants have been explored for forecasting

## Relationship to Other Architectures

KAN ideas have been extended in several directions:

- **KAN-Transformer Hybrids:** Replacing FFN sublayers in Transformers with KAN layers
- **Graph KANs:** Integrating KAN layers within graph neural network pipelines
- **U-KAN:** KAN-augmented U-Net architectures for image segmentation tasks

## Further Reading

- Liu et al. (2024), *KAN: Kolmogorov-Arnold Networks* — the original paper
- Liu et al. (2024), *KAN 2.0: Kolmogorov-Arnold Networks Meet Science* — extensions and applications
- Kolmogorov (1957), *On the representation of continuous functions of many variables*
