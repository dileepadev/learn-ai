---
title: Equivariant Neural Networks
description: A comprehensive guide to equivariant neural networks — architectures that respect physical symmetries such as rotation, translation, and reflection — with applications in molecular modeling, protein structure, and physics simulation.
---

# Equivariant Neural Networks

Standard neural networks treat inputs as flat arrays of numbers, discarding geometric structure. **Equivariant neural networks** explicitly encode physical symmetries — rotation, translation, reflection, permutation — into their architecture, so that transforming the input produces a predictably transformed output. This is crucial for applications in molecular modeling, protein structure prediction, materials science, and physics simulation, where the laws of nature respect these symmetries.

## Symmetry and Equivariance

A function $f$ is **equivariant** with respect to a group $G$ if:

$$f(g \cdot x) = \rho(g) \cdot f(x) \quad \forall g \in G$$

where $\rho$ is a (possibly different) representation of $g$ acting on the output space.

Special cases:

- **Invariance** (a special equivariance): $f(g \cdot x) = f(x)$ — output unchanged by symmetry (e.g., total energy of a molecule)
- **Equivariance to rotation**: rotate a molecule → force vectors rotate by the same amount

Enforcing these properties by construction eliminates the need to learn them from data — reducing sample complexity and improving generalization across orientations.

## Key Symmetry Groups in Physical ML

| Group | Transformations | Applications |
|---|---|---|
| SE(3) | 3D rotations + translations | Molecular forces, protein folding |
| E(3) | SE(3) + reflections | Crystal properties, atomic potentials |
| SO(2) | 2D rotations | Aerial images, point clouds |
| S_n | Permutations | Sets of atoms, point clouds |
| O(3) | 3D rotations + reflections | Scalar/vector/tensor molecular properties |

## Spherical Harmonics and Irreducible Representations

The building blocks of E(3)-equivariant networks are **spherical harmonic features** — functions $Y_l^m(\hat{r})$ that transform predictably under rotation according to Wigner D-matrices:

$$R \cdot Y_l^m(\hat{r}) = \sum_{m'} D_l^{m'm}(R) \, Y_l^{m'}(\hat{r})$$

Features of **degree $l$** transform as rank-$l$ spherical tensors:

- $l=0$: scalars (invariant)
- $l=1$: 3D vectors
- $l=2$: symmetric traceless tensors
- $l \geq 3$: higher-order tensors

## e3nn: E(3)-Equivariant Neural Networks Library

```python
import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet

# Define irreducible representations (irreps)
irreps_in = o3.Irreps("8x0e + 8x1o + 4x2e")   # scalars + vectors + tensors
irreps_out = o3.Irreps("16x0e + 8x1o")

# Equivariant linear layer (preserves symmetry)
linear = o3.Linear(irreps_in, irreps_out)

# Tensor product — the core equivariant operation
irreps_a = o3.Irreps("8x1o")   # 8 vectors
irreps_b = o3.Irreps("4x2e")   # 4 rank-2 tensors
tp = o3.FullyConnectedTensorProduct(
    irreps_a, irreps_b,
    o3.Irreps("16x0e + 8x1o + 4x2e"),   # output irreps
    irrep_normalization="component",
)

# Gate nonlinearity (scalars gate vectors/tensors)
from e3nn.nn import Gate
gate = Gate(
    irreps_scalars=o3.Irreps("16x0e"),
    act_scalars=[torch.nn.functional.silu],
    irreps_gates=o3.Irreps("8x0e"),
    act_gates=[torch.sigmoid],
    irreps_gated=o3.Irreps("8x1o"),
)
```

## SEGNN: Steerable E(3) Graph Neural Network

```python
from e3nn import o3
import torch.nn as nn

class SEGNNLayer(nn.Module):
    def __init__(self, irreps_node: str, irreps_edge: str):
        super().__init__()
        irreps_node = o3.Irreps(irreps_node)
        irreps_edge = o3.Irreps(irreps_edge)

        # Message = tensor product of neighbor features with edge features
        self.message_tp = o3.FullyConnectedTensorProduct(
            irreps_node, irreps_edge, irreps_node, shared_weights=False
        )
        # Radial network provides weights for tensor product
        self.radial_net = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, self.message_tp.weight_numel),
        )

    def forward(self, node_features, edge_features, edge_index, edge_distances):
        src, dst = edge_index
        radial_weights = self.radial_net(edge_distances.unsqueeze(-1))
        messages = self.message_tp(node_features[src], edge_features, radial_weights)
        # Aggregate messages
        agg = torch.zeros_like(node_features)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        return node_features + agg   # residual update
```

## MACE: Many-body Equivariant Networks

MACE (Batatia et al., 2022) achieves state-of-the-art accuracy on molecular dynamics benchmarks by computing equivariant many-body messages:

```python
class MACEInteraction(nn.Module):
    """One MACE interaction block: 2-body + 3-body equivariant features."""
    def __init__(self, irreps_in, irreps_out, irreps_sh, max_ell=3, num_elements=118):
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_ell)
        # First-order equivariant message
        self.linear_1 = o3.Linear(irreps_in, irreps_in)
        # Tensor product with spherical harmonics of edge direction
        self.tp_1 = o3.FullyConnectedTensorProduct(irreps_in, self.irreps_sh, irreps_out)
        # Second tensor product for many-body terms
        self.tp_2 = o3.FullyConnectedTensorProduct(irreps_out, irreps_out, irreps_out)

    def forward(self, node_feat, edge_index, edge_vec):
        src, dst = edge_index
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True)
        msg_1 = self.tp_1(self.linear_1(node_feat[src]), sh)
        # Aggregate
        agg_1 = torch.zeros(node_feat.shape[0], msg_1.shape[-1], device=node_feat.device)
        agg_1.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg_1), msg_1)
        # Many-body: tensor product of aggregated with itself
        agg_2 = self.tp_2(agg_1, agg_1)
        return agg_1 + agg_2
```

MACE models trained on datasets like rMD17 achieve sub-meV/atom energy errors and sub-meV/Å force errors.

## Applications

### Molecular Force Fields

Equivariant networks predict atomic forces $\vec{F}_i = -\nabla_{r_i} E$ by differentiating an invariant energy prediction:

```python
class EquivariantForceField(nn.Module):
    def __init__(self, energy_model):
        super().__init__()
        self.energy_model = energy_model   # outputs scalar (invariant) energy

    def forward(self, positions, atomic_numbers, edge_index):
        positions.requires_grad_(True)
        energy = self.energy_model(positions, atomic_numbers, edge_index)
        forces = -torch.autograd.grad(
            energy.sum(), positions, create_graph=True
        )[0]
        return energy, forces   # forces are automatically E(3)-equivariant
```

### Protein Structure Prediction

AlphaFold2 uses invariant point attention (IPA) — an attention mechanism operating in the local frame of each residue that is SE(3)-equivariant by construction. ESM-IF (inverse folding) uses GVP-GNN (Geometric Vector Perceptrons) that maintain separate scalar and vector feature channels.

### Crystal Property Prediction

Equivariant message-passing on periodic crystal graphs (where unit cell symmetries must be respected) enables prediction of bandgaps, formation energies, and phonon spectra from crystal structures.

## Equivariant vs Standard GNN

| Property | Standard GNN | Equivariant GNN |
|---|---|---|
| Rotation invariant output | Only after data augmentation | By construction |
| Rotation equivariant vectors | ❌ | ✅ |
| Sample efficiency | Lower | Higher |
| Inference cost | Fast | Moderate (tensor products) |
| Typical use case | Graph classification | Molecular dynamics, forces |
| Libraries | PyG, DGL | e3nn, MACE, NequIP, SE(3)-DNN |

## Summary

Equivariant neural networks embed the geometric symmetries of physical laws directly into the model architecture — yielding networks that generalize across rotations and translations by construction rather than by data augmentation. The key primitives are spherical harmonic features, irreducible representations, and equivariant tensor products, implemented efficiently in libraries like e3nn. State-of-the-art systems (MACE, NequIP, SEGNN) achieve chemical-accuracy force field predictions that enable molecular dynamics simulations previously only possible with expensive quantum chemistry calculations. As these models scale and speed up, equivariant networks are becoming a standard component of computational chemistry, drug discovery, and materials science workflows.
