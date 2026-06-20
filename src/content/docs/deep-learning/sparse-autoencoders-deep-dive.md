---
title: Sparse Autoencoders (SAEs) for Interpretability
description: Explore Sparse Autoencoders (SAEs) and how they solve superposition in neural networks, allowing mechanistic interpretability of high-dimensional activations.
---

Neural networks represent concepts as directions in activation space. However, because the number of concepts a model must learn exceeds its dimensionality (e.g., a 4096-dimensional residual stream representing millions of real-world concepts), networks squeeze multiple concepts into the same dimensions—a phenomenon known as **superposition**. This makes raw activations highly polysemantic and difficult for humans to interpret.

**Sparse Autoencoders (SAEs)** address this issue. By training a shallow autoencoder with an L1 regularization penalty on neural activations, SAEs decompose polysemantic activations into a sparse set of interpretable, monosemantic features.

---

## The Superposition Problem

Consider an activation vector $h \in \mathbb{R}^d$. If a model represents a concept $i$ as a unit direction vector $v_i$, we can write the activation as:

$$h = \sum_{i} f_i v_i$$

Where $f_i \ge 0$ is the activation value of concept $i$.
- If the concepts are orthogonal ($v_i^T v_j = 0$), we can represent at most $d$ concepts.
- To represent $M \gg d$ concepts, the model uses non-orthogonal directions. This introduces interference (cross-talk) between concepts, which the model suppresses by ensuring that only a small subset of concepts are active at any time (**sparsity**).

Because of superposition, looking at individual activations or neurons directly is confusing; a single neuron might fire for "clinical medical trials," "conversations about soccer," and "JavaScript syntax."

---

## Decomposing Activations with SAEs

An SAE is a neural network with a single hidden layer that is trained to reconstruct activation vectors $h$.

```
Activation Vector (h) ---> Encoder (W_enc) ---> ReLU/Sparsity ---> Latent Features (f)
                                                                      |
Activation Reconstruction (h_hat) <--- Decoder (W_dec) <-------------+
```

### 1. The Encoder
The encoder projects the activation vector $h$ into a higher-dimensional space ($D \gg d$, typically $8\text{x}$ to $32\text{x}$ larger than the model's residual stream) and applies a ReLU activation to enforce non-negativity:

$$f = \text{ReLU}\left( W_{\text{enc}} (h - b_{\text{dec}}) + b_{\text{enc}} \right)$$

Where $f \in \mathbb{R}^D$ is the sparse feature activation vector, $W_{\text{enc}} \in \mathbb{R}^{D \times d}$ is the encoder weight matrix, and $b_{\text{enc}}$ is the bias.

### 2. The Decoder
The decoder attempts to reconstruct the original activation vector $h$ from the sparse latent features $f$:

$$\hat{h} = W_{\text{dec}} f + b_{\text{dec}}$$

Where $W_{\text{dec}} \in \mathbb{R}^{d \times D}$ is the decoder weight matrix.

---

## The Loss Function: Enforcing Sparsity

To ensure that the hidden layer $f$ represents concepts monosemantically, we must force it to be **sparse** (most elements of $f$ should be exactly $0$). This is achieved by combining reconstruction loss with an L1 regularization penalty:

$$\mathcal{L}_{\text{SAE}}(h) = \|h - \hat{h}\|^2_2 + \lambda \|f\|_1$$

Where:
- $\|h - \hat{h}\|^2_2$ is the Mean Squared Error (MSE) reconstruction loss.
- $\|f\|_1 = \sum_{i} |f_i|$ is the L1 norm of the latent features, which penalizes the sum of active feature values.
- $\lambda$ is a hyperparameter balancing reconstruction fidelity against sparsity.

---

## Training Enhancements

Standard SAEs suffer from **dead latents**—features that stop firing during training because their weights are updated in a way that prevents the ReLU from activating. To address this, researchers use two main techniques:
- **Latent Resampling:** Periodically identifying dead latents and resetting their weights to match activation vectors that the SAE currently reconstructs poorly.
- **Top-K SAEs:** Instead of using L1 regularization, Top-K SAEs explicitly keep only the $K$ largest activations in $f$ and set the remaining $D-K$ features to $0$. This eliminates the need to tune $\lambda$.

---

## Code Concept: A Basic SAE Module

Below is a PyTorch implementation of a standard Sparse Autoencoder with L1 regularization.

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, activation_dim, dict_size, l1_coeff=1e-3):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size # D: typically activation_dim * 8 or 32
        self.l1_coeff = l1_coeff
        
        # Encoder: projects activation to high-dimensional space
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.relu = nn.ReLU()
        
        # Decoder: reconstructs activation
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.dec_bias = nn.Parameter(torch.zeros(activation_dim))
        
        # Normalize decoder columns to unit norm to prevent L1 scaling tricks
        self.normalize_decoder_weights()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        # Enforce unit norm on columns of decoder weight matrix
        norms = torch.norm(self.decoder.weight, p=2, dim=0, keepdim=True)
        self.decoder.weight.div_(norms)

    def forward(self, h):
        # h: [batch, activation_dim]
        # Step 1: Center activations and encode
        h_centered = h - self.dec_bias
        features = self.relu(self.encoder(h_centered)) # [batch, dict_size]
        
        # Step 2: Reconstruct
        h_reconstructed = self.decoder(features) + self.dec_bias # [batch, activation_dim]
        
        # Step 3: Compute Loss
        reconstruction_loss = nn.functional.mse_loss(h_reconstructed, h)
        l1_loss = torch.norm(features, p=1, dim=-1).mean()
        total_loss = reconstruction_loss + self.l1_coeff * l1_loss
        
        return h_reconstructed, total_loss, features
```
