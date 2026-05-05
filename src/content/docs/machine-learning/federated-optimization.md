---
title: Federated Optimization
description: A comprehensive guide to optimization algorithms for federated learning, covering FedAvg, FedProx, SCAFFOLD, FedNova, adaptive methods, and communication-efficient techniques.
---

# Federated Optimization

Federated optimization studies the algorithms used to train machine learning models across **distributed, heterogeneous clients** without centralizing raw data. While federated learning defines the privacy-preserving training paradigm, federated optimization focuses specifically on the convergence theory and practical algorithmic choices that determine how efficiently and robustly a global model can be learned from local updates.

## The Federated Optimization Problem

The canonical objective is:

$$\min_{w \in \mathbb{R}^d} f(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

where $F_k(w) = \frac{1}{n_k}\sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)$ is the local loss on client $k$, $n_k$ is the number of samples on client $k$, and $n = \sum_k n_k$ is the total sample count.

**Key challenges:**

- **Statistical heterogeneity** ($\mathcal{D}_k \not\sim \mathcal{D}_{k'}$): data distributions differ across clients (non-IID)
- **Systems heterogeneity**: clients have different compute, memory, and communication bandwidth
- **Partial participation**: only a fraction $C$ of clients participate per round
- **Communication bottleneck**: each round requires uploading/downloading model parameters

## FedAvg — Federated Averaging

McMahan et al. (2017) introduced **FedAvg**, the foundation of federated optimization. Each round:

1. Server broadcasts global model $w_t$
2. Each selected client $k$ runs $E$ epochs of local SGD:
   $$w_{k,t}^{(e+1)} = w_{k,t}^{(e)} - \eta \nabla F_k(w_{k,t}^{(e)})$$
3. Server aggregates:
   $$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{k,t}^{(E)}$$

```python
import torch
import copy
from typing import List


def fedavg_round(
    global_model: torch.nn.Module,
    client_datasets: List,
    client_fraction: float = 0.1,
    local_epochs: int = 5,
    lr: float = 0.01,
) -> torch.nn.Module:
    num_clients = len(client_datasets)
    selected = torch.randperm(num_clients)[:max(1, int(client_fraction * num_clients))]

    updates, weights = [], []

    for k in selected:
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(client_datasets[k], batch_size=32)

        local_model.train()
        for _ in range(local_epochs):
            for x, y in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(local_model(x), y)
                loss.backward()
                optimizer.step()

        updates.append({n: p.data.clone() for n, p in local_model.named_parameters()})
        weights.append(len(client_datasets[k]))

    # Weighted average
    total = sum(weights)
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data = sum(
                w / total * u[name] for u, w in zip(updates, weights)
            )

    return global_model
```

**Convergence issue**: With non-IID data, local SGD steps pull each client model toward its local optimum, causing **client drift** — the aggregated model diverges from the true global optimum.

## FedProx — Proximal Regularization

Li et al. (2018) added a proximal term to each client's objective to limit drift:

$$\min_{w} F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

This penalizes local updates that deviate far from the global model, guaranteeing convergence even with heterogeneous data and partial participation.

```python
def fedprox_local_step(model, global_params, data_loader, lr=0.01, mu=0.01, epochs=5):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            # Proximal penalty
            prox = sum(
                ((p - g) ** 2).sum()
                for p, g in zip(model.parameters(), global_params)
            )
            (loss + mu / 2 * prox).backward()
            optimizer.step()
```

**Trade-off**: Higher $\mu$ reduces drift but limits local adaptation — clients with very different data distributions can benefit from larger local updates.

## SCAFFOLD — Variance Reduction with Control Variates

Karimireddy et al. (2020) identified client drift as the key convergence barrier and addressed it with **control variates**:

Each client maintains a control variate $c_k$ estimating its gradient correction. The local update becomes:

$$w_{k}^{(e+1)} = w_{k}^{(e)} - \eta\left(\nabla F_k(w_k^{(e)}) - c_k + c\right)$$

where $c$ is the server control variate. After the round:

$$c_k^+ = c_k - c + \frac{1}{KE\eta}(w_t - w_k^{(E)})$$

SCAFFOLD achieves **linear speedup** with the number of clients and converges without assumptions on gradient similarity.

## FedNova — Normalized Averaging

Wang et al. (2020) found that FedAvg conflates the number of local steps and learning rate, causing objective inconsistency. FedNova normalizes client updates:

$$\Delta_k = \frac{\tau_k}{\sum_{j}\tau_j}(w_t - w_{k}^{(E)})$$

where $\tau_k$ is the effective number of local steps for client $k$. This removes the objective bias that appears when clients take different numbers of steps.

## FedAdam and Adaptive Server Optimizers

**FedAdam** (Reddi et al., 2020) applies adaptive optimizers on the server side:

$$\Delta = \sum_k \frac{n_k}{n}(w_t - w_{k,t}^{(E)}) \quad \text{(pseudo-gradient)}$$

$$m_{t+1} = \beta_1 m_t + (1-\beta_1)\Delta_t$$

$$v_{t+1} = \beta_2 v_t + (1-\beta_2)\Delta_t^2$$

$$w_{t+1} = w_t + \frac{\eta_s}{\sqrt{v_{t+1}} + \tau} m_{t+1}$$

```python
class FedAdamServer:
    def __init__(self, global_model, lr=0.01, beta1=0.9, beta2=0.99, tau=1e-3):
        self.model = global_model
        self.lr = lr
        self.beta1, self.beta2, self.tau = beta1, beta2, tau
        self.m = {n: torch.zeros_like(p) for n, p in global_model.named_parameters()}
        self.v = {n: torch.zeros_like(p) for n, p in global_model.named_parameters()}
        self.t = 0

    def step(self, pseudo_grads: dict):
        self.t += 1
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                g = pseudo_grads[name]
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g ** 2
                param.data += self.lr * self.m[name] / (self.v[name].sqrt() + self.tau)
```

## Communication Efficiency

### Gradient Compression

**Top-k sparsification**: send only the $k$ largest-magnitude gradient components.

```python
def topk_compress(tensor: torch.Tensor, k: int):
    flat = tensor.flatten()
    values, indices = flat.abs().topk(k)
    mask = torch.zeros_like(flat)
    mask[indices] = flat[indices]
    return mask.reshape(tensor.shape)
```

**Random k**: unbiased but higher variance than top-k.

### Quantization

**1-bit SGD**: quantize each gradient to $\pm 1$ with error feedback to maintain convergence.

```python
def one_bit_quantize(tensor: torch.Tensor):
    sign = tensor.sign()
    scale = tensor.abs().mean()
    return sign, scale   # dequantize as sign * scale
```

### Local SGD with Periodic Communication

Rather than communicating every step, clients synchronize every $H$ local steps. Communication cost reduces by $H\times$ at the cost of more client drift.

## Personalized Federated Learning

Standard federated optimization returns one global model, which may perform poorly on clients with very different data. Personalization approaches include:

| Method | Idea | Formulation |
|---|---|---|
| Per-FedAvg | MAML-style meta-learning | Optimize for fast local adaptation |
| pFedMe | Moreau envelope regularization | Separate local and global models |
| FedBN | Local batch normalization | Only synchronize non-BN parameters |
| APFL | Mixture of local + global | $\alpha w_k + (1-\alpha) w_{\text{global}}$ |
| Ditto | Dual objective | Alternating local/global updates |

## Convergence Summary

| Algorithm | Non-IID Convergence | Communication | Extra Memory |
|---|---|---|---|
| FedAvg | Suboptimal (biased) | $O(R)$ rounds | None |
| FedProx | Convergent (biased) | $O(R)$ rounds | None |
| SCAFFOLD | Unbiased, linear speedup | $2\times$ upload | Control variates |
| FedNova | Unbiased | $O(R)$ rounds | Step counts |
| FedAdam | Faster convergence | $O(R)$ rounds | Moment buffers |

## Practical Recommendations

- Start with **FedAvg** for IID or mildly heterogeneous data
- Use **FedProx** ($\mu \approx 0.01$) when client data distributions diverge significantly
- Use **SCAFFOLD** for strongly heterogeneous settings requiring theoretical guarantees
- Apply **FedAdam** at the server when local step counts vary across clients
- Enable **gradient compression** (top-10% sparsification) when communication is the bottleneck
- Monitor **client drift** via gradient cosine similarity between client updates

## Summary

Federated optimization bridges the gap between the privacy guarantees of federated learning and the efficiency demands of large-scale model training. FedAvg established the paradigm; FedProx, SCAFFOLD, FedNova, and FedAdam address its convergence shortcomings under non-IID data, variable participation, and communication constraints. As federated learning moves into production for healthcare, mobile, and finance applications, robust optimization algorithms are central to achieving models that are simultaneously accurate, fair, and communication-efficient.
