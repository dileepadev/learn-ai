---
title: "Dropout and Its Variants"
description: "Understanding dropout, spatial dropout, dropout for RNNs, and other regularization techniques."
date: "2026-06-06"
tags: ["deep-learning", "regularization", "training"]
---

Dropout randomly zeroes neurons during training to prevent overfitting. This simple technique forces the network to learn redundant representations and acts as an ensemble of many sub-networks.

## Standard Dropout

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = torch.rand_like(x) > self.p
            return x * mask / (1 - self.p)  # Invert scaling
        return x


# PyTorch built-in
dropout = nn.Dropout(p=0.5)  # p is probability of zeroing
```

The inverted scaling $(1-p)$ ensures the expected output magnitude remains the same during training and inference.

## Spatial Dropout for CNNs

Instead of dropping individual pixels, drop entire feature maps:

```python
class SpatialDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        if self.training:
            mask = (torch.rand(x.size(0), x.size(1), 1, 1) > self.p)
            return x * mask / (1 - self.p)
        return x


# PyTorch
spatial_dropout = nn.Dropout2d(p=0.5)  # Dropout2d is spatial dropout
```

## Dropout for RNNs

Standard dropout destroys temporal dependencies. Use variational dropout:

```python
class RNNDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = torch.ones_like(x)
            # Same dropout pattern across all timesteps
            mask = torch.rand(x.shape[1]) > self.p  # Shape: (seq_len,)
            mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            return x * mask / (1 - self.p)
        return x


# PyTorch RNN dropout parameters
rnn = nn.LSTM(128, 256, num_layers=2, dropout=0.5)  # Dropout between layers
```

## Variational Dropout

Use the same dropout mask across all timesteps and forward passes:

```python
class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, x, reset=False):
        if self.training or reset:
            mask = (torch.rand_like(x) > self.p) / (1 - self.p)
            self.mask = mask
        return x * self.mask
    
    def clear_mask(self):
        self.mask = None


# Use same mask throughout sequence for RNNs
def forward_with_variational_dropout(rnn, x, dropout_layer):
    h0, c0 = rnn.hidden0
    hiddens = []
    
    dropout_mask = (torch.rand_like(x[:, 0, :]) > 0.3) / 0.7
    
    for t in range(x.size(0)):
        if dropout_layer.training:
            x_t = x[t] * dropout_mask
        else:
            x_t = x[t]
        h0, c0 = rnn.cell(x_t, (h0, c0))
        hiddens.append(h0)
    
    return torch.stack(hiddens)
```

## Alpha Dropout

Preserves mean and variance for SELU networks:

```python
alpha = 1.6732632423543772848170429916717
lambd = 1.0507009873554804934193349852946

class AlphaDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, x):
        if self.training:
            if self.mask is None:
                self.mask = (torch.rand_like(x) > self.p)
            return x * self.mask + (1 - self.mask) * self._alpha_noise(x)
        return x
    
    def _alpha_noise(self, x):
        return torch.full_like(x, self._alpha)
    
    def extra_repr(self):
        return f'p={self.p}'


# PyTorch built-in
alpha_dropout = nn.AlphaDropout(p=0.5)
```

## DropConnect

Instead of dropping outputs, drop the weights:

```python
class DropConnect(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x, weight, bias=None):
        if self.training:
            mask = (torch.rand_like(weight) > self.p) / (1 - self.p)
            weight = weight * mask
        return F.linear(x, weight, bias)


# Applied to weight matrices
dropconnect_linear = DropConnect()
output = dropconnect_linear(x, layer.weight, layer.bias)
```

## Scheduled Dropout

Anneal dropout probability during training:

```python
class ScheduledDropout:
    def __init__(self, initial_p=0.0, final_p=0.5, epochs=100):
        self.initial_p = initial_p
        self.final_p = final_p
        self.epochs = epochs
        self.epoch = 0
    
    def get_p(self):
        if self.epoch >= self.epochs:
            return self.final_p
        return self.initial_p + (self.final_p - self.initial_p) * self.epoch / self.epochs
    
    def step(self):
        self.epoch += 1
        return self.get_p()


# Usage
dropout_rate = ScheduledDropout(initial_p=0.0, final_p=0.5, epochs=50)
dropout_layer = nn.Dropout(p=dropout_rate.get_p())
```

## Practical Guidelines

| Architecture | Dropout Rate | Placement |
| --- | --- | --- |
| CNNs | 0.2 - 0.5 | After conv layers, before pooling |
| RNNs | 0.2 - 0.4 | After RNN layers, same mask |
| Transformers | 0.0 - 0.1 | After attention and FFN |
| Small networks | 0.5+ | Higher for smaller models |
| Large networks | 0.1 - 0.3 | Lower for larger models |

Dropout is less critical when using batch normalization, which also regularizes.