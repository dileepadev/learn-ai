---
title: Neural Ordinary Differential Equations
description: Explore Neural ODEs — a family of deep learning models that parameterize the derivative of a hidden state with a neural network, enabling continuous-depth models, latent dynamics learning, time-series interpolation, and memory-efficient training via the adjoint method.
---

**Neural Ordinary Differential Equations (Neural ODEs)** reframe deep learning as a continuous dynamical system. Instead of stacking discrete layers that transform a hidden state $\mathbf{h}_t \to \mathbf{h}_{t+1}$, a Neural ODE defines the **derivative** of the hidden state as a neural network and solves an initial value problem:

$$\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t), \quad \mathbf{h}(t_0) = \mathbf{h}_0$$

The output $\mathbf{h}(T)$ at time $T$ is obtained by integrating this ODE from $t_0$ to $T$ using any off-the-shelf numerical solver (Euler, Runge-Kutta, adaptive step-size solvers). Introduced by Chen et al. (NeurIPS 2018), Neural ODEs unify residual networks and recurrent models under a continuous framework and unlock capabilities that discrete architectures cannot easily provide.

## Connection to ResNets

A standard residual network computes:

$$\mathbf{h}_{t+1} = \mathbf{h}_t + f_\theta(\mathbf{h}_t, t)$$

This is the **Euler discretization** of the ODE $d\mathbf{h}/dt = f_\theta(\mathbf{h}(t), t)$ with step size $\Delta t = 1$. A Neural ODE is a ResNet with:

- **Infinite layers**: The continuous limit of stacking infinitely many thin residual blocks.
- **Adaptive depth**: The ODE solver chooses step sizes adaptively — allocating more computation to "difficult" parts of the transformation.
- **Continuous parameterization**: The same network $f_\theta$ is evaluated at many time points, not separate weights per layer.

This framing shows Neural ODEs as a natural generalization of residual networks, with the additional property that the forward pass is now an ODE solve rather than a fixed computation graph.

## The Adjoint Method for Training

Backpropagating through an ODE solver is expensive — storing all intermediate states for reverse-mode automatic differentiation requires $O(L)$ memory, where $L$ is the number of solver steps.

The **adjoint method** computes gradients by solving a second ODE backwards in time, requiring only $O(1)$ memory regardless of integration depth:

Define the **adjoint state**:

$$\mathbf{a}(t) = -\frac{d\mathcal{L}}{d\mathbf{h}(t)}$$

It evolves according to:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\top \frac{\partial f_\theta(\mathbf{h}(t), t)}{\partial \mathbf{h}(t)}$$

The parameter gradients are:

$$\frac{d\mathcal{L}}{d\theta} = -\int_{t_1}^{t_0} \mathbf{a}(t)^\top \frac{\partial f_\theta(\mathbf{h}(t), t)}{\partial \theta} \, dt$$

Both integrals are solved in a single backward ODE call, augmenting the system with the adjoint state and parameter gradients. This gives Neural ODEs **constant memory cost** for training — a significant advantage for very deep effective models.

```python
import torch
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(torch.nn.Module):
    """The neural network that parameterizes dh/dt."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, hidden_dim),
        )
    
    def forward(self, t, h):
        # t is scalar time, h is hidden state [batch, hidden_dim]
        return self.net(h)

class NeuralODE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, t_span=None):
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0])
        
        h0 = self.encoder(x)
        
        # Integrate ODE from t=0 to t=1
        # odeint_adjoint uses the adjoint method for memory-efficient gradients
        h_trajectory = odeint(
            self.odefunc,
            h0,
            t_span,
            method='dopri5',        # Dormand-Prince adaptive Runge-Kutta
            rtol=1e-3,
            atol=1e-4,
        )
        # h_trajectory: [len(t_span), batch, hidden_dim]
        h_final = h_trajectory[-1]  # State at t=1
        return self.decoder(h_final)

# Training
model = NeuralODE(input_dim=2, hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    pred = model(batch_x)
    loss = torch.nn.functional.mse_loss(pred, batch_y)
    loss.backward()  # Adjoint method handles gradients automatically
    optimizer.step()
```

## Latent Neural ODEs for Time Series

One of the most powerful applications of Neural ODEs is modeling **irregularly sampled time series** — data where observations occur at arbitrary, non-uniform time points (medical records, sensor networks, event logs).

Standard RNNs require equal-spaced inputs and must impute missing values. **Latent Neural ODEs** instead:

1. **Encode** all observations into an initial latent state $\mathbf{z}(t_0)$ using an RNN that processes observations backwards in time.
2. **Evolve** the latent state continuously forward in time via an ODE: $d\mathbf{z}/dt = f_\theta(\mathbf{z}(t), t)$.
3. **Decode** the latent state at any query time into observations.

```python
class LatentODE(torch.nn.Module):
    """
    Latent Neural ODE for irregularly sampled time series.
    
    Encodes observations into a latent initial state,
    evolves with an ODE, decodes at arbitrary query times.
    """
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        # Encoder: RNN processes reversed-time observations
        self.encoder_rnn = torch.nn.GRU(obs_dim + 1, hidden_dim, batch_first=True)
        self.encoder_fc = torch.nn.Linear(hidden_dim, 2 * latent_dim)  # mean + log_var
        
        # ODE dynamics
        self.odefunc = ODEFunc(latent_dim)
        
        # Decoder
        self.decoder = torch.nn.Linear(latent_dim, obs_dim)
    
    def encode(self, obs_times, obs_values):
        """Encode observations into initial latent state."""
        # Concatenate time as extra feature, reverse sequence
        time_features = obs_times.unsqueeze(-1)
        inputs = torch.cat([obs_values, time_features], dim=-1)
        inputs_reversed = torch.flip(inputs, dims=[1])
        
        _, hidden = self.encoder_rnn(inputs_reversed)
        params = self.encoder_fc(hidden.squeeze(0))
        mean, log_var = params.chunk(2, dim=-1)
        return mean, log_var
    
    def decode_at(self, z0, query_times):
        """Evolve latent state and decode at query times."""
        t_span = torch.cat([torch.tensor([0.0]), query_times])
        z_trajectory = odeint(self.odefunc, z0, t_span)
        return self.decoder(z_trajectory[1:])  # Predictions at query times
    
    def forward(self, obs_times, obs_values, query_times):
        mean, log_var = self.encode(obs_times, obs_values)
        # Reparameterization trick
        z0 = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        predictions = self.decode_at(z0, query_times)
        kl_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(-1).mean()
        return predictions, kl_loss
```

Latent ODEs can predict future observations, impute missing values at arbitrary times, and extrapolate beyond the observation window — all without any modification to the model for different sampling patterns.

## Continuous Normalizing Flows

**Continuous Normalizing Flows (CNFs)** use the Neural ODE framework to define generative models with tractable log-likelihoods. The **instantaneous change of variables formula** gives:

$$\frac{d \log p(\mathbf{z}(t))}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial \mathbf{z}(t)}\right)$$

Starting from a simple distribution $p(\mathbf{z}(t_0))$ (e.g., Gaussian) and integrating forward, the model learns a complex target distribution $p(\mathbf{z}(t_1))$ via the ODE dynamics. The log-likelihood can be computed exactly — unlike discrete normalizing flows that require invertible architectures with Jacobian determinants that are expensive to compute.

**FFJORD** (Free-Form Jacobian of Reversible Dynamics) uses Hutchinson's trace estimator to make CNF training scalable, avoiding the $O(d^2)$ cost of explicit trace computation.

## Neural CDEs: Handling Controlled Inputs

**Neural Controlled Differential Equations (Neural CDEs)** extend Neural ODEs for sequences with input control signals:

$$d\mathbf{h}(t) = f_\theta(\mathbf{h}(t)) \, dX(t)$$

where $X(t)$ is a continuous path constructed by interpolating the input sequence (using natural cubic splines). This formulation is theoretically equivalent to an RNN in the limit but handles irregular sampling naturally and benefits from ODE solver adaptive step-size control.

## Stiffness and Solver Selection

Neural ODE training performance depends critically on solver choice:

| Solver | Method | Use Case |
|--------|--------|----------|
| `euler` | Fixed-step Euler | Quick debugging only |
| `rk4` | Fixed-step RK4 | Smooth dynamics, known step size |
| `dopri5` | Adaptive RK4/5 | Default for most tasks |
| `adams` | Adams-Moulton | Non-stiff, high accuracy |
| `bosh3` | Adaptive RK3 | Fast, lower accuracy |

For **stiff ODEs** (dynamics varying across very different timescales), implicit solvers are needed — `torchdiffeq` provides `implicit_adams` for such cases. Neural ODEs trained on stiff problems will produce large NFE (number of function evaluations) counts, significantly slowing training.

Regularizing the dynamics to reduce stiffness — via a penalty on the Frobenius norm of the Jacobian $\partial f_\theta / \partial \mathbf{h}$ — improves training efficiency.

## Applications

**Image classification**: The OdeNet architecture replaces the second half of ResNet with an ODE block — achieving comparable accuracy to ResNet with fewer parameters and adaptive computation.

**Density estimation and generative modeling**: CNFs (described above) model complex continuous distributions for scientific applications including molecular conformations, cosmological density fields, and normalizing variational posteriors.

**Physics simulation**: Neural ODEs are a natural fit for learning dynamics from physical systems — the inductive bias of ODE structure aligns with physical systems governed by differential equations.

**Drug pharmacokinetics**: Latent ODEs model drug concentration in blood over time from sparse irregular clinical measurements — capturing absorption, distribution, metabolism, and excretion dynamics.

**Financial time series**: Continuous-time models of asset prices and volatility benefit from the irregular observation handling of latent ODEs.

Neural ODEs represent a conceptual bridge between classical numerical methods for differential equations and modern deep learning — providing continuous, memory-efficient, and physically interpretable models for a wide range of tasks involving dynamics and change over time.
