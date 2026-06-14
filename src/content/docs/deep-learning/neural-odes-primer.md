---
title: Neural ODEs Primer
description: An introduction to Neural Ordinary Differential Equations — treating neural network depth as continuous, using ODE solvers for forward and backward passes, and their applications.
---

Neural Ordinary Differential Equations (Neural ODEs), introduced by Chen et al. (NeurIPS 2018), reframe the residual network as a continuous dynamical system. Instead of a finite sequence of discrete layers, the hidden state evolves according to a differential equation parameterized by a neural network.

## From ResNets to Neural ODEs

A residual network layer computes:

$$\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta_t)$$

This looks like the Euler method for solving an ODE. Neural ODEs take this analogy to its limit: instead of discrete steps indexed by layer, define a **continuous-time hidden state** $\mathbf{h}(t)$ governed by:

$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$$

The output at time $T$ is obtained by integrating from initial condition $\mathbf{h}(t_0)$ (the input) to $\mathbf{h}(T)$ (the output):

$$\mathbf{h}(T) = \mathbf{h}(t_0) + \int_{t_0}^{T} f(\mathbf{h}(t), t, \theta) \, dt$$

The integral is computed by a black-box ODE solver. The neural network $f$ defines the derivative (the dynamics), not the values directly.

## The Forward Pass

Computing $\mathbf{h}(T)$ requires solving the ODE numerically. Any standard ODE solver works:
- **Euler method:** Simple but inaccurate; requires small steps.
- **Runge-Kutta (RK4):** More accurate; standard choice.
- **Adaptive solvers (Dormand-Prince, Adams):** Adaptively choose step sizes based on error tolerance — use fewer evaluations where dynamics are smooth, more where they're complex.

The solver treats the network $f$ as a black box, calling it as many times as needed to achieve the desired accuracy.

## The Adjoint Method for Backpropagation

Standard backpropagation through ODE solver steps would require storing all intermediate states — memory proportional to the number of function evaluations, which can be large for adaptive solvers.

Neural ODEs use the **adjoint sensitivity method** instead. The adjoint $\mathbf{a}(t) = \partial L / \partial \mathbf{h}(t)$ evolves backward in time according to its own ODE:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t, \theta)}{\partial \mathbf{h}}$$

Gradients with respect to parameters are computed alongside this backward ODE integration. This gives **O(1) memory** with respect to depth/integration time, independent of the number of function evaluations during the forward pass.

## Key Properties

### Continuous Depth
The "depth" of a Neural ODE is a continuous quantity — the integration time interval $[t_0, T]$. You can trade accuracy (more solver steps, more compute) for speed at inference time without changing the model.

### Adaptive Computation
Adaptive ODE solvers spend more function evaluations on difficult inputs and fewer on easy ones. This implicit adaptive computation is built-in, unlike the uniform depth of standard networks.

### Memory Efficiency
The adjoint method uses constant memory regardless of solver steps, enabling extremely deep (long integration time) models without the memory cost that would make deep ResNets infeasible.

### Continuous-Time Models
Neural ODEs are a natural fit for problems defined in continuous time: irregularly sampled time series, physical simulations, and systems where the dynamics are more naturally expressed as rates of change.

## Latent Neural ODEs for Time Series

One of the most impactful applications is modeling **irregularly sampled time series** — clinical patient records, financial tick data, sensor readings with missing values.

The approach:
1. **Encode** sparse observations into a latent initial condition $\mathbf{z}(t_0)$ using an RNN encoder.
2. **Integrate** the ODE from $t_0$ to any desired future time points — the ODE naturally handles arbitrary time gaps.
3. **Decode** the latent trajectory to observations at the queried time points.

This unifies sequence modeling with continuous dynamics and handles missing data gracefully.

## Normalizing Flows and Continuous Normalizing Flows

Neural ODEs enable **Continuous Normalizing Flows (CNFs)**, a generative modeling approach. A CNF defines a probability density transformation via an ODE. The change in log-density is computed using the instantaneous change-of-variables formula:

$$\frac{d \log p(\mathbf{z}(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right)$$

CNFs allow flexible density estimation with exact likelihood computation, avoiding the architectural constraints of discrete normalizing flows.

## Limitations

- **Slow training:** Each forward and backward pass calls an ODE solver potentially dozens to hundreds of times. Training Neural ODEs is much slower than comparable-depth ResNets.
- **Stiff ODEs:** Some dynamics cause solvers to require very small step sizes (stiffness), dramatically increasing compute cost.
- **Expressive power:** The continuous constraint (the vector field must be smooth enough for ODE integration) limits what dynamics can be represented compared to unconstrained discrete networks.
- **Debugging difficulty:** The interior of the network is opaque — you don't have discrete layers to inspect.

## Practical Use

Neural ODEs are best suited for problems where their specific properties matter:
- Irregularly sampled or continuous-time data.
- Physical systems with known continuous dynamics.
- Memory-constrained settings requiring deep models.
- Generative modeling via continuous normalizing flows (FFJORD framework).

For standard image classification or NLP tasks, residual networks and transformers remain far more practical.
