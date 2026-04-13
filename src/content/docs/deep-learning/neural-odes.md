---
title: Neural Ordinary Differential Equations
description: Explore Neural ODEs — a continuous-depth generalization of residual networks that model hidden state dynamics using differential equations, enabling adaptive computation, memory-efficient training, and natural handling of irregular time series.
---

Neural Ordinary Differential Equations (Neural ODEs) are a family of deep learning models that treat the **continuous transformation of hidden states** as a differential equation system, rather than a discrete sequence of layer operations. First proposed by Chen et al. (NeurIPS 2018), they represent a fundamentally different paradigm for deep learning that blurs the line between neural networks and dynamical systems.

## From Residual Networks to Differential Equations

The key insight behind Neural ODEs starts with the **residual connection** in ResNets:

$$h_{t+1} = h_t + f(h_t, \theta_t)$$

This is a discrete time Euler integration step for the ODE:

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

If ResNets are Euler discretizations of continuous dynamics, why not work in continuous time directly?

A Neural ODE defines hidden state dynamics with a neural network $f_\theta$:

$$\frac{dh(t)}{dt} = f_\theta(h(t), t)$$

The output at "time" $t_1$ is obtained by integrating from an initial state $h(t_0)$ to $t_1$:

$$h(t_1) = h(t_0) + \int_{t_0}^{t_1} f_\theta(h(t), t)\, dt$$

This integral is solved using a numerical ODE solver (e.g., Dormand-Prince, Runge-Kutta 4/5).

## Computing Gradients: The Adjoint Method

The standard backpropagation approach would require storing all intermediate states of the ODE solver — memory cost proportional to the number of function evaluations, which can be in the hundreds for an adaptive solver.

Neural ODEs solve this with the **adjoint sensitivity method** — a classical technique from optimal control theory:

1. Define the **adjoint** $a(t) = \frac{\partial L}{\partial h(t)}$ — the gradient of the loss with respect to the hidden state at any time $t$
2. The adjoint evolves backwards in time according to its own ODE:
$$\frac{da(t)}{dt} = -a(t)^T \frac{\partial f_\theta}{\partial h}$$
3. Gradients with respect to parameters are accumulated by integrating:
$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^T \frac{\partial f_\theta}{\partial \theta}\, dt$$

This backward ODE run requires only storing the initial and final states plus running the system backward — making gradient computation **constant memory**, regardless of the number of solver steps.

## Adaptive Depth

A key property of Neural ODEs is **adaptive computation**: the ODE solver uses more function evaluations for inputs that require more complex transformations (high curvature dynamics), and fewer for simpler inputs. This is automatic and input-dependent.

This is in contrast to fixed-depth networks where every input receives exactly the same number of operations regardless of difficulty — an inefficiency for heterogeneous real-world data.

## Comparison with Residual Networks

| Property | ResNet (Discrete) | Neural ODE (Continuous) |
|---|---|---|
| Layers | Fixed, discrete | Continuous, adaptive |
| Memory (training) | O(depth) | O(1) via adjoint |
| Parameterization | Per-layer weights | Single shared network $f_\theta$ |
| Extrapolation | At fixed positions | At any $t \in [t_0, t_1]$ |
| Trajectory control | None | Via ODE solver tolerance |
| Evaluation cost | Fixed per forward pass | Adaptive (error-controlled) |

## Latent ODEs and Generative Modeling

**Latent Neural ODEs** extend the framework to irregular time series and generative modeling:

1. An **encoder RNN** processes an observed sequence $x_{t_1}, \ldots, x_{t_n}$ into an initial latent state $z_0$
2. A **Neural ODE** evolves $z_0$ forward (or backward) continuously in time
3. A **decoder** maps the latent trajectory to observations at any desired time points

This is particularly powerful for:
- **Missing data:** The ODE naturally handles irregular observation times — no imputation needed
- **Extrapolation:** The latent trajectory can be rolled forward to predict future states
- **Clinical time series:** Patient vitals measured at irregular intervals fit naturally into this framework

## Augmented Neural ODEs

A theoretical limitation: Neural ODEs define **homeomorphisms** (bijective continuous maps) in the hidden state space. This means they cannot solve problems that require a topology change — for example, separating two interlocked rings in 2D requires "lifting" to a higher dimension.

**Augmented Neural ODEs** (Dupont et al., 2019) append extra dimensions to the hidden state (initialized to zero) and allow the dynamics to utilize them. This overcomes the homeomorphism constraint at minimal cost.

## Neural Controlled Differential Equations (CDE)

Neural CDEs (Kidger et al., 2020) extend Neural ODEs to handle **sequential input data** (not just evolution from an initial condition):

$$h(T) = h(t_0) + \int_{t_0}^T f_\theta(h(t))\, dX(t)$$

where $X(t)$ is a continuous interpolation of the input time series, and the integral is a **Stochastic (or controlled) integral** accumulating input information. Neural CDEs are the continuous analog of RNNs — processing sequential inputs in continuous time.

## Neural Stochastic Differential Equations (SDE)

Adding a diffusion term introduces **stochasticity** into the dynamics:

$$dh(t) = f_\theta(h(t), t)\, dt + g_\phi(h(t), t)\, dW(t)$$

where $W(t)$ is a Wiener process (Brownian motion). Neural SDEs enable probabilistic trajectory modeling — natural for uncertainty-aware time series forecasting and generative modeling.

## Connection to Score-Based Diffusion Models

There is a deep theoretical connection between Neural ODEs/SDEs and **diffusion models**. The forward noising process of a diffusion model is an SDE; the reverse denoising process is also an SDE (or, under specific conditions, an ODE — the **probability flow ODE**). Neural ODEs with the probability flow formulation enable:
- Exact likelihood computation for diffusion models
- Faster, deterministic sampling without stochasticity
- DDIM sampling as a special case

## Applications

- **Physics simulation:** Learning continuous dynamics from trajectory data
- **Pharmacokinetics:** Modeling drug concentration over time with irregular sampling
- **Normalizing flows:** ODE-based flows (FFJORD) for density estimation with $O(d)$ cost (vs. $O(d^3)$ for exact Jacobian)
- **Graph dynamical systems:** Combining Neural ODEs with GNNs for spatiotemporal evolution
- **Robotic control:** Continuous-time dynamics learning for model-based RL

## Further Reading

- Chen et al. (2018), *Neural Ordinary Differential Equations* — NeurIPS Best Paper
- Dupont et al. (2019), *Augmented Neural ODEs*
- Kidger et al. (2020), *Neural Controlled Differential Equations for Irregular Time Series*
- Tzen & Raginsky (2019), *Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit*
- Grathwohl et al. (2019), *FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models*
