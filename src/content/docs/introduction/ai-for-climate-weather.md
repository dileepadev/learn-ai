---
title: AI for Climate and Weather Modeling
description: Discover how deep learning models like GraphCast, Pangu-Weather, and FourCastNet are transforming weather prediction and climate science — outperforming traditional numerical models in speed and accuracy while opening new frontiers in climate projection and extreme event forecasting.
---

Weather forecasting and climate modeling are among the most computationally demanding scientific disciplines. For 70 years, **numerical weather prediction (NWP)** — solving partial differential equations that govern atmospheric physics on massive supercomputers — has been the gold standard. AI-based models are now challenging this paradigm, producing comparable or better 10-day forecasts in **seconds** rather than **hours**, at a fraction of the cost.

## How Traditional NWP Works

Numerical weather prediction divides the atmosphere into a 3D grid of cells, typically 9–25 km horizontal resolution, and propagates the physical state of each cell forward in time by solving:

- **Navier-Stokes equations** for fluid dynamics (wind, pressure)
- **Thermodynamic equations** for temperature and moisture
- **Radiative transfer equations** for solar and infrared energy

The European Centre for Medium-Range Weather Forecasts (ECMWF) HRES model is the gold standard NWP system, requiring thousands of CPU-hours per forecast.

## The AI Model Revolution

AI-based weather models learn to emulate the mapping from the current atmospheric state to the future state **directly from reanalysis data** (ERA5 — a historical gridded reconstruction of global atmospheric conditions from 1940 to present).

Common input/output format:
- **Input:** State of the atmosphere at time $t$ (temperature, wind, humidity, surface pressure at multiple pressure levels, across a global grid)
- **Output:** Predicted atmospheric state at time $t + \Delta t$ (typically $\Delta t = 6$ hours)
- **Autoregressive rollout:** Iterate to produce multi-day forecasts

### Key AI Weather Models

| Model | Team | Architecture | Key Contribution |
|---|---|---|---|
| **FourCastNet** | NVIDIA | Fourier Neural Operator (SFNO) | First competitive global AI forecast |
| **Pangu-Weather** | Huawei | 3D Earth Attention Transformer | Outperformed ECMWF on 1–7 day forecasts |
| **GraphCast** | Google DeepMind | Graph Neural Network | SOTA on 10-day forecasts; Nature paper |
| **GenCast** | Google DeepMind | Diffusion model on spherical latent | Probabilistic forecasts, ensemble |
| **Aurora** | Microsoft | Swin Transformer + 3D attention | Generalizes to air quality, ocean |
| **NeuralGCM** | Google Research | Hybrid: Learned dynamics + ODE solver | Interpretable, stable long-range |

## GraphCast

GraphCast (Lam et al., 2023, *Science*) uses a **multiscale Graph Neural Network** that operates over an icosahedral mesh of the Earth:

1. **Encoder:** Map the lat/lon grid-based input state to mesh nodes via learned interpolation
2. **Processor:** 16 rounds of GNN message passing over a hierarchical mesh (coarse-to-fine scale)
3. **Decoder:** Map mesh node states back to lat/lon grid

GraphCast processes 10 days of global weather in under 1 minute on a single TPU v4, compared to hours for ECMWF HRES. It matches or beats HRES on 90% of tracked variables over 10 days at 0.25° resolution.

## Pangu-Weather

Pangu-Weather (Bi et al., 2023, *Nature*) uses a 3D Earth Transformer with separate models for 1h, 3h, 6h, and 24h lead times. A hierarchical temporal aggregation strategy produces longer forecasts by composing short-step models.

Key insight: separating models by lead time avoids error accumulation from iterative rollouts.

## Probabilistic Forecasting: GenCast

Deterministic models predict the single most likely atmospheric trajectory. In practice, small initial condition errors cause uncertainty to grow. **GenCast** (Price et al., 2024) produces **ensemble forecasts** — multiple plausible future weather trajectories — by formulating forecast as a **diffusion process on a compressed latent space**.

This is critical for:
- **Extreme event prediction:** Representing the tail distribution of possible hurricane tracks
- **Operational forecasting:** Uncertainty quantification for decision-making (flood evacuation, aviation routing)

## Hybrid Models: NeuralGCM

**NeuralGCM** (Kochkov et al., 2024, *Nature*) combines learned dynamics with traditional **ODE solvers**. A neural network represents sub-grid-scale physics (clouds, turbulence, convection) that classical models parameterize with crude approximations, while the large-scale dynamics are solved by a traditional differentiable solver.

This hybrid approach produces long-range stable forecasts that are also interpretable and conserve physical quantities.

## Beyond Forecasting: AI for Climate Science

### Downscaling
Global climate models run at coarse resolution (50–100 km). **Statistical downscaling** uses deep learning (super-resolution CNNs) to predict fine-scale local conditions from coarse model output:

- DeepSD, ClimateGAN, and diffusion-based downscaling
- Critical for regional impact assessments (drought, flood risk at county scale)

### Extreme Event Attribution
AI accelerates **climate attribution** — estimating how much climate change has altered the probability of specific extreme events (heatwaves, hurricanes, floods):
- Train a model on climate simulations with and without anthropogenic forcing
- Evaluate the counterfactual probability of an observed event

### Carbon Flux Estimation
Neural networks trained on satellite data (MODIS, Sentinel, OCO-2) estimate:
- Terrestrial carbon uptake by forests and vegetation
- Ocean CO₂ absorption
- Methane emission patterns from agriculture and wetlands

### Ice Sheet and Sea Level
AI models speed up ice sheet dynamics simulations critical for long-range sea level rise projections (century-scale). Neural network emulators replace expensive ice dynamics solvers in large ensemble studies.

## NVIDIA Earth-2

NVIDIA's **Earth-2** initiative builds a **digital twin of the Earth's climate system**:
- **CorrDiff:** A km-scale diffusion model for downscaling to 2.5 km resolution
- **FourCastNet 2.0:** Planetary-scale forecast backbone
- **Inference on Omniverse:** Interactive visualization of climate simulations

The goal is to make high-resolution climate impact projections accessible to governments and infrastructure planners.

## Challenges

### Distribution Shift
AI models trained on ERA5 reanalysis data (1940–present) may underperform in a future climate that diverges significantly from historical conditions — especially for rare extremes.

### Physical Consistency
NWP models conserve mass, energy, and momentum by construction. AI models may violate these conservation laws, especially over long rollout periods, leading to unphysical states (negative precipitation, exploding temperature).

### Data Availability
ERA5 is the dominant training corpus, but it is itself a model-based reconstruction with its own errors. Ocean, stratosphere, and polar data are less reliable.

### Interpretability
NWP models provide physically interpretable intermediate states. Black-box AI models do not naturally explain *why* they predict a hurricane will intensify.

## Further Reading

- Lam et al. (2023), *Learning Skillful Medium-Range Global Weather Forecasting (GraphCast)*, Science
- Bi et al. (2023), *Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks (Pangu-Weather)*, Nature
- Kochkov et al. (2024), *Neural General Circulation Models for Weather and Climate (NeuralGCM)*, Nature
- Price et al. (2024), *GenCast: Diffusion-based Ensemble Weather Forecasting at Scale*
- Bodnar et al. (2024), *Aurora: A Foundation Model of the Atmosphere*, Microsoft Research
