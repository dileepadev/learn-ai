---
title: AI for Weather Forecasting
description: Discover how machine learning foundation models are transforming numerical weather prediction — covering GraphCast, Pangu-Weather, FourCastNet, NeuralGCM, and Aurora — with key architectural choices, skill scores, and the path toward operational AI weather forecasting.
---

Weather forecasting has been one of the great success stories of computational science: modern 5-day forecasts are more accurate than 1-day forecasts were 40 years ago, driven by increasingly accurate numerical weather prediction (NWP) models that simulate atmospheric physics on supercomputers. In 2023, a wave of AI-based forecasting models dramatically challenged NWP supremacy — matching or exceeding the accuracy of ECMWF's Integrated Forecasting System (IFS), the world's best operational NWP model, at a fraction of the computational cost.

## Background: Numerical Weather Prediction

Traditional NWP works by numerically integrating the equations of atmospheric dynamics (Navier-Stokes, thermodynamics, continuity) on a 3D global grid, updated every 6–12 hours with assimilated observations from satellites, radiosondes, and surface stations. ECMWF's IFS runs at ~9 km horizontal resolution with 137 vertical levels and requires millions of CPU-hours per forecast. While physically principled and interpretable, NWP is:

- Computationally expensive: a 10-day ensemble forecast costs ~10,000 CPU-core-hours.
- Slow to produce: the full forecast pipeline takes 1–2 hours of wall-clock time.
- Constrained by the parameterization problem: subgrid-scale processes (convection, cloud microphysics) must be approximated.

AI models learn the mapping from an initial atmospheric state directly to future states, bypassing explicit physics simulation.

## The Reanalysis Training Paradigm

All major AI weather models are trained on **ERA5** — ECMWF's fifth-generation atmospheric reanalysis, covering 1979–present at 0.25° resolution (~28 km), 37 pressure levels, and hourly frequency. ERA5 contains ~6.5 petabytes of gridded data representing temperature, wind, geopotential, humidity, and more at each grid point.

The standard supervised learning setup:

- **Input:** Global atmospheric state at time $t$ (and usually $t - 6h$ for velocity tendencies).
- **Target:** Atmospheric state at time $t + \Delta t$ (one step ahead).
- **Autoregressive rollout:** Iteratively apply the model to produce forecasts at $t, t+\Delta t, t+2\Delta t, \ldots$

The models are evaluated against ECMWF IFS on the **WeatherBench 2** benchmark.

## GraphCast (Google DeepMind, 2023)

**GraphCast** (Lam et al., 2023) uses a **Graph Neural Network** over a multi-scale icosahedral mesh to model atmospheric dynamics.

### GraphCast Architecture

- **Grid-to-mesh encoder:** Projects ERA5 grid points onto graph nodes in an icosahedral mesh (O(40,000 nodes) at ~1° resolution).
- **Processor:** 16 layers of message passing on the mesh, with edges representing both local (short-range) and long-range connections across multiple mesh resolutions.
- **Mesh-to-grid decoder:** Projects mesh node features back to the 0.25° ERA5 grid.

The multi-resolution mesh (6 levels of icosahedral refinement) allows GraphCast to efficiently propagate information across both local and planetary scales — crucial for weather phenomena that span continents.

### GraphCast Performance

GraphCast achieves better skill than ECMWF IFS on **90% of 1,380 verification targets** (all pressure levels, all variables, all lead times from 6h to 10 days) on the HRES analysis. It is particularly strong at:

- **Medium-range forecasting (3–7 days):** 500 hPa geopotential height, 850 hPa temperature.
- **Tropical cyclone tracking:** 15–20% improvement in track error vs. IFS.
- **Extreme event 2m temperature forecasts.**

## Pangu-Weather (Huawei, 2023)

**Pangu-Weather** (Bi et al., 2023) uses a 3D Earth attention transformer trained to minimize a weighted $\ell_1$ + $\ell_2$ loss on ERA5.

### Pangu-Weather Architecture

A hierarchical transformer with:

- **3D window attention:** Attention computed within overlapping 3D windows in the (latitude, longitude, pressure level) space, capturing local vertical and horizontal correlations.
- **Shifted windows:** Analogous to Swin Transformer — alternating attention windows across layers provide cross-window communication.
- **Multi-scale design:** Four separate models for forecast lead times of 1h, 3h, 6h, and 24h; at inference, models are combined to produce arbitrary-horizon forecasts with minimal error accumulation.

The multi-resolution model strategy (separate networks per horizon) reduces autoregressive error accumulation: instead of rolling out 40 consecutive 6-hour steps to reach 10 days, Pangu-Weather can use fewer, larger steps.

### Pangu-Weather Performance

Pangu-Weather surpasses ECMWF IFS HRES on **nearly all upper-air variables** at forecast horizons of 1–7 days, with particular strength on upper-level wind forecasts.

## FourCastNet (NVIDIA, 2022)

**FourCastNet** (Pathak et al., 2022) was the first AI model to demonstrate competitive NWP skill at global resolution, predating GraphCast and Pangu-Weather. It uses an **Adaptive Fourier Neural Operator (AFNO)** backbone:

- The global atmospheric state is processed as a 2D image at each pressure level.
- AFNO applies token mixing in Fourier space — efficient global mixing at $O(n \log n)$ cost.
- FourCastNet generates a 10-day forecast in under 2 seconds on a single A100 GPU.

FourCastNet enabled **large ensemble forecasting** at previously unaffordable scale: generating 1,000-member ensembles to characterize forecast uncertainty, vs. the 51-member operational ECMWF ensemble.

## NeuralGCM (Google, 2024)

**NeuralGCM** (Kochkov et al., 2024) represents a hybrid approach: combining a differentiable atmospheric dynamical core (physics equations for large-scale dynamics) with neural network parameterizations for subgrid processes.

Unlike pure ML models, NeuralGCM:

- Enforces physical conservation laws (mass, energy, momentum) exactly through the dynamical core.
- Uses learned neural networks only for the parameterization residuals (convection, cloud processes) that NWP traditionally approximates with handcrafted schemes.
- Is end-to-end differentiable — the full hybrid model can be trained with gradient descent.

NeuralGCM outperforms both pure ML models and IFS at multi-week horizons (beyond 10 days) where ML models' error accumulation begins to dominate. It also produces better calibrated uncertainty estimates.

## Aurora (Microsoft, 2024)

**Aurora** is a large foundation model (1.3B parameters) pretrained on a diverse corpus of atmospheric data: ERA5, CMIP6 climate model outputs, MERRA-2 reanalysis, HRES analysis, and regional high-resolution datasets. Key features:

- **3D Swin Transformer** backbone operating on pressure-level and surface variables jointly.
- **Variable-resolution pretraining:** Trained on data at multiple resolutions (0.1° to 1°), enabling zero-shot adaptation to new grids.
- **Multistep fine-tuning:** Fine-tuned with autoregressive rollout loss over multiple steps to improve long-range stability.

Aurora achieves state-of-the-art performance on AIFS (ECMWF's operational AI model) benchmarks at 0.1° resolution, with strong performance on **air quality forecasting** and **ocean wave height** in addition to standard atmospheric variables.

## Skill Metrics

AI weather models are evaluated using standard meteorological skill metrics:

| Metric | Definition | Notes |
| --- | --- | --- |
| RMSE | Root mean square error vs. ERA5 analysis | Lower is better |
| ACC | Anomaly correlation coefficient | Higher is better; >0.6 threshold for "useful" forecast |
| FSS | Fraction Skill Score | For precipitation and extreme events |
| CRPS | Continuous Ranked Probability Score | For probabilistic forecasts |

The **500 hPa geopotential height Z500** is the canonical benchmark variable: it governs large-scale weather patterns and is reliably observed and analyzed. All major AI models achieve ACC > 0.6 out to **~10 days** on Z500 — matching or exceeding IFS.

## Advantages and Limitations of AI Weather Models

### Advantages

- **Speed:** Generate a 10-day global forecast in seconds on a single GPU vs. hours on a supercomputer.
- **Cost:** $100–$1,000 per forecast vs. $100,000+ for operational NWP.
- **Ensemble generation:** Affordable 1,000+ member ensembles for improved uncertainty quantification.
- **Extreme event skill:** Some AI models show improved skill on rare extremes vs. IFS.

### Limitations

- **Blurry high-resolution predictions:** Autoregressive ML models tend to produce spatially smooth (blurry) precipitation forecasts at fine scales — failing to capture extreme local intensity.
- **Physical consistency:** AI forecasts may violate conservation laws (mass, energy); wind and temperature fields may be locally inconsistent.
- **Training data ceiling:** All models are trained on ERA5, which itself has biases and limited resolution. Models cannot generalize beyond ERA5's quality.
- **Rare events:** Extremely rare events (once-per-century storms) are poorly represented in 40 years of training data.
- **Climate change extrapolation:** ML models trained on historical data may not extrapolate to a warming climate that diverges from historical distributions.

## The Path to Operational Deployment

ECMWF launched its own AI model **AIFS** (Artificial Intelligence Forecasting System) in 2024, integrating an AI model into operational forecasting alongside IFS. The hybrid approach:

1. AI model provides fast, accurate deterministic and ensemble forecasts.
1. IFS provides physically consistent, high-resolution forecasts for safety-critical decisions.
1. AI models are used for **post-processing** — correcting systematic biases in NWP output (Model Output Statistics, MOS).

National weather agencies (NOAA, ECMWF, MetOffice) are actively testing AI models for operational use, with full AI-primary forecasting expected to begin by 2027.

## Summary

AI weather foundation models — GraphCast, Pangu-Weather, FourCastNet, NeuralGCM, and Aurora — have demonstrated that machine learning trained on ERA5 reanalysis can match or exceed the world's best NWP models at medium-range forecasting, at 10,000× lower compute cost. GNN-based architectures (GraphCast) and hierarchical transformers (Pangu-Weather, Aurora) capture planetary-scale dynamics; hybrid physics-ML models (NeuralGCM) combine physical conservation with learned parameterizations. Key remaining challenges are high-resolution precipitation, physical consistency, and extrapolation to climate change scenarios.
