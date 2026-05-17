---
title: AI for Carbon Capture and Storage
description: Explore how machine learning accelerates carbon capture and storage — from discovering novel sorbent materials and optimizing DAC/point-source processes to selecting geological storage sites, monitoring sequestration integrity, and modeling techno-economic feasibility.
---

Carbon capture and storage (CCS) is a family of technologies that prevent CO₂ from entering the atmosphere — either by capturing it at emission sources or pulling it directly from the air, then permanently storing it underground. Scaling CCS to gigatonne levels requires breakthroughs in materials, process efficiency, and monitoring that AI is increasingly positioned to deliver.

## The CCS Challenge

The global carbon budget to limit warming to 1.5°C requires removing or preventing billions of tonnes of CO₂ emissions annually by mid-century. CCS faces three bottlenecks:

1. **Materials cost**: current sorbents and membranes are expensive, degrade quickly, or require high energy for regeneration
1. **Process efficiency**: capture, compression, transport, and injection consume 15–30% of a plant's output energy
1. **Verification**: confirming that injected CO₂ stays in formation requires dense, expensive monitoring

Machine learning addresses all three by accelerating materials discovery, optimizing process operations, and enabling intelligent monitoring from sparse sensor data.

## AI for Sorbent Material Discovery

The heart of most capture processes is a sorbent — a material that selectively binds CO₂. Metal-organic frameworks (MOFs), zeolites, amines, and ionic liquids are candidates, but the design space is astronomically large: millions of hypothetical MOFs alone.

### Graph Neural Networks for MOF Screening

MOFs are naturally represented as graphs: metal nodes connected by organic linkers with 3D spatial structure. GNNs learn structure-property relationships directly from this representation:

```python
import torch
from torch_geometric.nn import DimeNet, SchNet

# SchNet: message passing over atomic distances
model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=10.0,  # Angstroms
)

# Input: atomic numbers, positions
# Output: predicted CO2 adsorption capacity (mmol/g)
```

Models like SchNet, DimeNet, and CGCNN predict CO₂ uptake, selectivity over N₂, working capacity, and regeneration energy from crystal structure alone — enabling virtual screening of millions of structures before synthesis.

### Generative Design

Rather than screening known structures, generative models propose novel MOFs optimized for capture performance:

- **Variational Autoencoders**: encode MOF topology and linker chemistry into a latent space; decode optimized structures from points with high predicted CO₂ uptake
- **Diffusion models**: generate valid 3D atomic configurations conditioned on target adsorption properties
- **Multi-objective optimization**: Pareto-optimize over CO₂ capacity, selectivity, stability, and synthesis feasibility using evolutionary algorithms guided by surrogate GNN models

### Experimental Synthesis Guidance

ML models trained on experimental databases (CoRE MOF, hMOF) guide synthesis priorities. Active learning loops close the cycle:

1. GNN model predicts top-$k$ candidate structures from hypothetical library
1. Synthesis team produces a subset; grand canonical Monte Carlo (GCMC) simulations validate adsorption
1. Experimental measurements update the training set
1. Model is retrained with new data, improving predictions in the most uncertain chemical space regions

## Direct Air Capture Process Optimization

Direct air capture (DAC) systems contact ambient air with a sorbent, then apply heat or moisture-swing cycles to release concentrated CO₂. Process efficiency is highly sensitive to operating conditions.

### Temperature-Swing Adsorption

In solid-sorbent DAC, the capture-regeneration cycle involves:

$$\Delta E_{\text{regen}} = \int_{T_{\text{cap}}}^{T_{\text{regen}}} C_p \, dT + \Delta H_{\text{ads}}$$

where $\Delta H_{\text{ads}}$ is the adsorption enthalpy and $C_p$ is the sorbent heat capacity. Minimizing $\Delta E_{\text{regen}}$ while maximizing CO₂ purity requires optimizing temperature profiles, flow rates, and cycle timing simultaneously.

Reinforcement learning agents have been applied to control the valve and heater schedules of DAC units:

```python
import gymnasium as gym
from stable_baselines3 import PPO

# State: sorbent loading, temperature profile, CO2 concentration, humidity
# Action: heater setpoints, fan speeds, valve positions
# Reward: CO2 captured per kWh of electricity consumed

env = DACControlEnvironment(
    sorbent_model="physics_sim",
    ambient_conditions={"T": 298, "RH": 0.5, "CO2_ppm": 420},
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

Trained RL controllers have demonstrated 10–20% efficiency improvements over fixed cycle schedules in simulation, with ongoing pilot deployments at commercial DAC facilities.

### Moisture-Swing Systems

Moisture-swing DAC (Lackner process) uses humidity to swing the sorbent between capture (dry) and release (wet) states — potentially using ambient wind energy. ML models predict optimal sorbent hydration cycles based on real-time weather forecasts:

- LSTM or Transformer models consume weather forecasts (temperature, humidity, wind speed, solar irradiance)
- Output: optimal exposure/regeneration schedule for the next 24–48 hours
- Objective: maximize net CO₂ captured per unit time subject to energy budget

## Point-Source Capture Optimization

Post-combustion capture at power plants and industrial facilities uses amine scrubbing: flue gas bubbles through an amine solution that absorbs CO₂, which is then stripped off by steam heating.

### Digital Twins for Amine Scrubbers

Physics-informed neural networks (PINNs) and surrogate models trained on high-fidelity process simulations enable real-time optimization of absorber-stripper columns:

$$\text{minimize} \quad E_{\text{reboiler}} \quad \text{s.t.} \quad X_{\text{CO}_2} \geq 0.90$$

where $X_{\text{CO}_2}$ is the capture fraction. Control variables include lean solvent loading, solvent flow rate, absorber temperature, and stripper pressure.

Bayesian optimization with surrogate models handles the continuous optimization at each operating point, adapting to varying flue gas flow rates and compositions as plant load changes.

### Predictive Maintenance

Amine degradation, foaming, and heat exchanger fouling reduce capture efficiency and increase costs. ML fault detection models monitor:

- Amine concentration and degradation products via online sensors + regression models
- Pressure drop anomalies indicating flooding or foaming (LSTM-based anomaly detection)
- Heat exchanger fouling via thermal resistance estimation from temperature measurements

## Geological Storage Site Selection

Injected CO₂ must remain sequestered for thousands of years in porous rock formations deep underground. Site selection requires evaluating capacity, injectivity, containment integrity, and proximity to emission sources.

### ML-Guided Site Screening

Traditionally, site evaluation requires expensive seismic surveys and exploratory wells. ML models trained on existing storage projects and analog geological data provide rapid screening:

- **Random forests / gradient boosting**: classify formations as suitable/unsuitable based on available geological attributes (porosity, permeability, depth, caprock thickness, fault density)
- **Graph neural networks**: model structural connectivity of formation-caprock-fault systems
- **Gaussian process regression**: interpolate subsurface property fields from sparse borehole data with uncertainty quantification

### Injection Optimization

Once a site is selected, CO₂ injection rates and pressures must be managed to avoid induced seismicity and caprock breach:

```python
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# Surrogate model over injection parameters
# State: formation pressure, saturation plume extent
# Objective: maximize injection rate while keeping pressure below threshold

gp_model = SingleTaskGP(train_X=X_obs, train_Y=y_obs)
EI = ExpectedImprovement(model=gp_model, best_f=y_obs.max())

candidate, acq_value = optimize_acqf(
    acq_function=EI,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=512,
)
```

## Monitoring, Reporting, and Verification

Regulatory frameworks require verifying that stored CO₂ does not leak. Monitoring relies on seismic surveys, satellite InSAR, groundwater geochemistry, and atmospheric sampling.

### Seismic Monitoring with Deep Learning

4D seismic surveys (repeated 3D surveys over time) detect changes in rock properties caused by CO₂ plume migration. CNNs and 3D vision transformers process seismic amplitude volumes to:

- Detect and map the growing CO₂ plume
- Identify potential leakage pathways (fault reactivation)
- Estimate saturation and pressure changes

Training data combines reservoir simulation outputs with seismic forward modeling — generating synthetic labeled pairs for supervised learning without requiring labeled field examples.

### Satellite InSAR Deformation Monitoring

CO₂ injection causes surface uplift detectable by Interferometric Synthetic Aperture Radar (InSAR). ML models decode deformation signals:

- Separate CO₂ injection signal from atmospheric noise, seasonal effects, and other geophysical signals
- Invert surface deformation to estimate subsurface pressure build-up using physics-constrained neural networks
- Detect anomalous deformation patterns that may indicate caprock breach

### Atmospheric Flux Inversion

Detecting surface leakage from sparse atmospheric CO₂ sensors is an inverse problem: given sensor readings, infer the surface flux distribution. Neural network emulators replace expensive atmospheric transport models (like GEOS-Chem) in Bayesian inversion frameworks, reducing computation from weeks to minutes:

$$p(\text{flux} \mid \text{sensors}) \propto p(\text{sensors} \mid \text{flux}) \cdot p(\text{flux})$$

Neural posterior estimation (NPE) trains an amortized inference network on simulation-observation pairs, enabling real-time leak detection from operational sensor networks.

## Techno-Economic Modeling

Deploying CCS at scale requires understanding cost trajectories and identifying bottlenecks. ML enhances techno-economic analysis (TEA) in several ways:

### Surrogate Models for Cost Estimation

High-fidelity process simulation (Aspen Plus, HYSYS) is computationally expensive. Surrogate neural networks trained on simulation outputs predict:

- Capture cost ($/tonne CO₂) as a function of plant size, fuel type, and sorbent cost
- Energy penalty (% of plant output) as a function of process configuration
- Capital cost scaling with capture rate

These surrogates enable rapid scenario exploration and uncertainty quantification across thousands of parameter combinations.

### Learning Curves and Technology Forecasting

Historical data on deployment and cost for solar PV, wind, and batteries reveals that costs follow experience curves (Wright's Law):

$$C(Q) = C_0 \cdot Q^{-\alpha}$$

where $Q$ is cumulative deployed capacity and $\alpha$ is the learning rate. ML methods fit these curves to sparse CCS deployment data and forecast long-run capture costs under different deployment trajectories — informing policy and investment decisions.

## Challenges and Open Problems

| Challenge | Current State | Research Frontier |
| --- | --- | --- |
| Sorbent stability prediction | GNNs predict initial properties well | Long-term degradation modeling under cycling |
| Out-of-distribution generalization | Models fail on novel chemistries | Pre-training on large simulation databases |
| Leakage detection sensitivity | ~1% leakage detectable with dense networks | Sub-0.1% detection with sparse sensors |
| DAC scale-up | ML-optimized pilots at small scale | Transferring controllers to 10x larger systems |
| Geological uncertainty | GP regression for property interpolation | Full uncertainty propagation in storage estimates |

## Summary

AI is accelerating carbon capture and storage across the full technology stack:

- **Materials discovery**: GNNs and generative models screen millions of hypothetical sorbents and MOFs, guiding synthesis toward high-performance candidates
- **Process optimization**: RL and Bayesian optimization controllers improve DAC and amine scrubber efficiency by 10–20%
- **Site selection**: ML classifiers and GPs enable rapid subsurface screening from sparse geological data
- **Monitoring**: deep learning on seismic, InSAR, and atmospheric data provides cost-effective sequestration verification
- **TEA**: surrogate models and learning curves inform policy by enabling fast, uncertainty-aware cost projections

As CCS deployment accelerates toward gigaton scale, ML-driven improvements in each of these areas translate directly into lower costs, higher confidence in permanence, and faster technology maturation — making AI a central tool in the climate response toolkit.
