---
title: AI for Earthquake Prediction
description: Explore how machine learning is advancing earthquake science — from deep learning seismic phase pickers (PhaseNet, EQTransformer) and aftershock forecasting to induced seismicity detection, ground motion prediction, and early warning systems. Covers convolutional and attention-based seismic models, physics-informed neural networks for wave propagation, and the challenges of rare-event prediction under non-stationary seismicity.
---

**AI for earthquake prediction** encompasses machine learning systems that detect seismic events, characterize earthquakes (phase picking, magnitude estimation, location), forecast aftershock sequences, predict ground shaking intensity, and contribute to early warning systems. While the holy grail of reliable short-term earthquake prediction (minutes to days before a major rupture) remains elusive, deep learning has dramatically improved the detection of weak and distant seismic events, accelerated seismic catalog construction, and enabled more accurate ground motion forecasting.

## Seismic Phase Picking

The most mature and impactful application of ML in seismology is **seismic phase picking** — identifying the arrival times of P-waves (primary compressional waves) and S-waves (secondary shear waves) in seismograms. Accurate phase picks are the foundation of earthquake location, magnitude estimation, and focal mechanism determination.

### PhaseNet

**PhaseNet** (Zhu & Beroza, 2019) is a U-Net-style convolutional neural network trained on labeled 3-component seismograms (vertical, north, east) from the Northern California Seismic Network. It outputs continuous probability functions for P and S arrival times at each sample, enabling sub-second temporal precision.

Key design choices:

- **3-channel input**: all three seismograph components processed jointly, allowing the model to exploit P-S amplitude ratio differences.
- **Encoder-decoder architecture**: the U-Net skip connections preserve temporal resolution for fine-grained arrival time estimation.
- **Probabilistic output**: Gaussian probability curves around predicted arrivals, naturally integrating with Bayesian location methods.

PhaseNet achieves mean absolute pick errors of $<0.1$ seconds on held-out data, comparable to expert human pickers and substantially outperforming classical STA/LTA (short-term/long-term average ratio) algorithms on noisy or overlapping waveforms.

### EQTransformer

**EQTransformer** (Mousavi et al., 2020) extends PhaseNet with a Transformer-based architecture incorporating **global self-attention** over the waveform sequence and a **multi-task learning head** that simultaneously:

1. Detects whether an earthquake is present in the window.
1. Picks P-wave arrival time.
1. Picks S-wave arrival time.

The attention mechanism allows EQTransformer to integrate information across the full waveform window — beneficial for identifying phase arrivals in waveforms contaminated by surface noise or cultural noise. Trained on STEAD (STanford EArthquake Dataset, 1.05 million labeled waveforms), EQTransformer generalizes to seismic networks worldwide without retraining.

### Benefits of Deep Seismic Pickers

Applying PhaseNet and EQTransformer to continuous seismic archives (years of data from regional networks) has produced **dramatically more complete seismic catalogs**: earthquake catalogs 5-10× larger than those produced by traditional human analysis, revealing previously undetected microseismicity ($M < 1$). This expanded catalog enables better characterization of fault structure, aftershock sequences, and seismic hazard.

## Aftershock Forecasting

**DeVries et al. (2018, Science)** demonstrated that a simple 2-layer fully connected neural network, trained on 131,000 mainshock-aftershock pairs, outperforms classic aftershock forecasting models (Coulomb stress, Omori-Utsu) in predicting the **spatial distribution** of aftershocks:

- Input: Coulomb stress components on a grid around the mainshock.
- Output: probability of aftershock occurrence at each grid cell.
- Improvement: ~6% AUC improvement over the best classical model.

The network learned to use **maximum shear stress** (not just Coulomb failure stress) as the primary predictor — a physically interpretable finding that emerged from data-driven learning.

**ETAS-ML** hybrid models combine the classical **Epidemic Type Aftershock Sequence (ETAS)** statistical framework with neural network components for better spatial-temporal aftershock rate forecasting, improving log-likelihood scores on held-out sequences from Italian and California earthquake catalogs.

## Ground Motion Prediction

Ground motion prediction equations (GMPEs) estimate peak ground acceleration (PGA) or spectral acceleration at a site given earthquake magnitude, depth, source-to-site distance, and local site conditions. ML models replace empirical GMPEs with data-driven predictors:

- **CNN-based GMPEs**: convolutional networks trained on accelerometer databases (NGA-West2, RESORCE) predict median PGA and aleatory variability with better residual variance than functional-form GMPEs.
- **Spatially-aware models**: graph neural networks model the spatial correlation of ground motions across recording stations, improving predictions in regions with sparse instrumentation.
- **Sequence-to-sequence models**: trained on early waveform snippets (first 1-3 seconds after P-arrival), predict full spectral response curves — relevant for **earthquake early warning** where rapid ground motion estimates are needed before S-wave arrival.

## Induced Seismicity Detection

**Induced seismicity** — earthquakes triggered by human activities (wastewater injection, hydraulic fracturing, geothermal operations) — requires careful monitoring to assess operational risk and comply with regulations.

ML methods contribute to induced seismicity monitoring by:

- **Discriminating induced from tectonic events**: classifiers trained on waveform features (frequency content, b-value distributions, spatio-temporal clustering) assign induced vs. tectonic probability to new events.
- **Template matching at scale**: cross-correlation-based matched filter detection, accelerated by GPU and approximate nearest neighbor algorithms, detects thousands of induced microearthquakes missed by standard cataloging.
- **Injection-seismicity correlation**: time-series models linking injection volumes/pressures to seismicity rates, predicting whether continued operations will cross risk thresholds.

## Physics-Informed Neural Networks for Seismology

**Physics-informed neural networks (PINNs)** augment standard neural networks with physics-based residual losses — ensuring outputs satisfy governing partial differential equations. In seismology:

- **Wave equation PINNs**: train networks to solve the 3D elastic wave equation in heterogeneous media, enabling rapid full-waveform simulation without finite-difference grid computations.
- **Velocity model inversion**: invert seismic waveforms for subsurface velocity structure using PINNs — improving on classical full-waveform inversion (FWI) for computing efficiency and gradient quality in complex models.
- **Rupture dynamics**: neural networks parameterize stress drop and fault geometry in kinematic rupture models, with physics constraints ensuring energy balance and radiation efficiency.

## Earthquake Early Warning

**Earthquake early warning (EEW)** systems detect the P-wave from an earthquake onset and issue alerts before the destructive S-wave arrives. ML enhances EEW in several ways:

- **Rapid magnitude estimation**: estimate $M_w$ from the first 3 seconds of P-wave waveform using deep CNNs — faster than the classical $P_d$ (peak displacement) method, with lower false alarm rates.
- **Shaking intensity maps**: real-time ML models generate probabilistic maps of expected shaking across a region as waveforms arrive at increasing distances from the epicenter.
- **Alert lead time optimization**: ML decision models trade off false alarm rate against alert lead time, tuning alert thresholds dynamically based on real-time waveform confidence scores.

**PLUM** (Propagation of Local Undamped Motion) and **EPIC** (Earthquake Point-source Integrated Code) EEW algorithms used by the ShakeAlert system on the US West Coast now incorporate ML-augmented magnitude estimation.

## Challenges in Earthquake Prediction

Despite significant progress, several fundamental challenges limit ML-based earthquake prediction:

- **Rare event statistics**: major earthquakes ($M > 7$) occur infrequently — insufficient training data for reliable short-term prediction of large events in specific regions.
- **Non-stationarity**: seismicity patterns are non-stationary (earthquake rates change over years due to stress evolution, transient loading, aftershock decay) — models trained on historical catalogs may not generalize to future seismicity.
- **Precursor signal controversy**: proposed precursory signals (GPS deformation anomalies, radon gas emissions, electromagnetic changes) have not been reproducibly validated — ML models trained to detect these signals risk learning spurious correlations in small datasets.
- **Physical interpretability**: for operational use in EEW and hazard assessment, black-box ML predictions must be interpretable and accompanied by reliable uncertainty estimates — an ongoing challenge for deep networks applied to sparse, noisy seismic data.

## Summary

Machine learning has transformed seismology through high-precision phase picking (PhaseNet, EQTransformer), producing earthquake catalogs 5-10× more complete than traditional methods. Aftershock forecasting networks outperform classical Coulomb stress models on spatial prediction; hybrid ETAS-ML models improve temporal forecasting. CNN-based GMPEs and spatially-aware graph networks advance ground motion prediction for hazard assessment. Physics-informed neural networks enable rapid wave simulation and velocity model inversion. Earthquake early warning systems now incorporate ML for rapid magnitude estimation and shaking map generation. The fundamental challenge of short-term earthquake prediction remains open — the field has not yet found reliable, reproducible physical precursors to imminent large earthquakes that ML could exploit.
