---
title: Foundation Models for Time Series
description: Explore how large pre-trained foundation models are transforming time series analysis — from zero-shot forecasting with TimesFM and Chronos to anomaly detection and classification, removing the need for per-dataset model training.
---

Foundation models for time series are large pre-trained models that can perform forecasting, anomaly detection, and classification on **arbitrary time series data** — without being retrained for each new dataset. Drawing inspiration from LLMs for text and ViTs for images, these models are trained on vast collections of diverse time series and applied zero-shot or with minimal fine-tuning.

## Why Time Series Is Different

Time series foundation models face unique challenges compared to language or vision:

- **Heterogeneous scales and units:** Temperature, stock prices, CPU utilization, and ECG signals have vastly different value ranges and semantics
- **Variable frequency:** Data may be sampled every second, every hour, or monthly — with no single canonical "token" length
- **Domain-specific patterns:** Seasonality, trends, and noise patterns differ radically across domains
- **Limited transfer:** A pattern learned from financial data may not transfer to meteorological data
- **Long-range dependencies:** Some series require lookback windows of thousands of points to capture seasonal patterns

## The Classical Approach vs. Foundation Models

### Classical Per-Dataset Models
Traditionally, time series models must be fitted to each dataset separately:
- **ARIMA/SARIMA:** Statistical models fit per series
- **Prophet:** Additive decomposition with holidays and seasonality
- **LSTM/Transformer:** Trained from scratch on domain-specific data

**Limitation:** No transfer across datasets; requires historical data for every new series; expensive for high-cardinality forecasting (millions of series).

### Foundation Model Approach
Pre-train once on diverse corpora → apply anywhere:

```
[Energy consumption data]
[Retail sales data]         → Pre-train → FTM  →  Zero-shot forecast
[Traffic data]                              ↓      on any new series
[Medical sensors data]               Fine-tune if needed
```

## Chronos (Amazon)

**Chronos** (Ansari et al., 2024) treats time series forecasting as **language modeling over quantized values**. The key steps:

1. **Normalization:** Scale each series by its mean absolute value
2. **Quantization:** Map continuous values to discrete bins (vocabulary of ~4096 tokens)
3. **Language modeling:** Pre-train a T5-based encoder-decoder model autoregressively on the token sequence
4. **Inference:** Decode token sequences and dequantize back to continuous values

Chronos can forecast any series zero-shot by tokenizing its history and sampling from the learned distribution. It was pre-trained on a corpus of 84 billion time series observations from diverse public datasets.

**Strengths:**
- Probabilistic forecasts (sample multiple futures)
- Strong zero-shot performance, especially on long-horizon forecasting
- Simple conceptual framework — inherits LLM scaling properties

## TimesFM (Google DeepMind)

**TimesFM** (Das et al., 2024) uses a patched decoder-only Transformer architecture:

- **Patching:** The time series is segmented into non-overlapping patches (e.g., 32 values per patch), each treated as a "token"
- **Causal attention** over patches enables autoregressive forecasting
- Pre-trained on **100 billion time points** from Google Trends, Wikipedia pageviews, and synthetic data

TimesFM supports variable context lengths and prediction horizons via input padding and horizon-specific output heads. It achieves zero-shot performance competitive with domain-specific supervised models on many benchmarks.

## MOIRAI (Salesforce)

**MOIRAI** (Unified Training of Universal Time Series Forecasting Transformers) introduces:
- A **unified tokenization** that handles multiple frequencies natively (hourly, daily, monthly) via patch size adaptation
- **Any-variate attention:** The model can handle multivariate series with variable numbers of channels
- A **mixture distribution head** that outputs a flexible probability distribution over future values

MOIRAI was trained on the LOTSA dataset — a 27 billion observation corpus from 9 diverse domains.

## MOMENT (CMU)

**MOMENT** takes a masked autoencoder pre-training approach (analogous to BERT for time series):
- Randomly mask patches of the input series
- Pre-train a Transformer to reconstruct the masked patches

This produces general-purpose representations that can be fine-tuned for:
- **Forecasting:** Project the final representation to future values
- **Classification:** Classify patterns (ECG rhythms, activity recognition)
- **Anomaly detection:** Reconstruction error on masked patches identifies anomalies
- **Imputation:** Fill in missing observations

## LLM-Based Time Series Models

Some approaches directly **use existing LLMs** for time series:

### TimeLLM
Reprograms a frozen LLM (Llama 2) for time series by mapping time series patches into the LLM's text embedding space via a learned reprogramming layer. The LLM backbone is frozen; only the lightweight reprogramming module is trained.

### GPT4TS (One Fits All)
Fine-tunes GPT-2 on time series data, freezing the self-attention layers and only training embedding and output layers. Achieves competitive performance even with minimal adaptation.

## Benchmarking Foundation Models for Time Series

### Darts Benchmark / LibriTS
Standard forecasting benchmarks: ETTh1, ETTm2, Weather, Electricity, Traffic datasets.

### Monash Forecasting Repository
135 diverse datasets across domains — canonical benchmark for universal forecasting models.

### GIFT-Eval (Salesforce)
A recent foundation model-specific benchmark covering 23 datasets across 7 domains, designed to evaluate zero-shot generalization.

**Findings:** Foundation models generally:
- Excel at zero-shot forecasting on short-to-medium horizons
- Struggle with very long-horizon forecasting (1000+ steps) relative to domain-specific specialists
- Show strong advantage in low-data scenarios where per-dataset training is impractical

## Handling Multivariate Series

Most early foundation models handle **univariate** series (one variable at a time). Extending to multivariate requires:
- **Channel-independent (CI) strategy:** Apply the foundation model to each variable independently; ignore inter-variable correlations
- **Channel-dependent (CD) strategy:** Jointly model cross-variable relationships

CI approaches transfer better across domains; CD approaches are stronger when cross-variable patterns are consistent (e.g., correlated sensors).

## Applications

- **Retail demand forecasting:** Zero-shot forecasts for millions of new SKUs without historical data
- **Energy grid management:** Rapid deployment on new substations with limited history
- **Clinical monitoring:** Anomaly detection on vital signs with limited patient history
- **Finance:** First-day forecasting for newly listed securities
- **IoT:** Forecasting and anomaly detection for newly deployed sensors

## Key Challenges

- **Scale vs. generalization tradeoff:** Larger training corpora improve zero-shot performance but risk memorizing specific patterns
- **Probabilistic calibration:** Foundation model confidence intervals are not always well-calibrated across all domains
- **Multivariate limitations:** Handling cross-variable dependencies remains harder than language models handle context
- **Evaluation validity:** Benchmark datasets may leak into pre-training corpora of models trained on web data

## Further Reading

- Ansari et al. (2024), *Chronos: Learning the Language of Time Series*
- Das et al. (2024), *A Decoder-Only Foundation Model for Time-Series Forecasting (TimesFM)*
- Liu et al. (2024), *MOIRAI: Unified Training of Universal Time Series Forecasting Transformers*
- Goswami et al. (2024), *MOMENT: A Family of Open Time-series Foundation Models*
