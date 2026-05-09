---
title: Foundation Models for Time Series
description: Explore the emerging class of large pretrained models for time series — including Chronos, TimesFM, MOIRAI, and Lag-Llama — that achieve zero-shot and few-shot forecasting across diverse domains without task-specific training.
---

Foundation models for time series represent a major shift in how sequential prediction is approached. Traditionally, time series forecasting required training a separate model for each dataset — ARIMA for one domain, XGBoost for another, an LSTM for a third. **Time series foundation models** are large pretrained models trained on diverse collections of temporal data that can forecast new, unseen time series **at zero shot** — without any fine-tuning on the target dataset.

This paradigm mirrors the revolution that large language models brought to NLP: instead of task-specific models, a single general-purpose model covers a wide distribution of tasks.

## What Makes Time Series Unique

Time series pose distinct challenges compared to language or images:

- **Heterogeneous scale:** A temperature sensor and a stock price occupy completely different numerical ranges, requiring normalization strategies.
- **Variable frequency:** Data can be collected at sub-second, hourly, daily, or monthly intervals with radically different autocorrelation structures.
- **Missing values and irregular sampling:** Real-world series have gaps, outliers, and non-uniform sampling.
- **No universal tokenization:** Language has a fixed vocabulary; continuous-valued time series have no natural discrete token space.
- **Domain-specific patterns:** Seasonality (annual, weekly, daily), trends, and cycles are domain-specific and must be captured generically.

## Chronos (Amazon, 2024)

**Chronos** is a family of pretrained language models for probabilistic time series forecasting. Its key innovation is treating time series forecasting as a **language modeling problem** by quantizing continuous values into discrete tokens.

### Architecture

1. **Normalization:** Scale the input series by its mean absolute value to remove scale differences across domains.
2. **Quantization:** Map normalized values to $B$ discrete bins (e.g., $B = 4096$) using a fixed vocabulary of "time series tokens."
3. **Autoregressive modeling:** Train a T5-style encoder-decoder (or a GPT-style decoder-only) model to predict the next token in the quantized sequence.
4. **Probabilistic output:** Sample multiple token sequences from the model's distribution; inverse-quantize to obtain a forecast distribution.

### Training Data

Chronos is trained on **large and diverse corpora** spanning electricity, retail sales, web traffic, weather, econometrics, and synthetic series generated from ARIMA, ETS, and GP processes. Synthetic data augments real data to cover rare distributional shapes.

### Performance

On the **GIFT-Eval** and **Monash** benchmark suites, Chronos-Large (710M parameters) achieves state-of-the-art zero-shot performance, matching or exceeding models that were trained specifically on the target domain — despite seeing none of the target data during forecasting.

## TimesFM (Google DeepMind, 2024)

**TimesFM** (Time Series Foundation Model) takes a different approach: rather than quantizing to tokens, it uses a **patched decoder-only Transformer** operating on continuous time series values.

### Patched Time Series Modeling

The series is divided into non-overlapping **patches** of length $P$ (e.g., $P = 32$). Each patch is projected into a $d$-dimensional embedding by a linear layer — analogous to how vision Transformers treat image patches. The model autoregressively predicts the next patch given all previous patches:

$$\hat{x}_{t:t+P} = f_\theta(x_{t-kP:t})$$

This allows the model to forecast at **multiple horizons** with a single forward pass by outputting multiple future patches.

### Key Design Choices

- **Input patching:** Reduces the effective sequence length from $L$ to $L/P$, enabling long-context training.
- **Frequency agnosticism:** No explicit frequency encoding — the model learns temporal patterns from data diversity.
- **Decoder-only:** Enables flexible horizon forecasting at inference time by varying how many future patches to decode.

TimesFM is trained on 100 billion time series points from Google's internal data repositories plus public datasets, achieving strong zero-shot results on M4, ETT, and the GIFT-Eval benchmarks.

## MOIRAI (Salesforce, 2024)

**MOIRAI** (Unified Training of Universal Time Series Forecasting Transformers) targets the challenge of **multi-frequency unification**: how to train a single model that works across series with radically different sampling frequencies (seconds to years).

### Universal Patch Projection

MOIRAI uses a **mixture of patch sizes** — multiple projection layers for different patch lengths — combined via a learned mixture coefficient. This allows the model to process short-horizon, high-frequency series and long-horizon, low-frequency series with the same architecture.

### Variate Attention

For **multivariate** time series (multiple correlated channels), MOIRAI uses cross-variate attention to capture inter-channel dependencies — important for datasets like electricity (many meters with correlated consumption) or financial markets (assets that co-move).

## Lag-Llama (2024)

**Lag-Llama** adapts the Llama transformer architecture to probabilistic time series forecasting by:

1. **Lag features:** Appending lagged values at multiple fixed offsets (e.g., $x_{t-1}, x_{t-7}, x_{t-30}$) to each input token, providing explicit multi-scale temporal context.
2. **Student-T output head:** Predicting parameters of a Student-t distribution for probabilistic outputs robust to outliers.
3. **Decoder-only architecture:** Generates forecasts autoregressively, one step at a time.

Lag-Llama is fully open-source and demonstrates strong zero-shot forecasting on the Monash benchmark, particularly on datasets with regular periodic structure.

## Comparison of Architectures

| Model | Architecture | Tokenization | Probabilistic | Open Source |
| --- | --- | --- | --- | --- |
| Chronos | T5 / GPT-2 variants | Quantized tokens | Yes (sampling) | Yes |
| TimesFM | Patched decoder | Continuous patches | No (point + PI) | Partial |
| MOIRAI | Universal Transformer | Multi-scale patches | Yes (mixture) | Yes |
| Lag-Llama | Llama decoder | Lag features | Yes (Student-T) | Yes |
| Timer | GPT-style decoder | Patch embeddings | No | Yes |

## Zero-Shot Forecasting Performance

Benchmarking time series foundation models is nuanced. Key evaluation protocols:

- **Zero-shot:** Model receives only the context series; no examples from the target dataset.
- **Few-shot fine-tuning:** Model is fine-tuned on a small sample of the target dataset.
- **Full fine-tuning:** Model is trained to convergence on the target dataset.

On average across the Monash and GIFT-Eval suites, foundation models at zero shot outperform:

- Classical methods (ETS, ARIMA) by 10–25%
- Dataset-specific DL models (N-BEATS, Informer) by 5–15%
- They are roughly matched by strong gradient-boosting baselines (LightGBM with careful feature engineering) on structured tabular-style series.

The **few-shot advantage** is most pronounced on small datasets (< 500 observations) where dataset-specific training is impossible.

## When to Use Time Series Foundation Models

Foundation models provide the greatest value when:

- **New dataset, no historical model:** Zero-shot forecasting avoids cold-start delays.
- **Low-data regimes:** Fine-tuning on as few as 32 examples provides meaningful signal.
- **Diverse domains in a single pipeline:** A unified model across many business units or sensor types reduces operational overhead.
- **Rapid prototyping:** Getting a strong baseline in minutes rather than days of feature engineering.

They are less advantageous when:

- The target series is very long (> millions of points) with stable, well-characterized patterns.
- The domain is highly specialized (e.g., high-frequency trading, ECG signals) with no overlap with pretraining data.
- Ultra-low latency inference is required (large models add latency vs. simple ARIMA).

## Pre-Training Data and the Coverage Problem

A persistent challenge is **domain coverage**: a model pretrained on web traffic and electricity may not zero-shot well on seismology or industrial sensor data. The field is actively working on:

- **Diverse synthetic generation:** GPT-like corpora of synthetic time series from a wide distribution of stochastic processes.
- **Domain-adaptive fine-tuning:** Efficient adaptation to narrow domains with minimal compute.
- **Data mixture recipes:** Understanding which source domains transfer best to target domains, analogous to data mixture research in LLM pretraining.

## The Future: Time Series as a Modality

Longer-term, time series foundation models are likely to become a modality input to **multimodal foundation models** — enabling joint reasoning over text instructions, image context, and temporal data streams. Use cases include:

- **Medical:** Joint LLM reasoning over clinical notes and vital-sign time series.
- **Finance:** LLM analysis of news combined with time series market context.
- **Industrial IoT:** Natural language query answering over sensor histories.

## Summary

Foundation models for time series bring the pretrain-then-adapt paradigm to sequential prediction, achieving zero-shot and few-shot forecasting across diverse domains with a single unified model. Architectures like Chronos, TimesFM, MOIRAI, and Lag-Llama each offer different trade-offs between probabilistic expressiveness, frequency generalization, and computational cost. Their emergence signals a shift away from bespoke, per-dataset forecasting pipelines toward general temporal reasoning systems that improve with scale — following the same trajectory established by language and vision foundation models.
