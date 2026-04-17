---
title: Time Series Forecasting
description: A comprehensive guide to time series forecasting, covering classical statistical models, modern deep learning approaches, and foundation models that enable accurate prediction of sequential temporal data.
---

Time series forecasting is the task of predicting future values based on historical observations ordered in time. It underpins critical applications across finance, energy, supply chain, climate science, and healthcare. The field has evolved from classical statistical methods to transformer-based foundation models, offering a rich toolkit for diverse forecasting challenges.

## Core Concepts

A time series is a sequence of observations $x_1, x_2, \ldots, x_T$ indexed by time. Forecasting models aim to predict future values $x_{T+1}, \ldots, x_{T+H}$ where $H$ is the forecast horizon.

Key properties of time series data:

- **Trend** — long-term upward or downward movement in the data
- **Seasonality** — periodic, repeating patterns (daily, weekly, annual)
- **Cyclicity** — irregular fluctuations without a fixed period
- **Noise** — random, unpredictable variation around the signal
- **Stationarity** — statistical properties (mean, variance) remain constant over time

Most classical models require stationary series; transformations like differencing, log-scaling, or Box-Cox normalisation are applied to achieve stationarity before modelling.

## Classical Statistical Models

### ARIMA

AutoRegressive Integrated Moving Average (ARIMA) models combine three components:

- **AR(p)** — regresses on its own $p$ lagged values
- **I(d)** — applies $d$ rounds of differencing to make the series stationary
- **MA(q)** — models the error as a linear combination of $q$ lagged forecast errors

The model is written as ARIMA$(p, d, q)$. Seasonal extensions (SARIMA) add seasonal AR, differencing, and MA terms.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(2, 1, 2))
result = model.fit()
forecast = result.forecast(steps=12)
```

ARIMA works well for univariate series with clear linear patterns but struggles with non-linear dynamics or multivariate interactions.

### Exponential Smoothing

Holt-Winters Exponential Smoothing applies weighted averages that exponentially decay older observations. It supports trend and seasonal decomposition via additive or multiplicative components — practical and interpretable for business metrics.

### Prophet

Facebook's Prophet model is designed for business time series with strong seasonality and holiday effects. It decomposes the series as:

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

where $g(t)$ is the trend (piecewise linear or logistic), $s(t)$ is seasonality modelled with Fourier series, and $h(t)$ captures holiday effects. Prophet handles missing data and outliers gracefully and is easy to configure for domain experts.

```python
from prophet import Prophet

m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(df)  # df requires columns: ds (date), y (value)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
```

## Deep Learning Approaches

### Recurrent Neural Networks

LSTMs and GRUs capture temporal dependencies via hidden state that is updated at each time step. They work well for irregular-length sequences but suffer from slow training and difficulty modelling very long-range dependencies.

### N-BEATS

N-BEATS (Neural Basis Expansion Analysis for Time Series) is a pure deep learning model that uses a stack of fully connected residual blocks with backward and forward projections. It avoids recurrence and attention, achieving competitive accuracy with interpretable basis expansions that decompose forecasts into trend and seasonal components.

### Temporal Fusion Transformer (TFT)

TFT combines multi-horizon forecasting with interpretability. It processes:

- **Static covariates** (e.g., store ID, geography) through variable selection networks
- **Known future inputs** (e.g., promotions, holidays) via gated residual networks
- **Observed past inputs** via LSTM encoder followed by multi-head attention

TFT outputs quantile forecasts, enabling uncertainty estimation alongside point predictions.

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

training = TimeSeriesDataSet(data, time_idx="time_idx", target="sales", ...)
tft = TemporalFusionTransformer.from_dataset(training, learning_rate=0.03)
trainer.fit(tft, train_dataloaders=train_dataloader)
```

## Foundation Models for Time Series

Recent research has produced large pretrained models for time series that generalise across domains via zero-shot or few-shot forecasting.

| Model | Architecture | Approach |
| --- | --- | --- |
| TimesFM | Transformer decoder | Pretrained on 100B real-world time points |
| Chronos | T5-based language model | Quantised time series as token sequences |
| MOIRAI | Unified transformer | Multi-task pretraining with any-variate masking |
| Lag-Llama | Llama decoder | Univariate probabilistic forecasting |

These models are particularly valuable when limited historical data is available for a new domain or when adapting a single model across hundreds of SKUs or sensors.

## Evaluation Metrics

| Metric | Formula | Notes |
| --- | --- | --- |
| MAE | $\frac{1}{H}\sum \lvert y_t - \hat{y}_t \rvert$ | Robust to outliers |
| RMSE | $\sqrt{\frac{1}{H}\sum (y_t - \hat{y}_t)^2}$ | Penalises large errors |
| MAPE | $\frac{100}{H}\sum \frac{\lvert y_t - \hat{y}_t \rvert}{y_t}$ | Undefined when $y_t = 0$ |
| MASE | MAE / MAE of naïve seasonal baseline | Scale-independent |
| CRPS | Probabilistic accuracy | For distributional forecasts |

Use multiple metrics and always compare against a naïve seasonal baseline before claiming model superiority.

## Choosing the Right Approach

| Scenario | Recommended Approach |
| --- | --- |
| Univariate, clear seasonality | Prophet or SARIMA |
| Many related series (retail, energy) | TFT or N-BEATS |
| Minimal data, quick deployment | TimesFM / Chronos zero-shot |
| Real-time streaming | Online learning or lightweight LSTM |
| Interpretability required | Prophet or TFT with attention weights |

## Best Practices

- **Backtest rigorously** using walk-forward (expanding or rolling window) cross-validation rather than a single train/test split.
- **Normalise per series** — especially in multi-series models — to prevent high-magnitude series from dominating gradients.
- **Encode time features** explicitly (hour of day, day of week, week of year) as additional inputs rather than relying on the model to learn periodicity from raw timestamps.
- **Handle leakage carefully** — ensure no future information leaks into the feature set during training windows.
- **Quantify uncertainty** — point forecasts alone are insufficient for decision-making; use quantile regression or conformal prediction to produce calibrated prediction intervals.
