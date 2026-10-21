---
title: ARIMA Models for Time Series Forecasting
description: Learn how autoregressive, differencing, and moving-average components combine in ARIMA and SARIMA models, and how to identify orders and validate forecasts.
---

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for forecasting a univariate time series from its own history. It remains a strong baseline, especially for short series.

## The Components: ARIMA(p, d, q)

- **AR(p)**: the value depends on its previous p values
- **I(d)**: differencing d times to make the series stationary
- **MA(q)**: the value depends on the previous q forecast errors

```text
AR(1):  y_t = c + phi * y_{t-1} + e_t
MA(1):  y_t = c + e_t + theta * e_{t-1}
```

## Stationarity

ARIMA assumes a stationary series: stable mean and variance, and autocovariance depending only on lag. Trending series are differenced:

```text
y'_t = y_t - y_{t-1}
```

The Augmented Dickey-Fuller (ADF) test checks for a unit root, which indicates non-stationarity.

## Choosing Orders

- **ACF** (autocorrelation) cut-off suggests q
- **PACF** (partial autocorrelation) cut-off suggests p
- Or search over orders and select the lowest **AIC/BIC**

## Seasonality: SARIMA

Add seasonal terms `(P, D, Q, m)`, where m is the season length (12 for monthly data with a yearly cycle).

## Fitting and Forecasting

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = model.fit(disp=False)

forecast = res.get_forecast(steps=12)
mean = forecast.predicted_mean
ci = forecast.conf_int()
```

## Validation

Use **time-ordered** splits (rolling-origin evaluation) rather than random shuffles. Check that residuals look like white noise: no remaining autocorrelation (Ljung-Box test) and roughly constant variance.

## Strengths and Limits

ARIMA gives interpretable models and prediction intervals with little data. It handles one series at a time, assumes linear structure, and needs manual attention to outliers and structural breaks. For many related series or rich covariates, gradient boosting or deep learning forecasters are often better.
