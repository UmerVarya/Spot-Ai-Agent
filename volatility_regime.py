"""
Volatility and market regime analysis for the Spot‑AI Agent.

This module provides utilities to characterise the current market
environment.  Volatility regime filters can help adjust stop‑loss
distances and take‑profit targets dynamically, while measures like the
Hurst exponent give a sense of whether the market is trending or
mean‑reverting.

Functions
---------

* ``atr_percentile(high, low, close, window, lookback)`` – Compute the
  percentile of the current Average True Range (ATR) relative to a
  rolling lookback window.  Values above 0.75 indicate high
  volatility, while values below 0.25 suggest a quiet regime.
* ``hurst_exponent(series, max_lag)`` – Estimate the Hurst exponent of a
  price series using the rescaled range method.  A value > 0.5
  implies trending behaviour, whereas < 0.5 indicates mean
  reversion.
* ``garch_volatility(close, p, q)`` – Placeholder for a GARCH model
  volatility estimator.  Users can implement their own volatility
  models here when the ``arch`` library becomes available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def atr_percentile(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, lookback: int = 100) -> float:
    """
    Compute the percentile of the current ATR relative to its historical
    distribution.

    Parameters
    ----------
    high, low, close : pandas.Series
        Price series used to compute the Average True Range.
    window : int, optional
        ATR lookback period (default 14).
    lookback : int, optional
        Number of past ATR values to compute the percentile against (default 100).

    Returns
    -------
    float
        Fraction between 0 and 1 representing the percentile of the most
        recent ATR within the past ``lookback`` values.  NaN is returned
        if insufficient data.
    """
    if len(close) < window + lookback:
        return float('nan')
    # True range calculation
    high = high[-(lookback + window):].reset_index(drop=True)
    low = low[-(lookback + window):].reset_index(drop=True)
    close = close[-(lookback + window):].reset_index(drop=True)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().dropna()
    recent_atr = atr.iloc[-1]
    hist_atrs = atr.iloc[-lookback:]
    percentile = (hist_atrs < recent_atr).sum() / float(len(hist_atrs))
    return float(percentile)


def hurst_exponent(series: Iterable[float], max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent of a time series using the rescaled
    range (R/S) method.

    Parameters
    ----------
    series : iterable of float
        The time series to analyse (e.g. closing prices).
    max_lag : int, optional
        Maximum lag used for computing the R/S statistic.

    Returns
    -------
    float
        The estimated Hurst exponent.  Values > 0.5 suggest trending
        behaviour; values < 0.5 indicate mean reversion.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ts = np.asarray(series, dtype=float)
    if len(ts) < max_lag * 2:
        return float('nan')
    lags = range(2, max_lag)
    rs = []
    for lag in lags:
        segs = [ts[i * lag:(i + 1) * lag] for i in range(len(ts) // lag)]
        if not segs:
            continue
        rs_values = []
        for seg in segs:
            dev = seg - np.mean(seg)
            cum_dev = np.cumsum(dev)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(seg) if np.std(seg) > 0 else 1e-8
            rs_values.append(R / S)
        if rs_values:
            rs.append(np.mean(rs_values))
    if not rs:
        return float('nan')
    hurst = np.polyfit(np.log(list(lags)[:len(rs)]), np.log(rs), 1)[0]
    return float(hurst)


def garch_volatility(close: Iterable[float], p: int = 1, q: int = 1) -> float:
    """
    Placeholder for a GARCH volatility estimator.

    This function returns the rolling standard deviation as a naive
    volatility measure when the ``arch`` library is not available.  If
    ``arch`` is installed, users can replace this implementation with
    ``arch.univariate.GARCH`` for more realistic volatility estimates.
    """
    arr = np.asarray(close, dtype=float)
    if len(arr) < 30:
        return float('nan')
    returns = np.diff(np.log(arr))
    return float(np.std(returns[-30:]) * np.sqrt(252))
