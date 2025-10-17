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
from typing import Iterable, Union


def _ensure_series(data: Union[pd.Series, pd.DataFrame, Iterable[float]], name: str) -> pd.Series:
    """Return ``data`` as a one-dimensional :class:`pandas.Series`.

    The volatility routines historically expect simple Series inputs but in
    practice upstream callers sometimes provide single-column DataFrames (for
    instance when the OHLC data is sliced using ``df[["high"]]``).  Pandas will
    happily propagate that two-dimensional shape which later confuses
    operations such as :func:`pandas.concat` or arithmetic that assumes a
    Series.  This helper normalises the inputs so the downstream logic always
    receives a plain Series regardless of whether callers provided Series,
    NumPy arrays or one-column DataFrames.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame or iterable
        Input data that should represent a single column of values.
    name : str
        Name used when inferring the column from a DataFrame or when creating
        a Series from an array-like object.

    Returns
    -------
    pandas.Series
        A view (where possible) of the original data as a one-dimensional
        Series.

    Raises
    ------
    ValueError
        If a DataFrame with multiple columns is provided and the desired
        column cannot be determined.
    """

    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        if name in data.columns:
            return data[name]
        if data.shape[1] == 1:
            series = data.iloc[:, 0]
            if series.name is None:
                series = series.rename(name)
            return series
        raise ValueError(
            f"DataFrame input for '{name}' must contain a '{name}' column or"
            " have exactly one column"
        )
    return pd.Series(data, name=name)


def atr_percentile(
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    window: int = 14,
    lookback: int = 100,
) -> float:
    """
    Compute the percentile of the current ATR relative to its historical
    distribution.

    Parameters
    ----------
    high, low, close : pandas.Series or pandas.DataFrame
        Price series used to compute the Average True Range.  DataFrames must
        either have a single column or contain a column matching the
        respective name (``"high"``, ``"low"`` or ``"close"``).
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
    high = _ensure_series(high, "high")[-(lookback + window):].reset_index(drop=True)
    low = _ensure_series(low, "low")[-(lookback + window):].reset_index(drop=True)
    close = _ensure_series(close, "close")[-(lookback + window):].reset_index(drop=True)
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
