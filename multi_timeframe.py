"""
Multi‑timeframe analysis utilities for the Spot‑AI Agent.

Trading strategies often benefit from confirming signals across
multiple timeframes.  This module provides simple helpers to
resample OHLCV data into higher intervals and to aggregate indicator
values across those intervals.  It can be used to ensure that a
short‑term signal aligns with the trend on a longer timeframe,
or to apply higher‑timeframe filters (e.g. requiring the daily trend
to agree with an intraday setup).
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd
from typing import Callable, Dict, List


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample a lower‑frequency OHLCV DataFrame to a higher timeframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``['open','high','low','close','volume']`` and a
        datetime index.
    timeframe : str
        Pandas offset alias (e.g. ``'5T'`` for 5 minutes, ``'15T'``, ``'1H'``).

    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame with OHLCV columns. Bars are labelled by their
        closing timestamp and only include intervals that have fully closed.
    """
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    return df.resample(timeframe, label="right", closed="right").apply(ohlc).dropna()


def multi_timeframe_confluence(df: pd.DataFrame, timeframes: List[str], indicator_func: Callable[[pd.Series], float]) -> Dict[str, float]:
    """
    Compute a simple indicator across multiple timeframes.

    ``indicator_func`` should accept a Series of closing prices and
    return a numeric indicator (e.g. moving average slope, RSI value).
    The function returns a mapping from timeframe to the indicator
    value.  Users can then decide whether short‑term signals align
    with longer‑term indicators.

    Example::

        from multi_timeframe import multi_timeframe_confluence
        import numpy as np
        def slope(series):
            x = np.arange(len(series))
            m, _ = np.polyfit(x, series, 1)
            return m
        slopes = multi_timeframe_confluence(df, ['5T','15T','1H'], slope)
        # Check if all slopes are positive
        if all(v > 0 for v in slopes.values()):
            ...
    """
    results: Dict[str, float] = {}
    for tf in timeframes:
        # Resample the full OHLCV dataframe so all required
        # columns are present for the aggregation. Previously this
        # function attempted to resample only the ``close`` column,
        # which raised ``KeyError`` when ``resample_ohlcv`` expected
        # the other OHLCV columns.  Passing the entire dataframe
        # ensures the resampler has the correct inputs.
        resampled = resample_ohlcv(df.copy(), tf)
        now = datetime.now(resampled.index.tz) if resampled.index.tz else datetime.now()
        resampled = resampled[resampled.index <= now]
        if len(resampled) < 2:
            continue
        results[tf] = float(indicator_func(resampled['close']))
    return results


def multi_timeframe_indicator_alignment(
    df: pd.DataFrame,
    timeframes: List[str],
    indicator_funcs: Dict[str, Callable[[pd.DataFrame], float]],
) -> Dict[str, Dict[str, float]]:
    """Compute multiple indicators across several timeframes.

    This helper allows strategies to confirm that the *same* indicators
    agree on different charts.  ``indicator_funcs`` is a mapping of
    indicator name to a callable that accepts an OHLCV DataFrame and
    returns a numeric value (e.g. RSI value, moving‑average difference).

    The return value is a nested dictionary of the form::

        {
            '5T': {'rsi': 55.0, 'ema_trend': 0.1},
            '1H': {'rsi': 48.0, 'ema_trend': 0.05},
        }

    Parameters
    ----------
    df : pandas.DataFrame
        Source OHLCV data with a datetime index.
    timeframes : List[str]
        Timeframes to resample to (e.g. ``['5T', '1H']``).
    indicator_funcs : Dict[str, Callable[[pandas.DataFrame], float]]
        Mapping of indicator names to callables.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping of timeframe to indicator/value pairs.
    """
    results: Dict[str, Dict[str, float]] = {}
    for tf in timeframes:
        resampled = resample_ohlcv(df.copy(), tf)
        now = datetime.now(resampled.index.tz) if resampled.index.tz else datetime.now()
        resampled = resampled[resampled.index <= now]
        if len(resampled) < 2:
            continue
        tf_vals: Dict[str, float] = {}
        for name, func in indicator_funcs.items():
            tf_vals[name] = float(func(resampled))
        results[tf] = tf_vals
    return results
