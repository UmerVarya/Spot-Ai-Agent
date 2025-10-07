"""Market auction state classification utilities.

This module combines volatility and trend diagnostics to infer whether a
market is currently balanced or in an out-of-balance auction.  The
classification can help the agent decide whether to favour breakout or
mean-reversion setups.
"""

from __future__ import annotations

import math

import pandas as pd

from volatility_regime import atr_percentile, hurst_exponent

__all__ = ["get_auction_state"]


def get_auction_state(
    df: pd.DataFrame,
    atr_window: int = 14,
    atr_lookback: int = 100,
    hurst_max_lag: int = 100,
    atr_threshold: float = 0.6,
    trend_hurst_threshold: float = 0.55,
    revert_hurst_threshold: float = 0.45,
) -> str:
    """Classify the current auction state for a market symbol.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least ``high``, ``low`` and ``close``
        columns.  The data should be ordered chronologically.
    atr_window : int, optional
        Lookback used when computing the Average True Range (default 14).
    atr_lookback : int, optional
        Number of historical ATR values used to determine the percentile
        (default 100).
    hurst_max_lag : int, optional
        Maximum lag for the Hurst exponent estimate (default 100).
    atr_threshold : float, optional
        Percentile threshold signalling an out-of-balance volatility
        regime (default 0.6).
    trend_hurst_threshold : float, optional
        Threshold above which the market is considered trending (default
        0.55).
    revert_hurst_threshold : float, optional
        Threshold below which the market is considered mean-reverting
        (default 0.45).

    Returns
    -------
    str
        One of ``"out_of_balance_trend"``,
        ``"out_of_balance_revert"``, ``"balanced"``, or ``"unknown"``
        when insufficient data prevents a reliable classification.

    Raises
    ------
    KeyError
        If the required OHLC columns are missing from ``df``.
    """

    required_columns = {"high", "low", "close"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise KeyError(f"DataFrame is missing required columns: {missing_str}")

    atr_p = atr_percentile(
        df["high"],
        df["low"],
        df["close"],
        window=atr_window,
        lookback=atr_lookback,
    )
    hurst = hurst_exponent(df["close"], max_lag=hurst_max_lag)

    if math.isnan(atr_p) or math.isnan(hurst):
        return "unknown"

    if atr_p >= atr_threshold and hurst >= trend_hurst_threshold:
        return "out_of_balance_trend"
    if atr_p >= atr_threshold and hurst <= revert_hurst_threshold:
        return "out_of_balance_revert"
    return "balanced"

