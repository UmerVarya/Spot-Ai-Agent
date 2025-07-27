"""
Chart pattern detection utilities for the Spot‑AI Agent.

This module implements heuristic detectors for common technical chart
patterns such as **triangles/wedges**, **flags/pennants** and
**head‑and‑shoulders** formations.  These detectors are not meant to
be foolproof – accurately identifying patterns is an open research
problem – but they provide a reasonable approximation without relying
on heavy external libraries.  They operate on OHLCV DataFrames and
return boolean flags indicating whether a pattern appears in the most
recent segment of the data.

Patterns detected
-----------------

* **Triangle/Wedge** – Identified by converging trendlines on the highs
  and lows.  A simple linear regression is fit to the recent highs and
  lows; if the slopes have opposite signs and the angle between them
  is small, the pattern is flagged.
* **Flag/Pennant** – Characterised by a sharp price movement followed by
  a brief, shallow retracement or sideways consolidation.  The
  algorithm checks for a recent trend exceeding a percentage threshold
  followed by a consolidation where the price range remains tight.
* **Head‑and‑Shoulders** – Consists of three peaks with the middle peak
  (head) higher than the shoulders.  Local maxima are identified and
  compared in height and spacing.

These functions require only pandas and numpy; if ``scipy`` is
available, a more sophisticated peak detection may be used.
"""

import numpy as np
import pandas as pd
from typing import Tuple

try:
    from scipy.signal import argrelextrema  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def _linear_slope(y: np.ndarray) -> float:
    """Compute the slope of a simple linear regression line through y."""
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def detect_triangle_wedge(df: pd.DataFrame, lookback: int = 40) -> bool:
    """
    Detect a triangle or wedge pattern in the most recent segment of data.

    The function fits linear regressions to the highs and lows over the
    ``lookback`` period and checks whether the lines converge (slopes of
    opposite sign and the difference between their absolute values is
    small).  If the distance between the lines narrows toward the most
    recent candles, the pattern is considered present.
    """
    if df is None or len(df) < lookback:
        return False
    segment = df[-lookback:]
    highs = segment['high'].to_numpy()
    lows = segment['low'].to_numpy()
    slope_high = _linear_slope(highs)
    slope_low = _linear_slope(lows)
    # Opposite signs indicate convergence
    if slope_high * slope_low >= 0:
        return False
    # Slopes should not be drastically different in magnitude
    if abs(abs(slope_high) - abs(slope_low)) > 0.001 * np.mean([abs(slope_high), abs(slope_low)] + [1e-9]):
        return False
    # Distance between lines should shrink over time
    x = np.arange(lookback)
    line_high = slope_high * x + highs[0]
    line_low = slope_low * x + lows[0]
    distances = line_high - line_low
    return bool(distances[-1] < distances[0])


def detect_flag_pattern(df: pd.DataFrame, trend_lookback: int = 30, flag_lookback: int = 10, trend_threshold: float = 0.05) -> bool:
    """
    Detect a flag or pennant pattern.

    Parameters
    ----------
    df : pandas.DataFrame
        Price DataFrame containing a 'close' column.
    trend_lookback : int
        Number of bars to analyse for the preceding trend.
    flag_lookback : int
        Number of bars for the consolidation area.
    trend_threshold : float
        Minimum fractional price change required to consider the initial
        move strong enough to form a flag (e.g., 0.05 = 5 %).
    """
    if df is None or len(df) < trend_lookback + flag_lookback + 1:
        return False
    prices = df['close'].to_numpy()
    # Measure preceding trend
    pre_move = (prices[-flag_lookback - 1] - prices[-flag_lookback - trend_lookback]) / prices[-flag_lookback - trend_lookback]
    if abs(pre_move) < trend_threshold:
        return False
    # Check consolidation: price stays within a small range
    flag_section = prices[-flag_lookback:]
    max_flag = flag_section.max()
    min_flag = flag_section.min()
    range_flag = max_flag - min_flag
    # Range should be small relative to preceding move
    return bool(range_flag / abs(pre_move * prices[-flag_lookback - trend_lookback]) < 0.25)


def detect_head_and_shoulders(df: pd.DataFrame, lookback: int = 60, distance: int = 5, threshold: float = 0.02) -> bool:
    """
    Detect a head‑and‑shoulders pattern over the specified lookback.

    The algorithm looks for three local maxima with the middle peak
    exceeding its neighbours by at least ``threshold`` (fraction of price).
    It also ensures that the shoulders are roughly at the same level.
    """
    if df is None or len(df) < lookback:
        return False
    closes = df['close'].to_numpy()[-lookback:]
    if SCIPY_AVAILABLE:
        idx_max = argrelextrema(closes, np.greater)[0]
    else:
        # Fallback: simple maxima detection
        idx_max = [i for i in range(1, len(closes) - 1) if closes[i] > closes[i - 1] and closes[i] > closes[i + 1]]
    if len(idx_max) < 3:
        return False
    # Consider combinations of three peaks
    for i in range(len(idx_max) - 2):
        i1, i2, i3 = idx_max[i], idx_max[i + 1], idx_max[i + 2]
        # Roughly equally spaced
        if not (distance <= (i2 - i1) <= 2 * distance and distance <= (i3 - i2) <= 2 * distance):
            continue
        p1, p2, p3 = closes[i1], closes[i2], closes[i3]
        # Middle peak should be the highest
        if not (p2 > p1 and p2 > p3):
            continue
        # Shoulders roughly equal
        if abs(p1 - p3) / p2 > threshold:
            continue
        # Head height relative to shoulders
        if (p2 - (p1 + p3) / 2) / ((p1 + p3) / 2) < threshold:
            continue
        return True
    return False
