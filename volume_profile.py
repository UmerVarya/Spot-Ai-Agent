"""Utilities for computing lightweight volume-by-price profiles from OHLCV data.

This module approximates a traditional volume profile (a histogram of traded
volume by price) using aggregated OHLCV candles.  When full tick data is not
available – or would be too expensive to download on-demand – the helper
functions below bin candle prices into fixed percentage increments and sum the
associated volumes.  The resulting histogram can be used to identify price
levels such as the Point of Control (POC) and Low-Volume Nodes (LVNs) that
often act as support/resistance when auction market theory dynamics are in
play.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math

import numpy as np
import pandas as pd

__all__ = [
    "VolumeProfileResult",
    "compute_volume_profile",
]


@dataclass(frozen=True)
class VolumeProfileResult:
    """Container for the derived volume profile statistics."""

    poc: Optional[float]
    lvns: List[float]
    histogram: pd.DataFrame
    bin_size: float
    price_min: float
    price_max: float

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of the volume profile."""

        return {
            "poc": float(self.poc) if self.poc is not None and math.isfinite(self.poc) else None,
            "lvns": [float(x) for x in self.lvns if math.isfinite(x)],
            "bin_size": float(self.bin_size),
            "price_min": float(self.price_min),
            "price_max": float(self.price_max),
        }


def _validate_inputs(df: pd.DataFrame, price_col: str, volume_col: str) -> bool:
    if df is None or df.empty:
        return False
    required = {price_col, volume_col}
    if not required.issubset(df.columns):
        return False
    return True


def _infer_bin_size(prices: pd.Series, default_pct: float) -> float:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if prices.empty:
        return default_pct
    median_price = float(prices.median())
    if not math.isfinite(median_price) or median_price <= 0:
        return default_pct
    return max(median_price * default_pct, median_price * 1e-5)


def _build_histogram(
    prices: np.ndarray,
    volumes: np.ndarray,
    bin_size: float,
    price_min: float,
    price_max: float,
) -> pd.DataFrame:
    if price_max <= price_min:
        price_max = price_min + bin_size
    num_bins = max(int(math.ceil((price_max - price_min) / bin_size)), 1)
    edges = price_min + np.arange(num_bins + 1) * bin_size
    indices = np.clip(((prices - price_min) / bin_size).astype(int), 0, num_bins - 1)
    volume_totals = np.bincount(indices, weights=volumes, minlength=num_bins)
    bin_start = edges[:-1]
    bin_end = edges[1:]
    bin_mid = (bin_start + bin_end) / 2.0
    histogram = pd.DataFrame(
        {
            "bin_start": bin_start,
            "bin_end": bin_end,
            "price": bin_mid,
            "volume": volume_totals,
        }
    )
    return histogram


def _find_lvns(histogram: pd.DataFrame, lv_threshold: float, hv_threshold: float) -> List[float]:
    lvns: List[float] = []
    if histogram.empty:
        return lvns
    volumes = histogram["volume"].to_numpy()
    if volumes.size == 0:
        return lvns
    max_volume = float(volumes.max())
    if max_volume <= 0:
        return lvns
    low_cutoff = max_volume * lv_threshold
    high_cutoff = max_volume * hv_threshold
    prices = histogram["price"].to_numpy(dtype=float)
    for idx, volume in enumerate(volumes):
        volume = float(volume)
        if volume <= 0 or volume > low_cutoff:
            continue
        prev_idx = idx - 1
        while prev_idx >= 0 and volumes[prev_idx] <= 0:
            prev_idx -= 1
        next_idx = idx + 1
        while next_idx < len(volumes) and volumes[next_idx] <= 0:
            next_idx += 1
        prev_volume = float(volumes[prev_idx]) if prev_idx >= 0 else max_volume
        next_volume = float(volumes[next_idx]) if next_idx < len(volumes) else max_volume
        if (prev_idx >= 0 and prev_volume >= high_cutoff) or (
            next_idx < len(volumes) and next_volume >= high_cutoff
        ):
            lvns.append(float(prices[idx]))
    return sorted(lvns)


def compute_volume_profile(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    volume_col: str = "volume",
    bin_pct: float = 0.001,
    lv_threshold: float = 0.2,
    hv_threshold: float = 0.5,
) -> VolumeProfileResult:
    """Build a lightweight volume profile from candle data.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV candle data ordered chronologically.
    price_col : str, optional
        Column containing representative prices (default ``"close"``).
    volume_col : str, optional
        Column containing traded volume (default ``"volume"``).
    bin_pct : float, optional
        Relative bin width expressed as a fraction of price (default ``0.001``
        ≈ 0.1%).  The actual bin size is inferred from the median price.
    lv_threshold : float, optional
        Fraction of the maximum volume used to flag low-volume nodes.
    hv_threshold : float, optional
        Fraction of the maximum volume that neighbouring bins must exceed in
        order for a low-volume bin to qualify as an LVN.

    Returns
    -------
    VolumeProfileResult
        Contains the POC, LVNs and the histogram table.
    """

    if not _validate_inputs(df, price_col, volume_col):
        return VolumeProfileResult(
            poc=None,
            lvns=[],
            histogram=pd.DataFrame(columns=["bin_start", "bin_end", "price", "volume"]),
            bin_size=float("nan"),
            price_min=float("nan"),
            price_max=float("nan"),
        )

    price_series = pd.to_numeric(df[price_col], errors="coerce")
    volume_series = pd.to_numeric(df[volume_col], errors="coerce")
    prices = price_series.to_numpy(dtype=float)
    volumes = volume_series.to_numpy(dtype=float)
    mask = np.isfinite(prices) & np.isfinite(volumes)
    prices = prices[mask]
    volumes = volumes[mask]
    if prices.size == 0 or volumes.size == 0:
        return VolumeProfileResult(
            poc=None,
            lvns=[],
            histogram=pd.DataFrame(columns=["bin_start", "bin_end", "price", "volume"]),
            bin_size=float("nan"),
            price_min=float("nan"),
            price_max=float("nan"),
        )

    price_min = float(np.min(prices))
    price_max = float(np.max(prices))
    inferred_bin = _infer_bin_size(price_series, default_pct=bin_pct)
    histogram = _build_histogram(prices, volumes, inferred_bin, price_min, price_max)
    poc_price: Optional[float] = None
    if not histogram.empty:
        poc_idx = int(histogram["volume"].idxmax())
        poc_price = float(histogram.loc[poc_idx, "price"])
    lvns = _find_lvns(histogram, lv_threshold=lv_threshold, hv_threshold=hv_threshold)
    return VolumeProfileResult(
        poc=poc_price,
        lvns=lvns,
        histogram=histogram,
        bin_size=float(inferred_bin),
        price_min=price_min,
        price_max=price_max,
    )
