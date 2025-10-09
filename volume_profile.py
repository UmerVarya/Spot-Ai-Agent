"""Volume profile analytics for impulse and reclaim legs.

This module provides lightweight volume-by-price calculations using
minute-level OHLCV data.  It avoids relying on exchange-specific tick
downloads while still surfacing price levels – such as the point of
control (POC) and low-volume nodes (LVNs) – that matter for execution.
The primary entry points are :func:`compute_trend_leg_volume_profile`
for breakout legs and :func:`compute_reversion_leg_volume_profile` for
failed-breakdown reclaims.

The implementation intentionally steers clear of pandas' internal
``_infer_bin_size`` helper by constructing histogram bins manually.  The
agent can therefore request profiles on demand without triggering
warnings about ambiguous bin sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "VolumeProfileResult",
    "compute_volume_profile",
    "compute_trend_leg_volume_profile",
    "compute_reversion_leg_volume_profile",
]


@dataclass
class VolumeProfileResult:
    """Container describing a computed volume profile."""

    poc: float
    lvns: List[float]
    bins: pd.DataFrame
    bin_width: float
    leg_start_index: int
    leg_end_index: int
    leg_type: str = "generic"
    metadata: Dict[str, float] = field(default_factory=dict)

    def touched_lvn(
        self,
        *,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        tolerance: float = 0.0015,
    ) -> Optional[float]:
        """Return the LVN that price is currently interacting with.

        Parameters
        ----------
        close : float
            The most recent closing price.
        high, low : float, optional
            High/low of the most recent candle.  If supplied the method
            will also consider intra-candle touches instead of relying
            solely on the close.
        tolerance : float, optional
            Allowed percentage distance (0.15 % by default) when
            comparing the close to an LVN.
        """

        if not self.lvns:
            return None
        try:
            close_val = float(close)
        except (TypeError, ValueError):
            return None
        if close_val <= 0:
            return None
        high_val = None
        low_val = None
        try:
            if high is not None:
                high_val = float(high)
        except (TypeError, ValueError):
            high_val = None
        try:
            if low is not None:
                low_val = float(low)
        except (TypeError, ValueError):
            low_val = None

        for level in self.lvns:
            if not math.isfinite(level) or level <= 0:
                continue
            if low_val is not None and high_val is not None:
                if low_val <= level <= high_val:
                    return level
            distance = abs(close_val - level) / close_val
            if distance <= tolerance:
                return level
        return None

    def to_dict(self, max_levels: int = 6) -> Dict[str, object]:
        """Return a JSON-serialisable summary of the profile."""

        snapshot_bins = self.bins.copy()
        snapshot_bins["price"] = snapshot_bins["price"].round(6)
        snapshot_bins["volume"] = snapshot_bins["volume"].round(6)
        if len(snapshot_bins) > 60:
            snapshot_bins = snapshot_bins.iloc[-60:]
        return {
            "poc": float(round(self.poc, 6)) if math.isfinite(self.poc) else None,
            "lvns": [float(round(l, 6)) for l in self.lvns[:max_levels]],
            "bin_width": float(round(self.bin_width, 6)),
            "leg_start_index": int(self.leg_start_index),
            "leg_end_index": int(self.leg_end_index),
            "leg_type": self.leg_type,
            "metadata": {k: float(round(v, 6)) for k, v in self.metadata.items() if math.isfinite(v)},
            "bins": snapshot_bins.to_dict(orient="records"),
        }


def _safe_series(values: Union[pd.Series, Sequence[float], np.ndarray, pd.Index]) -> pd.Series:
    """Return a float series while tolerating numpy array input."""

    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.astype(float)

    if isinstance(values, pd.Index):
        base_series = values.to_series(index=values)
    elif isinstance(values, np.ndarray):
        base_series = pd.Series(values)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        base_series = pd.Series(values)
    else:
        base_series = pd.Series([values])

    numeric = pd.to_numeric(base_series, errors="coerce")
    if isinstance(numeric, pd.Series):
        return numeric.astype(float)

    # ``pd.to_numeric`` can return an ndarray for array-like input; wrap it back
    # into a Series so callers always receive an object supporting ``dropna``.
    return pd.Series(numeric, index=base_series.index, dtype=float)


def _determine_bin_configuration(
    min_price: float,
    max_price: float,
    *,
    bin_size_pct: float,
    min_bins: int,
) -> tuple[np.ndarray, float]:
    price_span = max_price - min_price
    if price_span <= 0 or not math.isfinite(price_span):
        padded_span = max(abs(min_price), 1.0) * bin_size_pct
        padded_span = max(padded_span, 1e-6)
        edges = np.array([min_price - padded_span, max_price + padded_span], dtype=float)
        return edges, max(padded_span, 1e-6)

    reference_price = max((min_price + max_price) / 2.0, 1e-6)
    approx_width = max(reference_price * bin_size_pct, price_span / max(min_bins, 1))
    approx_width = max(approx_width, 1e-6)
    bin_count = max(int(math.ceil(price_span / approx_width)), min_bins)
    bin_count = min(bin_count, 500)
    bin_width = max(price_span / max(bin_count, 1), 1e-6)
    edges = min_price + np.arange(bin_count + 1, dtype=float) * bin_width
    edges[-1] = max_price + bin_width
    return edges, bin_width


def compute_volume_profile(
    df: pd.DataFrame,
    *,
    bin_size_pct: float = 0.001,
    min_bins: int = 12,
) -> Optional[VolumeProfileResult]:
    """Calculate a price/volume histogram for ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Minute-level OHLCV data ordered chronologically.
    bin_size_pct : float, optional
        Target bin width as a percentage of the mid-price (default 0.1 %).
    min_bins : int, optional
        Minimum number of bins to allocate.
    """

    if df is None or df.empty:
        return None
    required_columns = {"high", "low", "close", "volume"}
    if not required_columns.issubset(df.columns):
        return None

    highs = _safe_series(df["high"])
    lows = _safe_series(df["low"])
    typical = ((highs + lows + _safe_series(df["close"])) / 3.0).to_numpy(dtype=float)
    volumes = _safe_series(df["volume"]).to_numpy(dtype=float)

    finite_mask = np.isfinite(typical) & np.isfinite(volumes)
    if finite_mask.sum() == 0:
        return None
    typical = typical[finite_mask]
    volumes = volumes[finite_mask]

    min_price = float(np.min(typical))
    max_price = float(np.max(typical))
    edges, bin_width = _determine_bin_configuration(
        min_price,
        max_price,
        bin_size_pct=bin_size_pct,
        min_bins=min_bins,
    )
    bin_indices = np.clip(np.digitize(typical, edges, right=False) - 1, 0, len(edges) - 2)
    volume_bins = np.zeros(len(edges) - 1, dtype=float)
    for idx, vol in zip(bin_indices, volumes):
        if math.isfinite(vol):
            volume_bins[idx] += max(vol, 0.0)

    centers = edges[:-1] + bin_width / 2.0
    bins_df = pd.DataFrame({"price": centers, "volume": volume_bins})
    if bins_df["volume"].sum() == 0:
        return None

    poc_idx = int(np.argmax(volume_bins))
    poc_price = float(centers[poc_idx])
    poc_volume = float(volume_bins[poc_idx])
    low_volume_threshold = poc_volume * 0.2
    high_volume_threshold = poc_volume * 0.4
    lvns: List[float] = []
    for i, vol in enumerate(volume_bins):
        if not math.isfinite(vol):
            continue
        if vol <= 0:
            lvns.append(float(centers[i]))
            continue
        if vol <= low_volume_threshold:
            prev_high = volume_bins[i - 1] if i > 0 else 0.0
            next_high = volume_bins[i + 1] if i + 1 < len(volume_bins) else 0.0
            if prev_high >= high_volume_threshold or next_high >= high_volume_threshold:
                lvns.append(float(centers[i]))

    lvns.sort()
    deduped_lvns: List[float] = []
    for level in lvns:
        if not deduped_lvns:
            deduped_lvns.append(level)
            continue
        if abs(level - deduped_lvns[-1]) >= bin_width * 0.5:
            deduped_lvns.append(level)

    return VolumeProfileResult(
        poc=poc_price,
        lvns=deduped_lvns,
        bins=bins_df,
        bin_width=float(bin_width),
        leg_start_index=0,
        leg_end_index=len(df) - 1,
        metadata={
            "leg_low": float(np.min(lows.to_numpy(dtype=float))),
            "leg_high": float(np.max(highs.to_numpy(dtype=float))),
        },
    )


def _slice_dataframe(df: pd.DataFrame, start: int, end: Optional[int] = None) -> pd.DataFrame:
    if end is None or end >= len(df):
        return df.iloc[start:]
    return df.iloc[start : end + 1]


def compute_trend_leg_volume_profile(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    breakout_buffer: float = 0.001,
) -> Optional[VolumeProfileResult]:
    """Return the profile for an impulse leg following a breakout."""

    if df is None or len(df) < 20:
        return None
    close = _safe_series(df["close"]).to_numpy(dtype=float)
    if close.size < 5:
        return None

    window = min(lookback, close.size - 1)
    rolling_max = pd.Series(close).shift(1).rolling(window=window, min_periods=window // 2).max().to_numpy()
    breakout_idx: Optional[int] = None
    for idx in range(close.size - 1, 0, -1):
        prev_max = rolling_max[idx]
        if not math.isfinite(prev_max) or prev_max <= 0:
            continue
        if close[idx] >= prev_max * (1.0 + breakout_buffer):
            breakout_idx = idx
            break
    if breakout_idx is None:
        return None

    pre_window = min(window, breakout_idx)
    start_search = max(breakout_idx - pre_window, 0)
    segment = close[start_search : breakout_idx + 1]
    if segment.size == 0:
        return None
    local_min_offset = int(np.argmin(segment))
    leg_start = start_search + local_min_offset
    leg_df = _slice_dataframe(df, leg_start)
    profile = compute_volume_profile(leg_df)
    if profile is None:
        return None
    profile.leg_start_index = leg_start
    profile.leg_end_index = len(df) - 1
    profile.leg_type = "trend_impulse"
    profile.metadata.update(
        {
            "breakout_index": float(breakout_idx),
            "breakout_price": float(close[breakout_idx]),
            "leg_low": float(np.min(_safe_series(leg_df["low"]).to_numpy(dtype=float))),
            "leg_high": float(np.max(_safe_series(leg_df["high"]).to_numpy(dtype=float))),
        }
    )
    return profile


def compute_reversion_leg_volume_profile(
    df: pd.DataFrame,
    *,
    balance_lookback: int = 240,
    reclaim_buffer: float = 0.0015,
) -> Optional[VolumeProfileResult]:
    """Profile the leg that follows a failed breakdown and reclaim."""

    if df is None or len(df) < 30:
        return None
    close = _safe_series(df["close"]).to_numpy(dtype=float)
    lows = _safe_series(df["low"]).to_numpy(dtype=float)
    lookback = min(balance_lookback, close.size)
    balance_slice = df.iloc[-lookback:]
    balance_profile = compute_volume_profile(balance_slice)
    if balance_profile is None:
        return None
    balance_poc = balance_profile.poc
    if not math.isfinite(balance_poc) or balance_poc <= 0:
        return None
    band = max(balance_profile.bin_width * 2.0, balance_poc * reclaim_buffer)
    balance_low = balance_poc - band

    failed_idx: Optional[int] = None
    reclaim_idx: Optional[int] = None
    for idx in range(close.size - 1, -1, -1):
        if close[idx] < balance_low:
            failed_idx = idx
            break
    if failed_idx is None:
        return None
    for idx in range(failed_idx, close.size):
        if close[idx] >= balance_poc:
            reclaim_idx = idx
            break
    if reclaim_idx is None:
        return None

    leg_start = max(failed_idx - 3, 0)
    leg_df = _slice_dataframe(df, leg_start)
    profile = compute_volume_profile(leg_df)
    if profile is None:
        return None
    profile.leg_start_index = leg_start
    profile.leg_end_index = len(df) - 1
    profile.leg_type = "failed_breakdown_reclaim"
    leg_low = float(np.min(lows[leg_start : reclaim_idx + 1]))
    profile.metadata.update(
        {
            "failed_low": leg_low,
            "reclaim_index": float(reclaim_idx),
            "balance_poc": float(balance_poc),
            "balance_band": float(band),
        }
    )
    return profile

