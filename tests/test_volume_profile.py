import math

import pandas as pd

from volume_profile import (
    compute_volume_profile,
    compute_trend_leg_volume_profile,
    compute_reversion_leg_volume_profile,
)


def _build_df(prices, volumes):
    index = pd.date_range("2024-01-01", periods=len(prices), freq="T")
    data = {
        "open": prices,
        "high": [p + 0.05 for p in prices],
        "low": [p - 0.05 for p in prices],
        "close": prices,
        "volume": volumes,
    }
    return pd.DataFrame(data, index=index)


def test_compute_volume_profile_basic():
    prices = [100, 100.2, 101.1, 101.2, 103.5, 103.4]
    volumes = [50, 60, 300, 280, 90, 80]
    df = _build_df(prices, volumes)
    result = compute_volume_profile(df, bin_size_pct=0.002)
    assert result is not None
    assert math.isfinite(result.poc)
    # Highest volume cluster is around 101
    assert abs(result.poc - 101.15) < 0.3
    assert result.lvns, "Expected at least one LVN"


def test_trend_leg_profile_detects_breakout():
    base_prices = [100 + (i % 5) * 0.1 for i in range(80)]
    breakout_prices = [101 + i * 0.2 for i in range(30)]
    prices = base_prices + breakout_prices
    volumes = [100] * 80 + [200] * 30
    df = _build_df(prices, volumes)
    profile = compute_trend_leg_volume_profile(df)
    assert profile is not None
    assert profile.leg_type == "trend_impulse"
    assert profile.leg_start_index < len(df) - 1
    touched = profile.touched_lvn(
        close=df["close"].iloc[-1],
        high=df["high"].iloc[-1],
        low=df["low"].iloc[-1],
    )
    assert touched is None or math.isfinite(touched)


def test_reversion_leg_profile_identifies_failed_breakdown():
    # Simulate balance around 100, breakdown to 98, then reclaim to 101
    balance = [100 + ((i % 3) - 1) * 0.1 for i in range(120)]
    breakdown = [99.2, 98.8, 98.6, 99.5, 100.2, 100.8, 101.3, 101.5]
    prices = balance + breakdown
    volumes = [150] * len(balance) + [220, 250, 260, 240, 210, 200, 190, 185]
    df = _build_df(prices, volumes)
    profile = compute_reversion_leg_volume_profile(df)
    assert profile is not None
    assert profile.leg_type == "failed_breakdown_reclaim"
    failed_low = profile.metadata.get("failed_low")
    assert failed_low is not None
    assert failed_low < profile.metadata.get("balance_poc", failed_low)  # breakdown below balance
    lvn_touch = profile.touched_lvn(
        close=df["close"].iloc[-1],
        high=df["high"].iloc[-1],
        low=df["low"].iloc[-1],
    )
    assert lvn_touch is None or math.isfinite(lvn_touch)
