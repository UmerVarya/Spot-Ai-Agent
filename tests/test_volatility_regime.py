import math

import numpy as np
import pandas as pd
import pytest

from volatility_regime import atr_percentile


def _generate_ohlc(rows: int = 200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="H")
    prices = np.linspace(100, 120, rows) + np.random.normal(scale=0.5, size=rows)
    high = prices + np.random.uniform(0.2, 1.0, size=rows)
    low = prices - np.random.uniform(0.2, 1.0, size=rows)
    close = prices + np.random.uniform(-0.5, 0.5, size=rows)
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=index)


def test_atr_percentile_accepts_dataframe_inputs():
    df = _generate_ohlc()

    series_result = atr_percentile(df["high"], df["low"], df["close"], window=14, lookback=50)
    frame_result = atr_percentile(df[["high"]], df[["low"]], df[["close"]], window=14, lookback=50)

    assert not math.isnan(frame_result)
    assert math.isclose(series_result, frame_result, rel_tol=1e-12)


def test_atr_percentile_dataframe_without_named_column_raises():
    df = _generate_ohlc()[["high", "low"]]

    unnamed = df.rename(columns={"high": 0, "low": 1})
    with pytest.raises(ValueError):
        atr_percentile(unnamed, df[["low"]], df[["low"]], window=14, lookback=50)
