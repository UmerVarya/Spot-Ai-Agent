import numpy as np
import pandas as pd

from pattern_detection import detect_double_bottom, detect_cup_and_handle


def _df_from_prices(prices, volumes):
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volumes,
        }
    )


def test_detect_double_bottom():
    prices = np.concatenate([
        np.linspace(100, 80, 10),
        np.linspace(80, 95, 5),
        np.linspace(95, 81, 5),
        np.linspace(81, 100, 5),
        [105],
    ])
    volumes = np.concatenate([np.full(len(prices) - 1, 100), [200]])
    df = _df_from_prices(prices, volumes)
    pattern, vol = detect_double_bottom(df, lookback=len(df))
    assert pattern and vol


def test_detect_cup_and_handle():
    cup = np.concatenate([np.linspace(100, 80, 10), np.linspace(80, 100, 10)])
    handle = np.array([100, 98, 97, 99, 100, 105])
    prices = np.concatenate([cup, handle])
    volumes = np.concatenate([np.full(len(prices) - 1, 100), [200]])
    df = _df_from_prices(prices, volumes)
    pattern, vol = detect_cup_and_handle(df, lookback=len(df))
    assert pattern and vol

