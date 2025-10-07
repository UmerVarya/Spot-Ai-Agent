import pandas as pd

from orderflow import compute_orderflow_features


def _build_price_df(closes):
    highs = [price + 0.05 for price in closes]
    lows = [price - 0.05 for price in closes]
    volumes = [1000.0 for _ in closes]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def test_absorption_score_highlight_ask_absorption():
    df = _build_price_df([100.0, 100.02, 100.01, 100.015])
    live_trades = {
        "total_base_volume": 120.0,
        "net_base_volume": 40.0,
        "buy_base_volume": 80.0,
        "sell_base_volume": 40.0,
        "trade_rate_per_sec": 1.0,
        "cumulative_total_base_volume": 120.0,
        "cumulative_net_base_volume": 40.0,
        "price_footprint_bin_size": 0.01,
        "price_footprint_bins": [
            {"price": 100.0, "buy_volume": 60.0, "sell_volume": 10.0},
            {"price": 100.01, "buy_volume": 20.0, "sell_volume": 10.0},
        ],
    }

    features = compute_orderflow_features(df, live_trades=live_trades)

    assert features["absorption_score"] > 0
    assert features["delta_divergence"] > -1.0  # should be finite


def test_absorption_score_highlight_bid_absorption():
    df = _build_price_df([100.0, 99.98, 100.01, 100.0])
    live_trades = {
        "total_base_volume": 140.0,
        "net_base_volume": -50.0,
        "buy_base_volume": 40.0,
        "sell_base_volume": 100.0,
        "trade_rate_per_sec": 1.0,
        "cumulative_total_base_volume": 140.0,
        "cumulative_net_base_volume": -50.0,
        "price_footprint_bin_size": 0.01,
        "price_footprint_bins": [
            {"price": 99.98, "buy_volume": 5.0, "sell_volume": 35.0},
            {"price": 100.0, "buy_volume": 10.0, "sell_volume": 40.0},
        ],
    }

    features = compute_orderflow_features(df, live_trades=live_trades)

    assert features["absorption_score"] < 0
    assert features["delta_divergence"] > -1.0
