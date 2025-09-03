import importlib
import pandas as pd

def test_optimize_indicator_weights(monkeypatch, tmp_path):
    sig_file = tmp_path / "signal_log.csv"
    trade_file = tmp_path / "trade_log.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
        "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "ema_trigger": [1, 1, 0],
        "macd_trigger": [1, 0, 1],
    }).to_csv(sig_file, index=False)
    pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
        "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        "outcome": ["win", "loss", "win"],
    }).to_csv(trade_file, index=False)
    monkeypatch.setenv("SIGNAL_LOG_FILE", str(sig_file))
    monkeypatch.setenv("TRADE_HISTORY_FILE", str(trade_file))
    import trade_storage
    importlib.reload(trade_storage)
    weight_optimizer = importlib.reload(importlib.import_module("weight_optimizer"))
    base = {"ema": 1.0, "macd": 1.0}
    optimized = weight_optimizer.optimize_indicator_weights(base)
    assert optimized["ema"] != base["ema"]
    assert optimized["macd"] == base["macd"]
