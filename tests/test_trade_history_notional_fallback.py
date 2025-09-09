import pandas as pd
import pytest
import trade_storage


def test_history_uses_size_as_notional(tmp_path, monkeypatch):
    data = {
        "symbol": ["BTCUSDT"],
        "direction": ["long"],
        "entry": [100.0],
        "exit": [110.0],
        "size": [100.0],  # already a notional amount
        "outcome": ["tp1"],
        "exit timestamp": ["2024-01-01T01:00:00Z"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(path))
    monkeypatch.setattr(trade_storage, "SIZE_AS_NOTIONAL", True)
    result = trade_storage.load_trade_history_df()
    row = result.iloc[0]
    assert pytest.approx(row["notional"], rel=1e-6) == 100.0
    assert pytest.approx(row["pnl_pct"], rel=1e-6) == 10.0


def test_history_computes_notional_from_quantity(tmp_path, monkeypatch):
    data = {
        "symbol": ["BTCUSDT"],
        "direction": ["long"],
        "entry": [100.0],
        "exit": [110.0],
        "size": [1.0],  # quantity
        "position_size": [1.0],
        "outcome": ["tp1"],
        "exit timestamp": ["2024-01-01T01:00:00Z"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(path))
    monkeypatch.setattr(trade_storage, "SIZE_AS_NOTIONAL", False)
    result = trade_storage.load_trade_history_df()
    row = result.iloc[0]
    assert pytest.approx(row["notional"], rel=1e-6) == 100.0
    assert pytest.approx(row["pnl_pct"], rel=1e-6) == 10.0
