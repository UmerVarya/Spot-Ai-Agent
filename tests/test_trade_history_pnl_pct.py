import pandas as pd
import pytest
import trade_storage


def test_load_trade_history_df_computes_pnl_pct(tmp_path, monkeypatch):
    data = {
        "symbol": ["BTCUSDT"],
        "direction": ["long"],
        "entry": [100.0],
        "exit": [110.0],
        "size": [100.0],
        "position_size": [1.0],
        "outcome": ["tp1"],
        "net pnl": [10.0],  # legacy header with space
        "notional value": [100.0],
        "exit timestamp": ["2024-01-01T01:00:00Z"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(path))
    result = trade_storage.load_trade_history_df()
    assert pytest.approx(result.iloc[0]["pnl_pct"], rel=1e-6) == 10.0
    assert pytest.approx(result.iloc[0]["PnL (%)"], rel=1e-6) == 10.0

