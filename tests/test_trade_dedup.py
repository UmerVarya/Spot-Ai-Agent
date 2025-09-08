import pandas as pd
import trade_storage


def test_deduplicate_history(tmp_path, monkeypatch):
    data = [
        {
            "trade_id": "1",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T01:00:00Z",
            "entry": 100.0,
            "exit": 110.0,
            "size": 100.0,
            "position_size": 1.0,
            "strategy": "s1",
            "outcome": "tp1_partial",
            "fees": 0,
            "slippage": 0,
        },
        {
            "trade_id": "1",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T02:00:00Z",
            "entry": 100.0,
            "exit": 120.0,
            "size": 100.0,
            "position_size": 1.0,
            "strategy": "s1",
            "outcome": "tp2_partial",
            "fees": 0,
            "slippage": 0,
        },
        {
            "trade_id": "1",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T02:00:00Z",
            "entry": 100.0,
            "exit": 120.0,
            "size": 100.0,
            "position_size": 1.0,
            "strategy": "s1",
            "outcome": "tp2_partial",
            "fees": 0,
            "slippage": 0,
        },
    ]
    df = pd.DataFrame(data)
    hist_file = tmp_path / "completed.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))
    result = trade_storage.load_trade_history_df()
    # partial exits collapsed into a single row
    assert len(result) == 1
    row = result.iloc[0]
    assert bool(row["tp1_partial"])
    assert bool(row["tp2_partial"])
    assert row["pnl"] == 30.0
    assert row["pnl_pct"] == 30.0

