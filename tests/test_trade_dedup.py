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
            "size": 50.0,
            "position_size": 0.5,
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
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s1",
            "outcome": "tp2_partial",
            "fees": 0,
            "slippage": 0,
        },
        {
            "trade_id": "2",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T03:00:00Z",
            "entry": 200.0,
            "exit": 210.0,
            "size": 200.0,
            "position_size": 1.0,
            "strategy": "s1",
            "outcome": "tp2",
            "fees": 0,
            "slippage": 0,
        },
    ]
    df = pd.DataFrame(data)
    hist_file = tmp_path / "completed.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))
    result = trade_storage.load_trade_history_df().sort_values("trade_id").reset_index(drop=True)

    # trade 1 collapsed into a single row
    t1 = result[result["trade_id"].astype(str) == "1"].iloc[0]
    assert bool(t1["tp1_partial"])
    assert bool(t1["tp2_partial"])
    assert t1["pnl"] == 15.0
    assert t1["pnl_tp1"] == 5.0
    assert t1["pnl_tp2"] == 10.0
    assert t1["notional"] == 100.0
    assert t1["notional_tp1"] == 50.0
    assert t1["notional_tp2"] == 50.0
    assert t1["size"] == 100.0
    assert t1["size_tp1"] == 0.5
    assert t1["size_tp2"] == 0.5
    assert t1["position_size"] == 1.0

    # second trade remains separate
    assert len(result) == 2
    t2 = result[result["trade_id"].astype(str) == "2"].iloc[0]
    assert t2["pnl"] == 10.0
    assert not bool(t2["tp1_partial"])
    assert not bool(t2["tp2_partial"])

