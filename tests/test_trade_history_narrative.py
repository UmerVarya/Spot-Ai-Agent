import csv
import pandas as pd
import trade_storage

def test_load_trade_history_df_handles_narrative(tmp_path, monkeypatch):
    # Craft a CSV row with a narrative containing commas, braces and quotes
    row = {
        "trade_id": "1",
        "timestamp": "2024-06-01T00:00:00Z",
        "symbol": "BTCUSDT",
        "direction": "long",
        "entry_time": "2024-06-01T00:00:00Z",
        "exit_time": "2024-06-01T01:00:00Z",
        "entry": 100,
        "exit": 110,
        "size": 1,
        "notional": 100,
        "fees": 0,
        "slippage": 0,
        "pnl": 10,
        "pnl_pct": 10,
        "tp1_partial": False,
        "tp2_partial": False,
        "pnl_tp1": 0,
        "pnl_tp2": 0,
        "size_tp1": 0,
        "size_tp2": 0,
        "notional_tp1": 0,
        "notional_tp2": 0,
        "outcome": "tp1",
        "strategy": "pattern1",
        "session": "us",
        "narrative": 'Error: something, details {"foo": 1}',
    }
    path = tmp_path / "history.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trade_storage.TRADE_HISTORY_HEADERS)
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(path))
    df = trade_storage.load_trade_history_df()
    assert not df.empty
    assert df.loc[0, "symbol"] == "BTCUSDT"
