import pandas as pd
import trade_storage

def test_load_trade_history_df_normalises_columns(tmp_path, monkeypatch):
    data = {
        "Entry Price": [100.0],
        "Exit Price": [110.0],
        "Position Size": [1.0],
        "Trade Outcome": ["tp1"],
        "Entry Timestamp": ["2024-01-01T00:00:00Z"],
        "Exit Timestamp": ["2024-01-01T01:00:00Z"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(path))
    result = trade_storage.load_trade_history_df()
    assert not result.empty
    assert {"entry", "exit", "position_size", "outcome", "entry_time", "exit_time"}.issubset(result.columns)
