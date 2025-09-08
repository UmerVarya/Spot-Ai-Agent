import pandas as pd
import trade_storage
import logging


def test_load_trade_history_df_drops_invalid_rows(tmp_path, monkeypatch, caplog):
    data = [
        {
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry": 100.0,
            "exit": 110.0,
            "size": 100.0,
            "position_size": 1.0,
            "outcome": "tp1",
        },
        {
            "symbol": "1000PEPEUSDT",
            "direction": "long",
            "entry": 50.0,
            "exit": 55.0,
            "size": 50.0,
            "position_size": 1.0,
            "outcome": "tp1",
        },
        {
            "symbol": 12345,
            "direction": "up",
            "entry": 200.0,
            "exit": 210.0,
            "size": 200.0,
            "position_size": 1.0,
            "outcome": "tp1",
        },
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(path))
    with caplog.at_level(logging.WARNING):
        result = trade_storage.load_trade_history_df()
    assert len(result) == 2
    assert set(result["symbol"]) == {"BTCUSDT", "1000PEPEUSDT"}
    assert "Dropped 1 malformed trade rows" in caplog.text
