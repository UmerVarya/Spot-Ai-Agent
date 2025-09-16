import importlib
import pandas as pd
import trade_storage


def test_trade_paths(monkeypatch, tmp_path):
    try:
        import binance.client
        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass
        monkeypatch.setattr(binance.client, "Client", DummyClient)
    except Exception:
        pass
    agent = importlib.import_module("agent")
    dashboard = importlib.import_module("dashboard")
    assert agent.ACTIVE_TRADES_FILE == trade_storage.ACTIVE_TRADES_FILE
    assert dashboard.ACTIVE_TRADES_FILE == trade_storage.ACTIVE_TRADES_FILE
    assert agent.TRADE_HISTORY_FILE == trade_storage.TRADE_HISTORY_FILE
    assert dashboard.TRADE_HISTORY_FILE == trade_storage.TRADE_HISTORY_FILE
    completed_path = tmp_path / "completed.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(completed_path))
    monkeypatch.setattr(trade_storage.os.path, "exists", lambda p: True)
    monkeypatch.setattr(trade_storage.os.path, "getsize", lambda p: 1)
    captured = {"paths": []}
    def fake_read_csv(path, *args, **kwargs):
        captured["paths"].append(path)
        return pd.DataFrame()
    monkeypatch.setattr(trade_storage.pd, "read_csv", fake_read_csv)
    trade_storage.load_trade_history_df()
    assert captured["paths"]
    assert captured["paths"][0] == str(completed_path)
