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
    completed_path = tmp_path / "completed.csv"
    other_path = tmp_path / "other.csv"
    monkeypatch.setattr(trade_storage, "COMPLETED_TRADES_FILE", str(completed_path))
    monkeypatch.setattr(trade_storage, "TRADE_LOG_FILE", str(other_path))
    monkeypatch.setattr(trade_storage.os.path, "exists", lambda p: True)
    monkeypatch.setattr(trade_storage.os.path, "getsize", lambda p: 1)
    captured = {}
    def fake_read_csv(path, *args, **kwargs):
        captured["path"] = path
        return pd.DataFrame()
    monkeypatch.setattr(trade_storage.pd, "read_csv", fake_read_csv)
    trade_storage.load_trade_history_df()
    assert captured["path"] == str(completed_path)
