import importlib
import trade_storage


def test_dashboard_paths(monkeypatch):
    try:
        import binance.client
        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass
        monkeypatch.setattr(binance.client, "Client", DummyClient)
    except Exception:
        pass
    dashboard = importlib.import_module("dashboard")
    assert dashboard.ACTIVE_TRADES_FILE == trade_storage.ACTIVE_TRADES_FILE
    assert dashboard.TRADE_HISTORY_FILE == trade_storage.TRADE_HISTORY_FILE
