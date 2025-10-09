import importlib


def load_dashboard(monkeypatch):
    try:
        import binance.client

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(binance.client, "Client", DummyClient)
    except Exception:
        pass
    return importlib.reload(importlib.import_module("dashboard"))


def test_normalise_live_positions_payload_variants(monkeypatch):
    dashboard = load_dashboard(monkeypatch)
    payload_list = [
        {"symbol": "BTCUSDT"},
        {"symbol": "ETHUSDT"},
    ]
    assert dashboard._normalise_live_positions_payload(payload_list) == payload_list

    payload_dict = {"data": {"positions": [{"symbol": "SOLUSDT"}]}}
    normalised = dashboard._normalise_live_positions_payload(payload_dict)
    assert normalised == [{"symbol": "SOLUSDT"}]

    payload_map = {
        "BTCUSDT": {"symbol": "BTCUSDT", "status": "open"},
        "ETHUSDT": {"symbol": "ETHUSDT", "status": "closed"},
    }
    normalised_map = dashboard._normalise_live_positions_payload(payload_map)
    assert {entry["symbol"] for entry in normalised_map} == {"BTCUSDT", "ETHUSDT"}


def test_is_trade_closed_heuristics(monkeypatch):
    dashboard = load_dashboard(monkeypatch)
    assert dashboard._is_trade_closed({"status": "closed"}) is True
    assert dashboard._is_trade_closed({"is_open": False}) is True
    assert dashboard._is_trade_closed({"status": {"closed": "true"}}) is True
    assert dashboard._is_trade_closed({"status": {"is_open": False}}) is True
    assert dashboard._is_trade_closed({"status": {"active": False}}) is True
    assert dashboard._is_trade_closed({"status": {"open": "false"}}) is True
    assert dashboard._is_trade_closed({"status": {"active": 0}}) is True
    assert dashboard._is_trade_closed({"status": {"closed": False}}) is False
    assert dashboard._is_trade_closed({"status": {"open": True}}) is False
    assert dashboard._is_trade_closed({"status": "open"}) is False
    assert (
        dashboard._is_trade_closed({"status": {"tp1": "hit"}, "active": "yes"})
        is False
    )


def test_trade_identity_prefers_trade_id(monkeypatch):
    dashboard = load_dashboard(monkeypatch)
    trade = {"trade_id": "abc123", "symbol": "XRPUSDT"}
    assert dashboard._trade_identity(trade) == "abc123"
    fallback = {"symbol": "XRPUSDT", "entry_time": "2024-01-01T00:00:00Z"}
    assert (
        dashboard._trade_identity(fallback)
        == "XRPUSDT|2024-01-01T00:00:00Z"
    )
