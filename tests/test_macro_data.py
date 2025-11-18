from __future__ import annotations

import importlib

import macro_data


def _reset_module(monkeypatch):
    importlib.reload(macro_data)
    monkeypatch.setattr(macro_data, "_FEAR_GREED_CACHE", None)
    monkeypatch.setattr(macro_data, "_BTC_DOM_CACHE", None)
    return macro_data


def test_fetch_fear_greed_parses_payload(monkeypatch):
    module = _reset_module(monkeypatch)

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"value": "42", "timestamp": "1700000000"}]}

    monkeypatch.setattr(module.requests, "get", lambda url, timeout=None: DummyResponse())

    value, ts = module.fetch_fear_greed_raw()
    assert value == 42.0
    assert ts == 1_700_000_000


def test_fetch_btc_dominance_parses_payload(monkeypatch):
    module = _reset_module(monkeypatch)

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "market_cap_percentage": {"btc": 49.5},
                    "updated_at": 1_700_000_000,
                }
            }

    monkeypatch.setattr(module.requests, "get", lambda url, timeout=None: DummyResponse())

    value, ts = module.fetch_btc_dominance_raw()
    assert value == 49.5
    assert ts == 1_700_000_000


def test_fear_greed_cache_refreshes_when_stale(monkeypatch):
    module = _reset_module(monkeypatch)
    module.FNG_MAX_AGE_SECONDS = 100.0

    calls = {"count": 0}

    def fake_fetch():
        calls["count"] += 1
        return 40.0 + calls["count"], 1_700_000_000 + calls["count"]

    monkeypatch.setattr(module, "fetch_fear_greed_raw", fake_fetch)

    snap1 = module.get_fear_greed_cached(now=1_700_000_050)
    assert snap1 is not None
    assert calls["count"] == 1

    # Within freshness window -> no extra fetch.
    snap2 = module.get_fear_greed_cached(now=1_700_000_100)
    assert snap2 is snap1
    assert calls["count"] == 1

    # Force stale -> refetch.
    module.FNG_MAX_AGE_SECONDS = 10.0
    snap3 = module.get_fear_greed_cached(now=1_700_000_500)
    assert snap3 is not None
    assert calls["count"] == 2


def test_btc_dominance_returns_none_on_failure(monkeypatch):
    module = _reset_module(monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "fetch_btc_dominance_raw", lambda: (None, None))

    assert module.get_btc_dominance_cached(now=1_700_000_000) is None
