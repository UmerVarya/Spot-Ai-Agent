import importlib
from typing import Iterator


def _reload_module(monkeypatch, api_key: str | None = "test-key"):
    if api_key is None:
        monkeypatch.delenv("BITQUERY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("BITQUERY_API_KEY", api_key)
    import onchain_bitquery as module

    module = importlib.reload(module)
    return module


def test_get_btc_onchain_signal_uses_cache(monkeypatch):
    module = _reload_module(monkeypatch)

    calls = []

    def fake_post(query, variables=None):
        calls.append((query, variables))
        return {"bitcoin": {"transactions": [{"count": 42}]}}

    monkeypatch.setattr(module, "_bitquery_post", fake_post)
    monkeypatch.setattr(module, "_now", lambda: 1_000.0)
    module.BITQUERY_REFRESH_SECONDS = 3_600

    result_first = module.get_btc_onchain_signal()
    assert result_first["total_transfers"] == 42.0
    assert len(calls) == 1

    monkeypatch.setattr(module, "_now", lambda: 1_000.0 + 10.0)
    result_second = module.get_btc_onchain_signal()
    assert result_second == result_first
    assert len(calls) == 1

    def fake_post_second(query, variables=None):
        calls.append((query, variables))
        return {"bitcoin": {"transactions": [{"count": 84}]}}

    monkeypatch.setattr(module, "_bitquery_post", fake_post_second)
    monkeypatch.setattr(module, "_now", lambda: 1_000.0 + 3_601.0)
    result_third = module.get_btc_onchain_signal()
    assert result_third["total_transfers"] == 84.0
    assert len(calls) == 2


def test_get_btc_onchain_signal_pauses_after_failures(monkeypatch):
    module = _reload_module(monkeypatch)
    module.BITQUERY_REFRESH_SECONDS = 3_600
    module.BITQUERY_PAUSE_SECONDS = 600  # shorten for the test

    call_count = 0

    def failing_post(query, variables=None):
        nonlocal call_count
        call_count += 1
        return None

    monkeypatch.setattr(module, "_bitquery_post", failing_post)

    times: Iterator[float] = iter([0.0, 3_600.0, 7_200.0, 7_300.0, 7_800.0, 7_200.0 + 600.0 + 1.0])
    monkeypatch.setattr(module, "_now", lambda: next(times))

    assert module.get_btc_onchain_signal() is None
    assert module.get_btc_onchain_signal() is None
    assert module.get_btc_onchain_signal() is None
    assert call_count == 3

    assert module.get_btc_onchain_signal() is None
    assert module.get_btc_onchain_signal() is None
    assert call_count == 3

    assert module.get_btc_onchain_signal() is None
    assert call_count == 4


def test_get_btc_onchain_signal_without_api_key_neutral(monkeypatch):
    module = _reload_module(monkeypatch)

    module._BITQUERY_CACHE[24] = (
        {"ok": True, "total_transfers": 10.0, "window_hours": 24, "fetched_at": 0.0},
        0.0,
    )

    module.BITQUERY_REFRESH_SECONDS = 3_600
    monkeypatch.setattr(module, "_now", lambda: 100.0)
    assert module.get_btc_onchain_signal()["total_transfers"] == 10.0

    monkeypatch.delenv("BITQUERY_API_KEY", raising=False)
    monkeypatch.setattr(module, "_now", lambda: 200.0)
    assert module.get_btc_onchain_signal() is None
    assert module._BITQUERY_CACHE == {}
