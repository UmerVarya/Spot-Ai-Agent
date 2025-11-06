"""Tests for resilient macro context fetching and caching."""

from __future__ import annotations

import importlib

import macro_filter


def _reload_module():
    return importlib.reload(macro_filter)


def test_macro_context_returns_live_values(monkeypatch):
    module = _reload_module()

    now = [1_000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "get_btc_dominance", lambda: 47.5)
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: 65)

    context = module.get_macro_context()

    assert context["btc_dominance"] == 47.5
    assert context["fear_greed"] == 65
    assert context["macro_sentiment"] == "risk_on"
    assert context["stale"] is False
    assert context["status"] == "live"
    assert context["stale_for"] == 0.0


def test_macro_context_uses_last_good_on_failure(monkeypatch):
    module = _reload_module()

    now = [2_000.0]

    def _now():
        return now[0]

    monkeypatch.setattr(module.time, "time", _now)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    monkeypatch.setattr(module, "get_btc_dominance", lambda: 48.1)
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: 40)

    first = module.get_macro_context()

    def _raise_dom():
        raise RuntimeError("dom down")

    monkeypatch.setattr(module, "get_btc_dominance", _raise_dom)
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: None)

    now[0] = 2_000.0 + 600.0

    second = module.get_macro_context()

    assert second["btc_dominance"] == first["btc_dominance"]
    assert second["fear_greed"] == first["fear_greed"]
    assert second["stale"] is True
    assert second["status"] == "cached"
    assert second["stale_for"] > 0
    assert second["timestamp"] >= first["timestamp"]


def test_macro_context_defaults_when_no_cache(monkeypatch):
    module = _reload_module()

    monkeypatch.setattr(module.time, "time", lambda: 3_000.0)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "get_btc_dominance", lambda: (_ for _ in ()).throw(RuntimeError("dom")))
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: (_ for _ in ()).throw(RuntimeError("fg")))

    context = module.get_macro_context()

    assert context["btc_dominance"] == module.DEFAULT_BTC_DOMINANCE
    assert context["fear_greed"] == module.DEFAULT_FEAR_GREED
    assert context["macro_sentiment"] == module.DEFAULT_SENTIMENT
    assert context["status"] == "neutral"
    assert context["stale"] is True
    assert context["stale_for"] == 0.0
