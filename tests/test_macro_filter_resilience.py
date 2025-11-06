"""Tests for resilient macro context fetching and caching."""

from __future__ import annotations

import importlib
import time

import macro_filter


def _reload_module():
    return importlib.reload(macro_filter)


def test_macro_context_returns_live_values(monkeypatch):
    module = _reload_module()

    monkeypatch.setattr(module, "get_btc_dominance", lambda: 47.5)
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: 65)

    context = module.get_macro_context()

    assert context["btc_dominance"] == 47.5
    assert context["fear_greed"] == 65
    assert context["macro_sentiment"] == "risk_on"
    assert context["stale"] is False


def test_macro_context_uses_last_good_on_failure(monkeypatch):
    module = _reload_module()

    monkeypatch.setattr(module, "get_btc_dominance", lambda: 48.1)
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: 40)

    first = module.get_macro_context()

    def _raise_dom():
        raise RuntimeError("dom down")

    monkeypatch.setattr(module, "get_btc_dominance", _raise_dom)
    monkeypatch.setattr(module, "get_fear_greed_index", lambda: None)

    time.sleep(0.01)
    second = module.get_macro_context()

    assert second["btc_dominance"] == first["btc_dominance"]
    assert second["fear_greed"] == first["fear_greed"]
    assert second["stale"] is True
    assert second["timestamp"] >= first["timestamp"]
