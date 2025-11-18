"""Tests for resilient macro context fetching and caching."""

from __future__ import annotations

import importlib
import time

import macro_filter
from macro_data import BTCDominanceSnapshot, FearGreedSnapshot


def _reload_module():
    return importlib.reload(macro_filter)


def test_macro_context_returns_live_values(monkeypatch):
    module = _reload_module()

    monkeypatch.setattr(
        module,
        "get_btc_dominance_cached",
        lambda: BTCDominanceSnapshot(value=47.5, ts=int(time.time())),
    )
    monkeypatch.setattr(
        module,
        "get_fear_greed_cached",
        lambda: FearGreedSnapshot(value=65.0, ts=int(time.time())),
    )

    module.refresh_macro_context_now()
    context = module.get_macro_context()

    assert context["btc_dominance"] == 47.5
    assert context["fear_greed"] == 65
    assert context["macro_sentiment"] == "risk_on"
    assert context["stale"] is False
    assert context["penalty"] == 0.0
    assert context["reason"] == "live"


def test_macro_context_uses_last_good_on_failure(monkeypatch):
    module = _reload_module()

    monkeypatch.setattr(
        module,
        "get_btc_dominance_cached",
        lambda: BTCDominanceSnapshot(value=48.1, ts=int(time.time())),
    )
    monkeypatch.setattr(
        module,
        "get_fear_greed_cached",
        lambda: FearGreedSnapshot(value=40.0, ts=int(time.time())),
    )

    module.refresh_macro_context_now()
    first = module.get_macro_context()

    def _raise_dom():
        raise RuntimeError("dom down")

    monkeypatch.setattr(module, "get_btc_dominance_cached", _raise_dom)
    monkeypatch.setattr(module, "get_fear_greed_cached", lambda: None)

    time.sleep(0.01)
    module.refresh_macro_context_now()
    second = module.get_macro_context()

    assert second["btc_dominance"] == first["btc_dominance"]
    assert second["fear_greed"] == first["fear_greed"]
    assert second["stale"] is True
    assert second["timestamp"] >= first["timestamp"]
    assert second["reason"] in {"stale_from_cache", "macro_missing_neutral"}
    assert second["penalty"] == module.MACRO_STALE_PENALTY
