import math

import pytest

import trade_utils

from alt_data import (
    BasisSnapshot,
    FundingSnapshot,
    OpenInterestSnapshot,
    TakerRatioSnapshot,
    fetch_taker_ratio_raw,
    get_taker_ratio_cached,
)


@pytest.fixture(autouse=True)
def _restore_alt_cache(monkeypatch):
    monkeypatch.setattr(trade_utils, "get_funding_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_basis_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_open_interest_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_taker_ratio_cached", lambda *_, **__: None)


def test_compute_alt_adj_penalises_crowded_longs(monkeypatch):
    monkeypatch.setattr(
        trade_utils,
        "get_funding_cached",
        lambda *_, **__: FundingSnapshot(value=0.02, ts=1),
    )
    monkeypatch.setattr(
        trade_utils,
        "get_basis_cached",
        lambda *_, **__: BasisSnapshot(value=0.02, ts=1),
    )
    monkeypatch.setattr(
        trade_utils,
        "get_open_interest_cached",
        lambda *_, **__: OpenInterestSnapshot(value=5_000_000, change_24h_pct=-15.0, ts=1),
    )
    monkeypatch.setattr(
        trade_utils,
        "get_taker_ratio_cached",
        lambda *_, **__: TakerRatioSnapshot(long_short_ratio=1.6, ts=1),
    )

    adjustment = trade_utils.compute_alt_adj("BTCUSDT")
    assert adjustment is not None
    assert adjustment <= 0
    assert math.isclose(adjustment, -2.0)


def test_compute_alt_adj_respects_symbol_tiers(monkeypatch):
    monkeypatch.setattr(
        trade_utils,
        "get_funding_cached",
        lambda *_, **__: FundingSnapshot(value=0.0022, ts=1),
    )
    monkeypatch.setattr(
        trade_utils,
        "get_basis_cached",
        lambda *_, **__: BasisSnapshot(value=0.0022, ts=1),
    )
    monkeypatch.setattr(
        trade_utils,
        "get_open_interest_cached",
        lambda *_, **__: OpenInterestSnapshot(value=5_000_000, change_24h_pct=6.0, ts=1),
    )
    monkeypatch.setattr(
        trade_utils,
        "get_taker_ratio_cached",
        lambda *_, **__: TakerRatioSnapshot(long_short_ratio=1.35, ts=1),
    )

    btc_adj = trade_utils.compute_alt_adj("BTCUSDT")
    assert btc_adj is not None
    assert abs(btc_adj) <= 1.0

    alt_adj = trade_utils.compute_alt_adj("PEPEUSDT")
    assert alt_adj is not None
    assert alt_adj < -0.5
    assert abs(alt_adj) > abs(btc_adj)


def test_compute_alt_adj_env_override_affects_thresholds(monkeypatch):
    monkeypatch.setenv("ALT_FUND_POS_ALT", "0.0001")
    monkeypatch.setattr(trade_utils, "ALT_THRESHOLDS", trade_utils._build_alt_thresholds())

    monkeypatch.setattr(
        trade_utils,
        "get_funding_cached",
        lambda *_, **__: FundingSnapshot(value=0.0002, ts=1),
    )
    monkeypatch.setattr(trade_utils, "get_basis_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_open_interest_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_taker_ratio_cached", lambda *_, **__: None)

    adjustment = trade_utils.compute_alt_adj("DOGEUSDT")
    assert adjustment is not None
    assert adjustment < 0


def test_compute_alt_adj_returns_none_without_data(monkeypatch):
    monkeypatch.setattr(trade_utils, "get_funding_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_basis_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_open_interest_cached", lambda *_, **__: None)
    monkeypatch.setattr(trade_utils, "get_taker_ratio_cached", lambda *_, **__: None)

    adjustment = trade_utils.compute_alt_adj("BTCUSDT")
    assert adjustment is None

    raw_score = 5.5
    final_score = raw_score + (adjustment or 0.0)
    assert final_score == pytest.approx(raw_score)


def test_fetch_taker_ratio_handles_null_payload(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    payload = [{"timestamp": 1_700_000_000_000, "longShortRatio": None}]
    monkeypatch.setattr(
        "alt_data.requests.get",
        lambda *_, **__: DummyResponse(payload),
    )

    value, ts = fetch_taker_ratio_raw("BTCUSDT")
    assert value is None
    assert ts is None


def test_get_taker_ratio_cached_returns_none_when_raw_unavailable(monkeypatch):
    monkeypatch.setattr("alt_data._TAKER_CACHE", {})
    monkeypatch.setattr("alt_data.fetch_taker_ratio_raw", lambda *_, **__: (None, None))

    snapshot = get_taker_ratio_cached("BTCUSDT", now=0)
    assert snapshot is None
