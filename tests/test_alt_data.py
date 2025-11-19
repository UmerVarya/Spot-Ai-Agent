import math

import pytest

import trade_utils

from alt_data import (
    BasisSnapshot,
    FundingSnapshot,
    OpenInterestSnapshot,
    TakerRatioSnapshot,
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
