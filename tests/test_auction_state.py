import math

import pandas as pd
import pytest

import auction_state


@pytest.fixture
def sample_df() -> pd.DataFrame:
    data = {
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
    }
    return pd.DataFrame(data)


def test_get_auction_state_trend(monkeypatch, sample_df):
    monkeypatch.setattr(auction_state, "atr_percentile", lambda *args, **kwargs: 0.8)
    monkeypatch.setattr(auction_state, "hurst_exponent", lambda *args, **kwargs: 0.6)

    state = auction_state.get_auction_state(sample_df)

    assert state == "out_of_balance_trend"


def test_get_auction_state_reversion(monkeypatch, sample_df):
    monkeypatch.setattr(auction_state, "atr_percentile", lambda *args, **kwargs: 0.85)
    monkeypatch.setattr(auction_state, "hurst_exponent", lambda *args, **kwargs: 0.4)

    state = auction_state.get_auction_state(sample_df)

    assert state == "out_of_balance_revert"


def test_get_auction_state_balanced(monkeypatch, sample_df):
    monkeypatch.setattr(auction_state, "atr_percentile", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr(auction_state, "hurst_exponent", lambda *args, **kwargs: 0.52)

    state = auction_state.get_auction_state(sample_df)

    assert state == "balanced"


def test_get_auction_state_unknown_on_nan(monkeypatch, sample_df):
    monkeypatch.setattr(auction_state, "atr_percentile", lambda *args, **kwargs: math.nan)
    monkeypatch.setattr(auction_state, "hurst_exponent", lambda *args, **kwargs: 0.5)

    state = auction_state.get_auction_state(sample_df)

    assert state == "unknown"


def test_get_auction_state_missing_columns(sample_df):
    with pytest.raises(KeyError):
        auction_state.get_auction_state(sample_df.drop(columns=["low"]))

