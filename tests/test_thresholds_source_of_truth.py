from config import DEFAULT_MIN_PROB_FOR_TRADE
from probability_gating import (
    get_effective_min_prob_for_symbol,
    should_veto_on_probability,
)
from risk_profiles.btc_profile import SymbolProfile


def _profile(min_prob_for_trade=None) -> SymbolProfile:
    return SymbolProfile(
        symbol="TESTUSDT",
        direction="long_only",
        min_quote_volume_1m=0.0,
        avg_quote_volume_20_min=0.0,
        vol_expansion_min=0.0,
        atr_min_ratio=0.0,
        min_score_for_trade=0.0,
        min_prob_for_trade=min_prob_for_trade,
        session_multipliers={},
    )


def test_default_prob_threshold_uses_global_default():
    profile = _profile(min_prob_for_trade=None)

    min_prob = get_effective_min_prob_for_symbol(
        symbol="BTCUSDT", profile=profile, override_min_prob=None
    )

    assert min_prob == DEFAULT_MIN_PROB_FOR_TRADE


def test_profile_prob_threshold_overrides_default():
    profile = _profile(min_prob_for_trade=0.6)

    min_prob = get_effective_min_prob_for_symbol(
        symbol="TESTUSDT", profile=profile, override_min_prob=None
    )

    assert min_prob == 0.6


def test_backtest_override_prob_has_highest_precedence():
    profile = _profile(min_prob_for_trade=0.6)

    min_prob = get_effective_min_prob_for_symbol(
        symbol="TESTUSDT", profile=profile, override_min_prob=0.55
    )

    assert min_prob == 0.55


def test_live_veto_uses_helper_threshold():
    profile = _profile(min_prob_for_trade=0.6)
    min_prob = get_effective_min_prob_for_symbol(
        symbol="TESTUSDT", profile=profile, override_min_prob=None
    )

    assert should_veto_on_probability(ml_prob=0.59, min_prob=min_prob) is True
    assert should_veto_on_probability(ml_prob=0.61, min_prob=min_prob) is False
