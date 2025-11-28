"""Tests for softened session multipliers in risk profiles."""

from risk_profiles import get_btc_profile, get_tier_profile
from risk_profiles.btc_profile import (
    BNB_PROFILE,
    BTC_PROFILE,
    ETH_PROFILE,
    SOL_PROFILE,
    TIER1_PROFILE,
    TIER2_PROFILE,
    TIER3_PROFILE,
)


def test_btc_session_multipliers_softened():
    btc_profile = get_btc_profile()

    assert btc_profile.session_multipliers == {
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.05,
    }


def test_tier2_session_multipliers_softened():
    tier2_profile = get_tier_profile("TIER2")

    assert tier2_profile.session_multipliers == {
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.08,
    }


def test_all_session_multipliers_within_soft_bounds():
    """Session multipliers softened to avoid over-penalizing US and over-favoring Asia."""

    profiles = [
        BTC_PROFILE,
        ETH_PROFILE,
        SOL_PROFILE,
        BNB_PROFILE,
        TIER1_PROFILE,
        TIER2_PROFILE,
        TIER3_PROFILE,
    ]

    for profile in profiles:
        for multiplier in profile.session_multipliers.values():
            assert 0.9 <= multiplier <= 1.08
