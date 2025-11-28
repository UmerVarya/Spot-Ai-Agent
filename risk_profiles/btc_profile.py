"""BTC-specific in-memory risk profile configuration."""
# Previous session multipliers before softening for gentle tilts:
# - BTC: Asia 0.7, Europe 1.0, US 1.2
# - Tier1: Asia 0.85, Europe 1.0, US 1.10
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SymbolProfile:
    """Lightweight container describing per-symbol trade constraints."""

    symbol: str
    direction: str
    min_quote_volume_1m: float
    avg_quote_volume_20_min: float
    vol_expansion_min: float
    atr_min_ratio: float
    min_score_for_trade: float
    session_multipliers: Dict[str, float]
    # Optional per-symbol/tier probability threshold. When None, the
    # global DEFAULT_MIN_PROB_FOR_TRADE is used.
    min_prob_for_trade: Optional[float] = None


BTC_PROFILE = SymbolProfile(
    symbol="BTCUSDT",
    direction="long_only",
    min_quote_volume_1m=800_000.0,
    avg_quote_volume_20_min=500_000.0,
    vol_expansion_min=1.03,
    atr_min_ratio=0.55,
    min_score_for_trade=4.5,
    min_prob_for_trade=None,
    # Session multipliers are intentionally mild (±5–8%) to gently tilt exposure
    # by time-of-day, without completely turning sessions on/off. US gets a
    # slightly higher bar; Asia slightly lower, but all sessions stay in a
    # narrow band around 1.0.
    session_multipliers={
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.05,
    },
)

ETH_PROFILE = SymbolProfile(
    symbol="ETHUSDT",
    direction="long_only",
    min_quote_volume_1m=400_000.0,
    avg_quote_volume_20_min=200_000.0,
    vol_expansion_min=1.02,
    atr_min_ratio=0.50,
    min_score_for_trade=4.2,
    min_prob_for_trade=None,
    session_multipliers={
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.05,
    },
)

SOL_PROFILE = SymbolProfile(
    symbol="SOLUSDT",
    direction="long_only",
    min_quote_volume_1m=120_000.0,
    avg_quote_volume_20_min=70_000.0,
    vol_expansion_min=1.01,
    atr_min_ratio=0.50,
    min_score_for_trade=4.0,
    min_prob_for_trade=None,
    session_multipliers={
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.05,
    },
)

BNB_PROFILE = SymbolProfile(
    symbol="BNBUSDT",
    direction="long_only",
    # BNB is very liquid but calmer than SOL/ETH
    min_quote_volume_1m=100_000.0,
    avg_quote_volume_20_min=50_000.0,
    # BNB doesn't need huge expansion; just avoid dead tape
    vol_expansion_min=1.00,
    # Comfortable in slightly quieter volatility regimes
    atr_min_ratio=0.45,
    # Easier to trade than BTC/ETH; similar or slightly easier than SOL
    min_score_for_trade=4.0,
    min_prob_for_trade=None,
    session_multipliers={
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.05,
    },
)

TIER1_PROFILE = SymbolProfile(
    symbol="__TIER1_DEFAULT__",
    direction="long_only",
    min_quote_volume_1m=80_000.0,
    avg_quote_volume_20_min=50_000.0,
    vol_expansion_min=1.00,
    atr_min_ratio=0.50,
    min_score_for_trade=4.0,
    min_prob_for_trade=None,
    session_multipliers={"asia": 0.95, "europe": 1.0, "us": 1.05},
)

TIER2_PROFILE = SymbolProfile(
    symbol="__TIER2_DEFAULT__",
    direction="long_only",
    min_quote_volume_1m=50_000.0,
    avg_quote_volume_20_min=30_000.0,
    vol_expansion_min=1.00,
    atr_min_ratio=0.45,
    min_score_for_trade=3.9,
    min_prob_for_trade=None,
    session_multipliers={"asia": 0.95, "europe": 1.0, "us": 1.08},
)

TIER3_PROFILE = SymbolProfile(
    symbol="__TIER3_DEFAULT__",
    direction="long_only",
    min_quote_volume_1m=30_000.0,
    avg_quote_volume_20_min=20_000.0,
    vol_expansion_min=1.00,  # just requires "normal" volume
    atr_min_ratio=0.40,
    min_score_for_trade=3.8,
    min_prob_for_trade=None,
    session_multipliers={"asia": 0.95, "europe": 1.0, "us": 1.08},
)


def get_btc_profile() -> SymbolProfile:
    """Return the static BTCUSDT profile."""

    return BTC_PROFILE


def get_eth_profile() -> SymbolProfile:
    """Return the static ETHUSDT profile."""

    return ETH_PROFILE


def get_sol_profile() -> SymbolProfile:
    """Return the static SOLUSDT profile."""

    return SOL_PROFILE


def get_bnb_profile() -> SymbolProfile:
    """Return the static BNBUSDT profile."""

    return BNB_PROFILE


def get_tier_profile(tier: str) -> SymbolProfile:
    """Return the default tier profile for ``tier``."""

    normalized = (tier or "").upper()
    if normalized == "TIER1":
        return TIER1_PROFILE
    if normalized == "TIER2":
        return TIER2_PROFILE
    if normalized == "TIER3":
        return TIER3_PROFILE
    return TIER3_PROFILE
