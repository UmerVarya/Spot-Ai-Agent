"""BTC-specific in-memory risk profile configuration."""
from dataclasses import dataclass
from typing import Dict


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


BTC_PROFILE = SymbolProfile(
    symbol="BTCUSDT",
    direction="long_only",
    min_quote_volume_1m=800_000.0,
    avg_quote_volume_20_min=500_000.0,
    vol_expansion_min=1.03,
    atr_min_ratio=0.55,
    min_score_for_trade=4.5,
    session_multipliers={
        "asia": 0.7,
        "europe": 1.0,
        "us": 1.2,
    },
)

ETH_PROFILE = SymbolProfile(
    symbol="ETHUSDT",
    direction="long_only",
    min_quote_volume_1m=400_000.0,
    avg_quote_volume_20_min=250_000.0,
    vol_expansion_min=1.02,
    atr_min_ratio=0.50,
    min_score_for_trade=4.2,
    session_multipliers={
        "asia": 0.8,
        "europe": 1.0,
        "us": 1.15,
    },
)

SOL_PROFILE = SymbolProfile(
    symbol="SOLUSDT",
    direction="long_only",
    min_quote_volume_1m=120_000.0,
    avg_quote_volume_20_min=80_000.0,
    vol_expansion_min=1.01,
    atr_min_ratio=0.50,
    min_score_for_trade=4.0,
    session_multipliers={
        "asia": 0.9,
        "europe": 1.0,
        "us": 1.15,
    },
)

BNB_PROFILE = SymbolProfile(
    symbol="BNBUSDT",
    direction="long_only",
    # BNB is very liquid but calmer than SOL/ETH
    min_quote_volume_1m=100_000.0,
    avg_quote_volume_20_min=60_000.0,
    # BNB doesn't need huge expansion; just avoid dead tape
    vol_expansion_min=1.00,
    # Comfortable in slightly quieter volatility regimes
    atr_min_ratio=0.45,
    # Easier to trade than BTC/ETH; similar or slightly easier than SOL
    min_score_for_trade=4.0,
    session_multipliers={
        "asia": 0.95,
        "europe": 1.0,
        "us": 1.10,
    },
)

TIER1_PROFILE = SymbolProfile(
    symbol="__TIER1_DEFAULT__",
    direction="long_only",
    min_quote_volume_1m=80_000.0,
    avg_quote_volume_20_min=60_000.0,
    vol_expansion_min=1.00,
    atr_min_ratio=0.50,
    min_score_for_trade=4.0,
    session_multipliers={"asia": 0.85, "europe": 1.0, "us": 1.10},
)

TIER2_PROFILE = SymbolProfile(
    symbol="__TIER2_DEFAULT__",
    direction="long_only",
    min_quote_volume_1m=60_000.0,
    avg_quote_volume_20_min=40_000.0,
    vol_expansion_min=1.00,
    atr_min_ratio=0.45,
    min_score_for_trade=3.9,
    session_multipliers={"asia": 0.9, "europe": 1.0, "us": 1.08},
)

TIER3_PROFILE = SymbolProfile(
    symbol="__TIER3_DEFAULT__",
    direction="long_only",
    min_quote_volume_1m=40_000.0,
    avg_quote_volume_20_min=30_000.0,
    vol_expansion_min=1.00,  # just requires "normal" volume
    atr_min_ratio=0.40,
    min_score_for_trade=3.8,
    session_multipliers={"asia": 0.95, "europe": 1.0, "us": 1.05},
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
