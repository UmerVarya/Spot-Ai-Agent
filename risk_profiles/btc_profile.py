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
    min_quote_volume_1m=1_000_000.0,
    avg_quote_volume_20_min=700_000.0,
    vol_expansion_min=1.1,
    atr_min_ratio=0.55,
    min_score_for_trade=4.5,
    session_multipliers={
        "asia": 0.7,
        "europe": 1.0,
        "us": 1.2,
    },
)


def get_btc_profile() -> SymbolProfile:
    """Return the static BTCUSDT profile."""

    return BTC_PROFILE
