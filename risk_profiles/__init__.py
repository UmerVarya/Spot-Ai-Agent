"""Risk profile definitions for spot trading symbols."""

from .btc_profile import (
    get_btc_profile,
    get_eth_profile,
    get_sol_profile,
    get_bnb_profile,
    get_tier_profile,
)

__all__ = [
    "get_btc_profile",
    "get_eth_profile",
    "get_sol_profile",
    "get_bnb_profile",
    "get_tier_profile",
]
