"""
Utility to fetch Bitcoin dominance from CoinGecko with graceful fallback.

This helper function retrieves the current Bitcoin market dominance (percentage
of total crypto market capitalization) from the CoinGecko public API.  The
Coingecko API occasionally changes its response structure, or network issues
may cause missing keys.  To keep the trading system resilient, the function
attempts multiple keys and returns a sensible default (50.0%) when data is
unavailable.

Examples
--------

>>> dominance = get_btc_dominance()
>>> logger.info(f"BTC dominance is {dominance}%")

If an error occurs or the API response is malformed, details are logged and
the function returns 50.0 as a neutral default.
"""

import requests
from log_utils import setup_logger

logger = setup_logger(__name__)


def get_btc_dominance() -> float:
    """
    Fetch the current Bitcoin dominance percentage from CoinGecko.

    Attempts to parse the dominance value from multiple possible nested keys
    in the API response.  If the request fails or the expected keys are
    missing, a fallback value of 50.0 is returned.

    Returns
    -------
    float
        Bitcoin dominance as a percentage of the global crypto market cap.
    """
    url = "https://api.coingecko.com/api/v3/global"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Try the standard nested key path first
        dominance = None
        try:
            dominance = float(data.get("data", {}).get("market_cap_percentage", {}).get("btc"))
        except (TypeError, ValueError):
            dominance = None
        # Fallback: some versions of the API return top-level keys
        if dominance is None:
            try:
                dominance = float(data.get("market_cap_percentage", {}).get("btc"))
            except (TypeError, ValueError):
                dominance = None
        if dominance is not None:
            return round(dominance, 2)
        else:
            logger.warning("BTC dominance data missing in API response; using default 50.0")
            return 50.0
    except Exception as e:
        logger.warning("Failed to fetch BTC dominance: %s", e, exc_info=True)
        # Use neutral fallback
        return 50.0
