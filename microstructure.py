"""
Microstructure and liquidity analysis utilities for the Spot‑AI Agent.

Scalping strategies depend heavily on execution quality.  Monitoring
order‑book depth, spreads and imbalances can help avoid entering
positions when the market is illiquid or bid/ask dynamics suggest
imminent volatility.  This module provides simple functions to
calculate such metrics from a depth snapshot.

Since network access is restricted in certain environments and the
``ccxt`` library may not be installed, these functions operate on
already‑fetched order‑book data.  An order book is represented as a
dict with keys ``'bids'`` and ``'asks'``, each being a list of
``[price, quantity]`` pairs sorted best‑to‑worst.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def compute_spread(order_book: Dict[str, List[Tuple[float, float]]]) -> float:
    """
    Compute the absolute spread between the best bid and best ask.

    Parameters
    ----------
    order_book : dict
        Must contain ``'bids'`` and ``'asks'`` lists where each entry is
        ``(price, quantity)``.  Lists should be sorted descending for
        bids and ascending for asks.

    Returns
    -------
    float
        The raw spread (ask_price - bid_price).  Returns NaN if the
        order book is malformed.
    """
    try:
        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        return float(best_ask - best_bid)
    except Exception:
        return float('nan')


def compute_order_book_imbalance(order_book: Dict[str, List[Tuple[float, float]]], depth: int = 10) -> float:
    """
    Compute the order‑book imbalance at the top of the book.

    The imbalance is defined as the ratio of bid volume minus ask
    volume to the total volume within the top ``depth`` levels.  A
    positive value indicates buying pressure, while a negative value
    indicates selling pressure.

    Parameters
    ----------
    order_book : dict
        Order book snapshot as described in ``compute_spread``.
    depth : int, optional
        Number of levels to consider (default 10).

    Returns
    -------
    float
        The imbalance ratio between -1 and 1.  NaN on error.
    """
    try:
        bids = order_book['bids'][:depth]
        asks = order_book['asks'][:depth]
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        total = bid_vol + ask_vol
        if total == 0:
            return float('nan')
        return float((bid_vol - ask_vol) / total)
    except Exception:
        return float('nan')


def is_liquid(order_book: Dict[str, List[Tuple[float, float]]], spread_threshold: float = 0.1, imbalance_threshold: float = 0.5) -> bool:
    """
    Determine whether the market is sufficiently liquid.

    A market is considered liquid if the spread is below
    ``spread_threshold`` (as a fraction of price) and the order‑book
    imbalance is below ``imbalance_threshold`` in magnitude.  Users
    should pass in the current mid‑price to scale the spread.
    """
    spread = compute_spread(order_book)
    imbalance = compute_order_book_imbalance(order_book)
    # Without price context, treat spread threshold as absolute
    if spread != spread:  # NaN check
        return False
    if imbalance != imbalance:
        return False
    return abs(imbalance) < imbalance_threshold and spread < spread_threshold
