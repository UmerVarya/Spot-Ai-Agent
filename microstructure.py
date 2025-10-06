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

import math
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


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


def _normalise_levels(levels: List[Tuple[Any, Any]], depth: int) -> List[Tuple[float, float]]:
    """Return cleaned order book levels limited to ``depth`` entries."""

    normalised: List[Tuple[float, float]] = []
    for raw_price, raw_qty in levels[:depth]:
        try:
            price = float(raw_price)
            qty = float(raw_qty)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        normalised.append((price, qty))
    return normalised


def _volume_baseline(levels: List[Tuple[float, float]]) -> float:
    """Return a robust baseline volume using the median of level sizes."""

    if not levels:
        return 0.0
    volumes = [qty for _, qty in levels if qty > 0]
    if not volumes:
        return 0.0
    base = median(volumes)
    if base <= 0:
        base = sum(volumes) / len(volumes)
    return max(base, 0.0)


def _find_significant_level(
    levels: List[Tuple[float, float]],
    *,
    side: str,
    reference_price: Optional[float],
    multiplier: float,
    max_distance: float,
) -> Optional[Dict[str, float]]:
    """Return the closest level whose size exceeds ``multiplier`` * baseline."""

    if reference_price is not None and reference_price <= 0:
        reference_price = None

    baseline = _volume_baseline(levels)
    if baseline <= 0:
        return None
    threshold = baseline * multiplier
    for price, qty in levels:
        if qty < threshold:
            continue
        if reference_price is not None and reference_price > 0:
            if side == "bid":
                distance = (reference_price - price) / reference_price
                if distance < 0 or distance > max_distance:
                    continue
            else:  # ask side
                distance = (price - reference_price) / reference_price
                if distance < 0 or distance > max_distance:
                    continue
        return {"price": price, "quantity": qty}
    return None


def plan_execution(
    side: str,
    current_price: Optional[float],
    order_book: Dict[str, List[Tuple[float, float]]],
    *,
    depth: int = 10,
    wall_multiplier: float = 3.0,
    support_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """Derive a microstructure-aware execution plan.

    Parameters
    ----------
    side : str
        ``"buy"`` or ``"sell"`` depending on the intended trade action.
    current_price : Optional[float]
        Reference price used to gauge proximity to large bids/asks.
    order_book : dict
        Snapshot containing ``bids`` and ``asks`` lists.
    depth : int, optional
        Number of levels to analyse on each side (default 10).
    wall_multiplier : float, optional
        Multiplier applied to the baseline volume to deem a level a wall.
    support_multiplier : float, optional
        Multiplier applied to identify supportive liquidity on the same side
        as the intended trade.

    Returns
    -------
    dict
        Dictionary containing the suggested execution price, aggressiveness,
        liquidity notes and any recommended position size adjustment.
    """

    side = (side or "").lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    bids = _normalise_levels(order_book.get("bids", []), depth)
    asks = _normalise_levels(order_book.get("asks", []), depth)
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    reference_price = current_price
    if reference_price is None:
        reference_price = best_ask if side == "buy" else best_bid

    notes: List[str] = []
    size_multiplier = 1.0
    recommended_price: Optional[float]
    if side == "buy":
        recommended_price = best_ask if best_ask is not None else reference_price
        support = _find_significant_level(
            bids,
            side="bid",
            reference_price=reference_price,
            multiplier=support_multiplier,
            max_distance=0.0015,
        )
        resistance = _find_significant_level(
            asks,
            side="ask",
            reference_price=reference_price,
            multiplier=wall_multiplier,
            max_distance=0.002,
        )
        if support:
            notes.append(
                f"Support bid {support['quantity']:.2f}@{support['price']:.6f}"
            )
            buffer = 0.0
            if best_ask is not None and best_bid is not None:
                buffer = max((best_ask - best_bid) * 0.25, support["price"] * 0.0002)
            elif reference_price is not None:
                buffer = reference_price * 0.0002
            candidate = support["price"] + buffer
            if best_ask is not None:
                candidate = min(candidate, best_ask)
            recommended_price = max(candidate, best_bid or candidate)
        if resistance:
            notes.append(
                f"Sell wall {resistance['quantity']:.2f}@{resistance['price']:.6f}"
            )
            buffer = 0.0
            if best_ask is not None and best_bid is not None:
                buffer = max((best_ask - best_bid) * 0.25, resistance["price"] * 0.0002)
            elif reference_price is not None:
                buffer = reference_price * 0.0002
            candidate = resistance["price"] - buffer
            if best_bid is not None:
                candidate = max(candidate, best_bid)
            if recommended_price is None:
                recommended_price = candidate
            else:
                recommended_price = min(recommended_price, candidate)
            size_multiplier = min(size_multiplier, 0.6)
    else:  # sell
        recommended_price = best_bid if best_bid is not None else reference_price
        resistance = _find_significant_level(
            asks,
            side="ask",
            reference_price=reference_price,
            multiplier=wall_multiplier,
            max_distance=0.002,
        )
        if resistance:
            notes.append(
                f"Sell wall {resistance['quantity']:.2f}@{resistance['price']:.6f}"
            )
            buffer = 0.0
            if best_ask is not None and best_bid is not None:
                buffer = max((best_ask - best_bid) * 0.25, resistance["price"] * 0.0002)
            elif reference_price is not None:
                buffer = reference_price * 0.0002
            candidate = resistance["price"] - buffer
            if best_bid is not None:
                candidate = max(candidate, best_bid)
            recommended_price = min(recommended_price if recommended_price is not None else candidate, candidate)
        support = _find_significant_level(
            bids,
            side="bid",
            reference_price=reference_price,
            multiplier=support_multiplier,
            max_distance=0.0015,
        )
        if support:
            notes.append(
                f"Bid support {support['quantity']:.2f}@{support['price']:.6f}"
            )
            candidate = support["price"]
            if recommended_price is None or candidate > recommended_price:
                recommended_price = candidate

    if recommended_price is None:
        recommended_price = reference_price
    if best_bid is not None and recommended_price is not None:
        recommended_price = max(recommended_price, best_bid)
    if best_ask is not None and recommended_price is not None:
        recommended_price = min(recommended_price, best_ask) if side == "buy" else min(recommended_price, best_ask)

    imbalance = compute_order_book_imbalance(order_book, depth=depth)
    aggressiveness = "aggressive"
    if side == "buy" and best_ask is not None and recommended_price is not None:
        if recommended_price <= best_bid or recommended_price < best_ask:
            aggressiveness = "passive"
    elif side == "sell" and best_bid is not None and recommended_price is not None:
        if recommended_price <= best_bid:
            aggressiveness = "aggressive"
        else:
            aggressiveness = "passive"

    return {
        "side": side,
        "recommended_price": recommended_price,
        "size_multiplier": size_multiplier,
        "notes": notes,
        "imbalance": imbalance,
        "aggressiveness": aggressiveness,
        "best_bid": best_bid,
        "best_ask": best_ask,
    }


def detect_sell_pressure(
    order_book: Dict[str, List[Tuple[float, float]]],
    *,
    depth: int = 10,
    reference_price: Optional[float] = None,
    ratio_threshold: float = 1.6,
    wall_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """Detect signs of emergent sell pressure in the order book."""

    bids = _normalise_levels(order_book.get("bids", []), depth)
    asks = _normalise_levels(order_book.get("asks", []), depth)
    bid_volume = sum(q for _, q in bids)
    ask_volume = sum(q for _, q in asks)
    imbalance = compute_order_book_imbalance(order_book, depth=depth)

    sell_pressure = False
    reason: Optional[str] = None
    urgency = "medium"

    if bid_volume <= 0 and ask_volume <= 0:
        return {
            "sell_pressure": False,
            "reason": None,
            "imbalance": imbalance,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "wall_price": None,
            "urgency": "low",
        }

    volume_ratio = (ask_volume / bid_volume) if bid_volume > 0 else float("inf")
    wall = _find_significant_level(
        asks,
        side="ask",
        reference_price=reference_price,
        multiplier=wall_multiplier,
        max_distance=0.002,
    )

    if volume_ratio >= ratio_threshold:
        sell_pressure = True
        reason = f"Ask volume dominates bids ({volume_ratio:.2f}x)"
        if volume_ratio >= ratio_threshold * 1.5:
            urgency = "high"
    if not sell_pressure and wall is not None:
        sell_pressure = True
        wall_price = wall["price"]
        reason = f"Sell wall detected at {wall_price:.6f}"
        distance = None
        if reference_price and reference_price > 0:
            distance = abs(wall_price - reference_price) / reference_price
        if distance is not None and distance < 0.0008:
            urgency = "high"
    else:
        wall_price = wall["price"] if wall is not None else None

    return {
        "sell_pressure": sell_pressure,
        "reason": reason,
        "imbalance": imbalance,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "wall_price": wall["price"] if wall is not None else None,
        "urgency": urgency,
        "volume_ratio": volume_ratio if math.isfinite(volume_ratio) else None,
    }
