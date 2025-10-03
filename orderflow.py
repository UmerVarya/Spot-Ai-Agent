"""Order-flow analysis helpers for the Spot AI trading agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math

import pandas as pd

try:
    from microstructure import compute_order_book_imbalance  # type: ignore
except Exception:  # pragma: no cover - fallback for optional dependency
    def compute_order_book_imbalance(order_book, depth: int = 10) -> float:  # type: ignore
        return float("nan")


OrderBookSide = List[Tuple[float, float]]
OrderBook = Dict[str, OrderBookSide]


@dataclass
class OrderFlowAnalysis:
    """Container for order-flow classification and derived features."""

    state: str = "neutral"
    features: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.state


_PREV_ORDER_BOOK: Dict[str, OrderBook] = {}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _clip_unit(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(max(-1.0, min(1.0, value)))


def _normalise_volume_ratio(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    # Convert raw ratio (e.g. 1.5x average) into roughly [-1, 1]
    # by subtracting one (no change) and bounding.
    return _clip_unit(value - 1.0)


def _clone_order_book(order_book: OrderBook, depth: int = 10) -> OrderBook:
    return {
        "bids": [(float(p), float(q)) for p, q in order_book.get("bids", [])[:depth]],
        "asks": [(float(p), float(q)) for p, q in order_book.get("asks", [])[:depth]],
    }


def _estimate_spoofing_intensity(
    order_book: Optional[OrderBook], symbol: Optional[str], depth: int = 5
) -> float:
    """Detect sudden order-book withdrawals that resemble spoofing."""

    global _PREV_ORDER_BOOK

    key = symbol or "__default__"

    if not order_book:
        _PREV_ORDER_BOOK.pop(key, None)
        return float("nan")

    snapshot = _clone_order_book(order_book, depth=depth)

    prev_snapshot = _PREV_ORDER_BOOK.get(key)

    if prev_snapshot is None:
        _PREV_ORDER_BOOK[key] = snapshot
        return 0.0

    def _side_drop(prev_side: OrderBookSide, curr_side: OrderBookSide) -> float:
        if not prev_side:
            return 0.0
        prev_avg = sum(q for _, q in prev_side) / max(len(prev_side), 1)
        size_threshold = prev_avg * 3.0
        drop_score = 0.0
        curr_lookup = {price: qty for price, qty in curr_side}
        for price, prev_qty in prev_side:
            curr_qty = curr_lookup.get(price, 0.0)
            if prev_qty <= 0:
                continue
            if prev_qty < size_threshold:
                continue
            reduction = max(prev_qty - curr_qty, 0.0)
            if reduction <= 0:
                continue
            drop_score = max(drop_score, reduction / prev_qty)
        return drop_score

    bid_drop = _side_drop(prev_snapshot.get("bids", []), snapshot.get("bids", []))
    ask_drop = _side_drop(prev_snapshot.get("asks", []), snapshot.get("asks", []))

    # Positive intensity implies ask-side liquidity vanished (bullish spoof),
    # negative implies bid-side liquidity vanished (bearish spoof).
    intensity = _clip_unit(ask_drop - bid_drop)

    _PREV_ORDER_BOOK[key] = snapshot

    return intensity


def _get_order_book_from_df(df: pd.DataFrame) -> Optional[OrderBook]:
    ob = df.attrs.get("order_book")
    if isinstance(ob, dict):
        return ob  # type: ignore[return-value]
    return None


def compute_orderflow_features(
    df: pd.DataFrame,
    order_book: Optional[OrderBook] = None,
    symbol: Optional[str] = None,
    depth: int = 5,
) -> Dict[str, float]:
    """Compute continuous microstructure features from candles and depth."""

    features: Dict[str, float] = {
        "order_book_imbalance": float("nan"),
        "cvd": float("nan"),
        "cvd_change": float("nan"),
        "taker_buy_ratio": float("nan"),
        "trade_imbalance": float("nan"),
        "aggressive_trade_rate": float("nan"),
        "spoofing_intensity": float("nan"),
    }

    if df is None or df.empty:
        return features

    order_book = order_book or _get_order_book_from_df(df)
    if order_book:
        obi = compute_order_book_imbalance(order_book, depth=depth)
        if math.isfinite(obi):
            features["order_book_imbalance"] = _clip_unit(obi)
        spoof = _estimate_spoofing_intensity(order_book, symbol=symbol, depth=depth)
        if math.isfinite(spoof):
            features["spoofing_intensity"] = spoof

    cols = set(df.columns)
    has_taker = {"volume", "taker_buy_base", "taker_buy_quote"}.issubset(cols)

    if has_taker:
        taker_buy = pd.to_numeric(df["taker_buy_base"], errors="coerce").fillna(0.0)
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        taker_sell = (volume - taker_buy).clip(lower=0.0)
        net_flow = taker_buy - taker_sell
        total_volume = volume.cumsum().iloc[-1]
        cvd_series = net_flow.cumsum()
        if total_volume > 0:
            features["cvd"] = _clip_unit(_safe_div(cvd_series.iloc[-1], total_volume))
        last_vol = float(volume.iloc[-1]) if not volume.empty else 0.0
        if last_vol > 0:
            features["cvd_change"] = _clip_unit(_safe_div(net_flow.iloc[-1], last_vol))
            buy_ratio = _safe_div(float(taker_buy.iloc[-1]), last_vol)
            features["taker_buy_ratio"] = _clip_unit(2.0 * buy_ratio - 1.0)
            features["trade_imbalance"] = _clip_unit(_safe_div(net_flow.iloc[-1], last_vol))

    if "number_of_trades" in df.columns and len(df) >= 5:
        trades = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0.0)
        recent = trades.tail(20)
        mean_trades = float(recent.mean()) if not recent.empty else 0.0
        last_trades = float(trades.iloc[-1])
        if mean_trades > 0:
            rate = (last_trades - mean_trades) / mean_trades
            features["aggressive_trade_rate"] = _clip_unit(math.tanh(rate))
        elif last_trades > 0:
            features["aggressive_trade_rate"] = _clip_unit(1.0)

    return features


def detect_aggression(
    df: pd.DataFrame,
    order_book: Optional[OrderBook] = None,
    symbol: Optional[str] = None,
    depth: int = 5,
) -> OrderFlowAnalysis:
    """Classify order-flow pressure and expose supporting microstructure features."""

    if df is None or df.empty or len(df) < 5:
        return OrderFlowAnalysis()

    features = compute_orderflow_features(df, order_book=order_book, symbol=symbol, depth=depth)

    recent = df.tail(5)
    avg_volume = float(recent["volume"].mean()) if "volume" in recent else float("nan")
    last_volume = float(recent["volume"].iloc[-1]) if "volume" in recent else float("nan")
    price_change = float(recent["close"].iloc[-1] - recent["open"].iloc[0])

    flow_strength = 0.0
    for key, weight in (
        ("trade_imbalance", 0.45),
        ("order_book_imbalance", 0.35),
        ("cvd_change", 0.2),
    ):
        value = features.get(key)
        if value == value:  # NaN-safe check
            flow_strength += weight * value

    taker_bias = features.get("taker_buy_ratio")
    if taker_bias == taker_bias:
        flow_strength += 0.15 * taker_bias

    trade_rate = features.get("aggressive_trade_rate")
    if trade_rate == trade_rate:
        flow_strength += 0.1 * trade_rate

    spoofing = features.get("spoofing_intensity")
    if spoofing == spoofing:
        flow_strength += 0.1 * spoofing

    volume_factor = 0.0
    if avg_volume == avg_volume and last_volume == last_volume and avg_volume > 0:
        volume_factor = 0.2 * _normalise_volume_ratio(last_volume / avg_volume)

    price_bias = 0.0
    if price_change > 0:
        price_bias = 0.1
    elif price_change < 0:
        price_bias = -0.1

    composite_score = flow_strength + volume_factor + price_bias

    if composite_score > 0.25:
        state = "buyers in control"
    elif composite_score < -0.25:
        state = "sellers in control"
    else:
        state = "neutral"

    return OrderFlowAnalysis(state=state, features=features)
