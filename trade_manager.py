"""
Extended trade management utilities for the Spot AI Super Agent (updated).

This module refactors the original ``trade_manager.py`` to record
additional metadata when closing trades.  Each time a trade is closed
– whether due to an early exit, stop‑loss hit, or take‑profit – the
manager computes exit timestamps, commissions and slippage and passes
those values into the enhanced ``log_trade_result`` function from
``trade_storage``.  The extra information allows downstream analytics to
calculate net PnL, durations and risk metrics.

In this update the module imports the path of the shared active trades
file from ``trade_storage`` instead of constructing its own.  By
centralising these paths, the trading engine and dashboard are
guaranteed to read and write the same JSON file regardless of the
working directory.
"""

from datetime import datetime, timedelta, timezone
import math
import os
import threading
import time
from typing import Any, Dict, List, Tuple, Optional, Union, Mapping

import pandas as pd

from trade_utils import (
    get_price_data,
    calculate_indicators,
    estimate_commission,
    simulate_slippage,
    update_stop_loss_order,
    get_order_book,
)
from macro_sentiment import analyze_macro_sentiment
from notifier import send_email
from log_utils import setup_logger
from trade_storage import (
    log_trade_result,
    store_trade,
    remove_trade,
    load_active_trades,
    save_active_trades,
    is_trade_active,
    MAX_CONCURRENT_TRADES,
)
from rl_policy import RLPositionSizer
from microstructure import plan_execution, detect_sell_pressure
from orderflow import detect_aggression
from trade_constants import (
    TP_ATR_MULTIPLIERS,
    TRAIL_FINAL_ATR,
    TRAIL_INITIAL_ATR,
    TRAIL_LOCK_IN_RATIO,
    TRAIL_TIGHT_ATR,
)
from observability import log_event
from management_explainer import explain_trailing_action

# === Constants ===

# Weighted early-exit signal configuration
EARLY_EXIT_WEIGHTS = {
    "rsi": 0.25,
    "macd": 0.25,
    "ema20": 0.20,
    "ema50": 0.15,
    "vwap": 0.15,
}
EXIT_SCORE_THRESHOLD = 0.6


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


ATR_MULTIPLIER = _env_float("EARLY_EXIT_ATR_MULTIPLIER", 1.0)
MIN_DRAWDOWN_PCT = max(0.0, _env_float("EARLY_EXIT_MIN_DRAWDOWN_PCT", 0.003))
EARLY_EXIT_ATR_TIMEFRAME = os.getenv("EARLY_EXIT_ATR_TIMEFRAME", "1m").strip().lower() or "1m"
_ATR_TIMEFRAME_FREQ = {"1m": "1T", "5m": "5T", "15m": "15T"}
_ATR_TIMEFRAME_WARNING_EMITTED = False
# Require fairly high confidence before exiting on bearish macro signals
MACRO_CONFIDENCE_EXIT_THRESHOLD = 7
# Maximum duration to hold a trade before forcing exit
MAX_HOLDING_TIME = timedelta(hours=6)

logger = setup_logger(__name__)
USE_RL_POSITION_SIZER = False
rl_sizer = RLPositionSizer() if USE_RL_POSITION_SIZER else None

_LIVE_MARKET_LOCK = threading.Lock()
_LIVE_MARKET: Dict[str, Dict[str, float]] = {}
_LAST_MANAGE_TRIGGER: Dict[str, float] = {}
_MANAGE_TRIGGER_COOLDOWN = 2.0
_ACTIVE_TRADES_LOCK = threading.RLock()
_KLINE_ID_LOCK = threading.Lock()
_KLINE_CLOSE_IDS: Dict[str, int] = {}


def generate_post_trade_summary(payload: Mapping[str, Any]) -> str:
    """Create a deterministic textual summary for closed trades."""

    symbol = str(payload.get("symbol", "Unknown")).upper()
    direction = str(payload.get("direction", "")).upper()
    entry = payload.get("entry_price")
    exit_price = payload.get("exit_price")
    outcome = payload.get("outcome") or "Outcome n/a"
    pnl = payload.get("pnl")
    holding = payload.get("holding_minutes")
    reason = payload.get("reason") or payload.get("notes")

    parts = [f"{symbol} {direction}".strip()]
    if entry is not None and exit_price is not None:
        parts.append(f"entry {entry} → exit {exit_price}")
    elif exit_price is not None:
        parts.append(f"exit {exit_price}")
    parts.append(f"result: {outcome}")
    if pnl is not None:
        parts.append(f"PnL {pnl}")
    if holding is not None:
        parts.append(f"held {holding}m")
    if reason:
        parts.append(f"notes: {reason}")

    return "; ".join(str(item) for item in parts if item).strip()


def _register_kline_close_id(symbol: str, close_time: Optional[int]) -> bool:
    if close_time is None:
        return True
    norm = str(symbol or "").upper()
    if not norm:
        return True
    with _KLINE_ID_LOCK:
        last = _KLINE_CLOSE_IDS.get(norm)
        if last == close_time:
            return False
        _KLINE_CLOSE_IDS[norm] = close_time
        return True


def _coerce_to_utc_datetime(value: Union[str, float, int, datetime, None]) -> Optional[datetime]:
    """Attempt to coerce a variety of timestamp formats into naive UTC datetimes."""

    if value is None:
        return None

    dt_value: Optional[datetime] = None
    if isinstance(value, datetime):
        dt_value = value
    elif isinstance(value, (int, float)):
        try:
            dt_value = datetime.utcfromtimestamp(value)
        except Exception:
            return None
    else:
        value_str = str(value).strip()
        if not value_str:
            return None
        try:
            dt_value = datetime.fromisoformat(value_str.replace("Z", "+00:00"))
        except Exception:
            try:
                dt_value = datetime.strptime(value_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

    if dt_value is None:
        return None

    if dt_value.tzinfo is not None:
        dt_value = dt_value.astimezone(timezone.utc).replace(tzinfo=None)

    return dt_value


def _to_float(value: Any) -> Optional[float]:
    """Safely convert ``value`` to ``float`` when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calculate_leg_pnl(
    trade: dict,
    exit_price: float,
    quantity: float,
    fees: float,
    slippage: float,
) -> Tuple[float, float, float]:
    """Return gross/net PnL and notional for a single trade leg."""

    try:
        entry_val = float(trade.get("entry"))
        exit_val = float(exit_price)
        qty_val = float(quantity)
    except Exception:
        return 0.0, 0.0, 0.0
    direction = str(trade.get("direction", "long")).lower()
    if direction == "short":
        gross = (entry_val - exit_val) * qty_val
    else:
        gross = (exit_val - entry_val) * qty_val
    notional = entry_val * qty_val if math.isfinite(entry_val * qty_val) else 0.0
    net = gross
    try:
        net -= float(fees or 0.0)
    except Exception:
        pass
    try:
        net -= float(slippage or 0.0)
    except Exception:
        pass
    return gross, net, notional


def _update_live_market(
    symbol: str,
    *,
    price: Optional[float],
    high: Optional[float],
    low: Optional[float],
    event_time: Optional[float],
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> Dict[str, float]:
    """Store the latest live price snapshot for ``symbol`` and return it."""

    norm = str(symbol or "").upper()
    if not norm:
        return {}
    with _LIVE_MARKET_LOCK:
        snapshot = _LIVE_MARKET.setdefault(norm, {})
        price_val = _to_float(price)
        if price_val is not None:
            snapshot["price"] = price_val
        bid_val = _to_float(bid)
        if bid_val is not None:
            snapshot["bid"] = bid_val
            if price is None:
                snapshot.setdefault("price", bid_val)
        ask_val = _to_float(ask)
        if ask_val is not None:
            snapshot["ask"] = ask_val
            if price is None and "price" not in snapshot and bid_val is not None:
                snapshot["price"] = (bid_val + ask_val) / 2.0
            elif price is None and "price" not in snapshot:
                snapshot["price"] = ask_val
        high_val = _to_float(high)
        if high_val is not None:
            snapshot["high"] = high_val
        low_val = _to_float(low)
        if low_val is not None:
            snapshot["low"] = low_val
        event_val = _to_float(event_time)
        if event_val is not None:
            snapshot["event_time"] = event_val
        return dict(snapshot)


def _get_live_market(symbol: str) -> Dict[str, float]:
    """Return the most recent live snapshot for ``symbol`` if available."""

    norm = str(symbol or "").upper()
    if not norm:
        return {}
    with _LIVE_MARKET_LOCK:
        snapshot = _LIVE_MARKET.get(norm)
        if not snapshot:
            return {}
        return dict(snapshot)


def _request_manage(symbol: str, event_time: Optional[float]) -> None:
    """Trigger ``manage_trades`` for ``symbol`` while respecting cooldown."""

    norm = str(symbol or "").upper()
    if not norm:
        return
    now = float(event_time if event_time is not None else time.time())
    with _LIVE_MARKET_LOCK:
        last = _LAST_MANAGE_TRIGGER.get(norm, 0.0)
        if now - last < _MANAGE_TRIGGER_COOLDOWN:
            return
        _LAST_MANAGE_TRIGGER[norm] = now
    try:
        manage_trades()
    except Exception:
        logger.exception("manage_trades() failed after live trigger for %s", norm)


def _check_live_triggers(
    symbol: str,
    price: Optional[float],
    high: Optional[float],
    low: Optional[float],
) -> None:
    """Check whether live prices trigger stop or target actions."""

    norm = str(symbol or "").upper()
    if not norm:
        return
    effective_price = price
    if effective_price is None:
        effective_price = high if high is not None else low
    if effective_price is None:
        return
    high_for_check = high if high is not None else effective_price
    low_for_check = low if low is not None else effective_price
    if not _ACTIVE_TRADES_LOCK.acquire(blocking=False):
        return
    try:
        trades = load_active_trades()
    except Exception:
        logger.debug(
            "Failed to load active trades for live trigger check: %s", norm, exc_info=True
        )
        return
    finally:
        _ACTIVE_TRADES_LOCK.release()
    triggered = False
    for trade in trades:
        trade_symbol = str(trade.get("symbol", "")).upper()
        if trade_symbol != norm:
            continue
        exit_signals = should_exit_position(
            trade,
            current_price=effective_price,
            recent_high=high_for_check,
            recent_low=low_for_check,
        )
        if exit_signals:
            triggered = True
            break
    if not triggered:
        return
    _request_manage(norm, time.time())


def process_live_kline(symbol: str, interval: str, payload: Mapping[str, Any]) -> None:
    """Handle live kline updates and trigger trade management when needed."""

    price = _to_float(payload.get("c"))
    high = _to_float(payload.get("h"))
    low = _to_float(payload.get("l"))
    event_time = _to_float(payload.get("T") or payload.get("t"))
    close_time_raw = payload.get("T") or payload.get("t")
    try:
        close_time = int(close_time_raw) if close_time_raw is not None else None
    except (TypeError, ValueError):
        close_time = None
    is_closed = bool(payload.get("x"))
    is_unique_close = True
    if is_closed:
        is_unique_close = _register_kline_close_id(symbol, close_time)
    _update_live_market(symbol, price=price, high=high, low=low, event_time=event_time)
    if is_closed and not is_unique_close:
        log_event(
            logger,
            "duplicate_bar_close",
            symbol=str(symbol).upper(),
            close_time=close_time,
        )
        return
    _check_live_triggers(symbol, price, high, low)
    if is_closed:
        log_event(
            logger,
            "bar_close_processed",
            symbol=str(symbol).upper(),
            interval=interval,
            close_time=close_time,
            price=price,
        )


def process_live_ticker(symbol: str, payload: Mapping[str, Any]) -> None:
    """Handle ticker updates for rapid stop/target evaluation."""

    price = _to_float(
        payload.get("c")
        or payload.get("C")
        or payload.get("lastPrice")
        or payload.get("close")
    )
    event_time = _to_float(payload.get("E") or payload.get("eventTime"))
    _update_live_market(symbol, price=price, high=None, low=None, event_time=event_time)
    _check_live_triggers(symbol, price, None, None)


def process_book_ticker(symbol: str, payload: Mapping[str, Any]) -> None:
    """Handle book ticker updates to keep bid/ask snapshots fresh."""

    bid = _to_float(payload.get("b") or payload.get("B") or payload.get("bidPrice"))
    ask = _to_float(payload.get("a") or payload.get("A") or payload.get("askPrice"))
    event_time = _to_float(payload.get("E") or payload.get("eventTime"))
    price = bid if bid is not None else ask
    _update_live_market(
        symbol,
        price=price,
        high=None,
        low=None,
        event_time=event_time,
        bid=bid,
        ask=ask,
    )
    if price is None:
        return
    high_for_check = bid if bid is not None else ask
    low_for_check = bid if bid is not None else ask
    _check_live_triggers(symbol, price, high_for_check, low_for_check)


def process_user_stream_event(payload: Mapping[str, Any]) -> None:
    """Handle Binance user data stream events (fills, account updates)."""

    if not isinstance(payload, Mapping):
        return
    event_type = str(payload.get("e") or payload.get("eventType") or "").lower()
    symbol = str(payload.get("s") or payload.get("symbol") or "").upper()
    event_time = _to_float(payload.get("E") or payload.get("eventTime"))
    should_manage = False
    last_price: Optional[float] = None

    if event_type == "executionreport":
        last_price = _to_float(
            payload.get("L")
            or payload.get("lPrice")
            or payload.get("ap")
            or payload.get("p")
        )
        status = str(payload.get("X") or payload.get("orderStatus") or "").upper()
        if status in {
            "FILLED",
            "PARTIALLY_FILLED",
            "CANCELED",
            "EXPIRED",
            "REJECTED",
            "EXPIRED_IN_MATCH",
        }:
            should_manage = True
    elif event_type == "liststatus":
        should_manage = True
        if not symbol:
            symbol = str(payload.get("s") or payload.get("symbol"))
    elif event_type in {"balanceupdate", "outboundaccountposition"}:
        # Balance change does not mandate trade management but we keep track
        should_manage = False

    if symbol and last_price is not None:
        _update_live_market(symbol, price=last_price, high=None, low=None, event_time=event_time)

    if should_manage and symbol:
        _request_manage(symbol, event_time)


def _record_partial_exit(
    trade: dict,
    label: str,
    *,
    exit_price: float,
    quantity: float,
    fees: float,
    slippage: float,
    exit_time: str,
) -> None:
    """Aggregate realised results from a partial exit on the trade object."""

    try:
        fee_val = float(fees or 0.0)
    except Exception:
        fee_val = 0.0
    try:
        slippage_val = float(slippage or 0.0)
    except Exception:
        slippage_val = 0.0
    gross_leg, net_leg, _ = _calculate_leg_pnl(
        trade, exit_price, quantity, fee_val, slippage_val
    )
    trade["realized_pnl"] = trade.get("realized_pnl", 0.0) + net_leg
    trade["realized_gross_pnl"] = trade.get("realized_gross_pnl", 0.0) + gross_leg
    trade["realized_fees"] = trade.get("realized_fees", 0.0) + fee_val
    trade["realized_slippage"] = trade.get("realized_slippage", 0.0) + slippage_val
    trade["last_partial_exit"] = exit_time
    trade["last_partial_pnl"] = net_leg
    try:
        quantity_val = float(quantity)
    except Exception:
        quantity_val = None
    try:
        entry_val = float(trade.get("entry"))
    except Exception:
        entry_val = None

    if label == "tp1":
        trade["tp1_partial"] = True
        trade["tp1_exit_price"] = exit_price
        trade["pnl_tp1"] = net_leg
        if quantity_val is not None:
            trade["size_tp1"] = quantity_val
        if entry_val is not None and quantity_val is not None:
            trade["notional_tp1"] = entry_val * quantity_val
    elif label == "tp2":
        trade["tp2_partial"] = True
        trade["tp2_exit_price"] = exit_price
        trade["pnl_tp2"] = net_leg
        if quantity_val is not None:
            trade["size_tp2"] = quantity_val
        if entry_val is not None and quantity_val is not None:
            trade["notional_tp2"] = entry_val * quantity_val
    else:
        trade["trail_partial_pnl"] = trade.get("trail_partial_pnl", 0.0) + net_leg
        trade["trail_partial_count"] = trade.get("trail_partial_count", 0) + 1
        trade["last_trail_partial_pnl"] = net_leg


def _finalize_trade_result(
    trade: dict,
    *,
    exit_price: float,
    quantity: float,
    fees: float,
    slippage: float,
    outcome: str,
) -> tuple[float, float]:
    """Combine partial leg results with the final exit leg."""

    try:
        fee_val = float(fees or 0.0)
    except Exception:
        fee_val = 0.0
    try:
        slippage_val = float(slippage or 0.0)
    except Exception:
        slippage_val = 0.0
    final_gross, final_net, _ = _calculate_leg_pnl(
        trade, exit_price, quantity, fee_val, slippage_val
    )
    trade["final_leg_pnl"] = final_net
    trade["final_leg_gross_pnl"] = final_gross
    trade["final_exit_size"] = quantity
    total_gross = trade.get("realized_gross_pnl", 0.0) + final_gross
    total_pnl = trade.get("realized_pnl", 0.0) + final_net
    total_fees = trade.get("realized_fees", 0.0) + fee_val
    total_slippage = trade.get("realized_slippage", 0.0) + slippage_val
    entry_val = _to_float(trade.get("entry"))
    initial_qty = _to_float(trade.get("initial_size"))
    if initial_qty is None:
        initial_qty = _to_float(trade.get("position_size"))
    if initial_qty in (None, 0.0):
        raw_size = _to_float(trade.get("size"))
        if raw_size is not None and entry_val not in (None, 0.0):
            try:
                initial_qty = float(raw_size) / float(entry_val)
            except Exception:
                initial_qty = None
    if initial_qty is None:
        initial_qty = quantity
    notional = None
    if entry_val is not None and initial_qty is not None:
        try:
            notional = float(entry_val) * float(initial_qty)
        except Exception:
            notional = None
    trade["realized_pnl"] = total_pnl
    trade["total_pnl"] = total_pnl
    trade["net_pnl"] = total_pnl
    trade["realized_gross_pnl"] = total_gross
    trade["gross_pnl"] = total_gross
    trade["realized_fees"] = total_fees
    trade["total_fees"] = total_fees
    trade["realized_slippage"] = total_slippage
    trade["total_slippage"] = total_slippage
    if notional is not None:
        trade["notional"] = notional
        if notional not in (0, 0.0):
            trade["pnl_pct"] = (total_pnl / notional) * 100.0
    trade["tp1_partial"] = bool(trade.get("tp1_partial"))
    trade["tp2_partial"] = bool(trade.get("tp2_partial"))
    outcome_token = str(outcome or trade.get("outcome", "")).lower()
    if "tp3" in outcome_token and "_partial" not in outcome_token:
        trade["tp3_reached"] = True
        trade["pnl_tp3"] = trade.get("pnl_tp3", 0.0) + final_net
        try:
            qty_val = float(quantity)
        except Exception:
            qty_val = None
        if qty_val is not None:
            trade["size_tp3"] = qty_val
        if entry_val is not None and qty_val is not None:
            trade["notional_tp3"] = entry_val * qty_val

    if notional and notional > 0:
        if abs(total_pnl) > notional * 0.2:
            logger.warning(
                "Suspicious PnL magnitude; check slippage/fees",
                extra={
                    "symbol": trade.get("symbol"),
                    "notional": notional,
                    "net_pnl": total_pnl,
                    "gross_pnl": total_gross,
                    "fees": total_fees,
                    "slippage": total_slippage,
                },
            )
        if total_slippage > notional * 0.02:
            logger.warning(
                "Unusually large slippage on exit",
                extra={
                    "symbol": trade.get("symbol"),
                    "notional": notional,
                    "slippage": total_slippage,
                    "net_pnl": total_pnl,
                },
            )
    return total_fees, total_slippage


def _log_trade_closed(trade: dict, *, reason: str, outcome: str) -> None:
    """Emit a structured log when a trade leaves the active set."""

    payload = {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "entry_price": _to_float(trade.get("entry")),
        "exit_price": _to_float(trade.get("exit_price")),
        "gross_pnl": trade.get("gross_pnl"),
        "net_pnl": trade.get("net_pnl", trade.get("realized_pnl")),
        "fees_usdt": trade.get("total_fees"),
        "slippage_usdt": trade.get("total_slippage"),
        "reason": reason,
        "outcome": outcome,
    }
    logger.info("Closed trade and removed from active_trades.json", extra=payload)


def execute_exit_trade(
    trade: dict,
    *,
    exit_price: float,
    reason: str,
    outcome: str,
    quantity: Optional[float] = None,
    exit_time: Optional[str] = None,
    fees: Optional[float] = None,
    slippage: Optional[float] = None,
    maker: bool = False,
) -> tuple[float, float, float]:
    """Finalize and log a closing trade leg."""

    symbol = trade.get("symbol")
    direction = str(trade.get("direction", "long")).lower()
    qty_val = _to_float(quantity if quantity is not None else trade.get("position_size", 0))
    if qty_val is None or qty_val < 0:
        qty_val = 0.0
    exit_price_val = _to_float(exit_price)
    if exit_price_val is None:
        exit_price_val = 0.0
    requested_exit_price = exit_price_val
    if exit_time is None:
        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    slippage_per_unit = 0.0
    if slippage is None:
        slip_price = simulate_slippage(exit_price_val, direction=direction)
        fill_price = slip_price
        slippage_per_unit = abs(slip_price - requested_exit_price)
        slippage = slippage_per_unit * qty_val
    else:
        fill_price = exit_price_val
        try:
            slippage = float(slippage)
        except Exception:
            slippage = 0.0
        if qty_val > 0:
            slippage_per_unit = abs(slippage) / qty_val
    if fees is None:
        commission_rate = estimate_commission(symbol, quantity=qty_val, maker=maker)
        fees = fill_price * qty_val * commission_rate

    trade["exit_price"] = exit_price_val
    trade["requested_exit_price"] = requested_exit_price
    trade["exit_time"] = exit_time
    trade["outcome"] = outcome
    trade["exit_reason"] = reason
    trade["slippage_per_unit"] = slippage_per_unit
    trade["fill_price"] = fill_price

    try:
        current_qty = float(trade.get("position_size", 0.0))
    except Exception:
        current_qty = 0.0
    remaining = max(0.0, current_qty - qty_val)
    trade["position_size"] = remaining
    entry_val = _to_float(trade.get("entry"))
    if entry_val is not None:
        trade["size"] = remaining * entry_val

    total_fees, total_slippage = _finalize_trade_result(
        trade,
        exit_price=exit_price_val,
        quantity=qty_val,
        fees=fees,
        slippage=slippage,
        outcome=outcome,
    )
    log_trade_result(
        trade,
        outcome=outcome,
        exit_price=exit_price_val,
        exit_time=exit_time,
        fees=total_fees,
        slippage=total_slippage,
    )
    gross_pnl = trade.get("gross_pnl")
    net_pnl = trade.get("net_pnl", trade.get("realized_pnl"))
    logger.info(
        "Trade exit summary",
        extra={
            "symbol": symbol,
            "entry_price": trade.get("entry"),
            "exit_price": fill_price,
            "requested_exit_price": requested_exit_price,
            "qty": qty_val,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "fees_usdt": total_fees,
            "slippage_usdt": total_slippage,
            "outcome": outcome,
        },
    )
    logger.info(
        "✅ Position closed for %s at %.6f (%s)",
        symbol or "unknown",
        exit_price_val,
        reason,
    )
    entry_dt = _coerce_to_utc_datetime(trade.get("entry_time"))
    exit_dt = _coerce_to_utc_datetime(exit_time)
    holding_minutes = None
    if entry_dt and exit_dt:
        holding_minutes = round((exit_dt - entry_dt).total_seconds() / 60.0, 2)
    summary_payload = {
        "symbol": symbol,
        "direction": trade.get("direction"),
        "entry_price": trade.get("entry"),
        "exit_price": exit_price_val,
        "outcome": outcome,
        "reason": reason,
        "pnl": trade.get("realized_pnl"),
        "holding_minutes": holding_minutes,
        "notes": trade.get("notes") or trade.get("narrative"),
    }
    post_summary = generate_post_trade_summary(summary_payload)
    if post_summary:
        logger.info("[Post-trade Summary] %s", post_summary.replace("\n", " "))
    return qty_val, total_fees, total_slippage

def _update_rl(trade: dict, exit_price: float) -> None:
    """Update RL position sizer based on trade outcome."""
    if not USE_RL_POSITION_SIZER or rl_sizer is None:
        return
    try:
        entry = float(trade.get('entry', 0))
        state = trade.get('rl_state', 'neutral')
        action = trade.get('rl_multiplier')
        if entry is None or action is None:
            return
        reward = (exit_price - entry) / entry
        if reward < -0.25:
            reward -= 1.0
        rl_sizer.update(state, action, reward)
    except Exception:
        pass


def _get_trade_atr(trade: dict) -> Optional[float]:
    """Return the ATR value associated with ``trade`` when available."""

    atr_entry = _to_float(trade.get("atr_at_entry"))
    if atr_entry is not None and atr_entry > 0:
        return atr_entry

    entry_price = _to_float(trade.get("entry"))
    tp1_price = _to_float(trade.get("tp1"))
    if entry_price is None or tp1_price is None:
        return None

    base_multiplier = TP_ATR_MULTIPLIERS[0] if TP_ATR_MULTIPLIERS else None
    if base_multiplier is None or base_multiplier <= 0:
        return None

    direction = str(trade.get("direction", "long")).lower()
    if direction == "short":
        distance = entry_price - tp1_price
    else:
        distance = tp1_price - entry_price
    if distance is None or distance <= 0:
        return None

    return distance / base_multiplier


def _activate_trailing_mode(
    trade: dict,
    tp_price: float,
    *,
    current_price: Optional[float],
) -> Optional[float]:
    """Enable trailing-stop management and return the locked-in stop price."""

    entry_price = _to_float(trade.get("entry"))
    if entry_price is None or tp_price is None:
        return None

    direction = str(trade.get("direction", "long")).lower()
    if direction == "short":
        move = entry_price - tp_price
        if move <= 0:
            return None
        lock_price = entry_price - move * TRAIL_LOCK_IN_RATIO
    else:
        move = tp_price - entry_price
        if move <= 0:
            return None
        lock_price = entry_price + move * TRAIL_LOCK_IN_RATIO

    lock_price = round(lock_price, 6)
    trade["trailing_active"] = True
    trade["profit_riding"] = True
    trade["locked_profit_price"] = lock_price

    reference = tp_price
    if current_price is not None:
        if direction == "short":
            reference = min(current_price, tp_price)
        else:
            reference = max(current_price, tp_price)

    existing_anchor = _to_float(trade.get("trail_high"))
    if direction == "short":
        if existing_anchor is None:
            trade["trail_high"] = reference
        else:
            trade["trail_high"] = min(existing_anchor, reference)
    else:
        if existing_anchor is None:
            trade["trail_high"] = reference
        else:
            trade["trail_high"] = max(existing_anchor, reference)

    trade["trail_multiplier"] = TRAIL_INITIAL_ATR
    trade.pop("next_trail_tp", None)
    trade.pop("trail_tp_pct", None)
    trade.pop("last_trail_partial_pnl", None)

    return lock_price


def _adjust_trailing_multiplier(trade: dict, new_multiplier: float) -> bool:
    """Reduce the trailing ATR multiplier when the new value is tighter."""

    if new_multiplier is None or new_multiplier <= 0:
        return False
    current = _to_float(trade.get("trail_multiplier"))
    if current is None or not math.isfinite(current) or new_multiplier < current - 1e-9:
        trade["trail_multiplier"] = new_multiplier
        return True
    return False


def _update_stop_loss(trade: dict, new_sl: float) -> None:
    """Update the trade's stop-loss and mirror the change on Binance."""
    symbol = trade.get("symbol")
    old_sl = trade.get("sl")
    try:
        qty = float(trade.get("position_size", 1))
    except Exception:
        qty = 1.0
    order_id = trade.get("sl_order_id")
    status = trade.get("status", {})
    tp_price = None
    strategy = str(trade.get("take_profit_strategy") or "").lower()
    trailing_mode = bool(trade.get("trailing_active"))
    if strategy != "atr_trailing":
        if not status.get("tp1"):
            tp_price = trade.get("tp1")
        elif not status.get("tp2"):
            tp_price = trade.get("tp2")
        elif not status.get("tp3"):
            tp_price = trade.get("tp3")
    elif not trailing_mode:
        # ATR trailing strategy keeps a dynamic stop without resting TP orders.
        tp_price = None
    if symbol:
        new_id = update_stop_loss_order(symbol, qty, new_sl, order_id, tp_price)
        if new_id is not None:
            trade["sl_order_id"] = new_id
    trade["sl"] = new_sl
    logger.info("%s SL updated from %s to %s", symbol, old_sl, new_sl)
    send_email(
        f"SL Updated: {symbol}",
        {
            "symbol": symbol,
            "old_sl": old_sl,
            "new_sl": new_sl,
        },
    )


def _persist_active_snapshot(
    updated_trades: List[dict],
    active_trades: Optional[List[dict]] = None,
    index: int = -1,
    *,
    include_current: bool = True,
) -> None:
    """Persist a best-effort snapshot of active trades immediately.

    ``manage_trades`` normally batches calls to :func:`save_active_trades`
    until the end of the loop.  If the process crashes after a trade result
    has been written to the history log but before the batch save completes,
    the on-disk active trades JSON can become stale.  To reduce that window we
    flush the current in-memory view whenever a trade mutates or exits.
    """

    try:
        if active_trades is None:
            snapshot = [trade for trade in updated_trades if trade is not None]
        else:
            snapshot = list(updated_trades)
            if include_current and 0 <= index < len(active_trades):
                snapshot.append(active_trades[index])
            if index + 1 < len(active_trades):
                snapshot.extend(active_trades[index + 1 :])
            snapshot = [trade for trade in snapshot if trade is not None]
        save_active_trades(snapshot)
    except Exception:
        logger.exception("Failed to persist active trades snapshot")


def _append_trailing_explanation(
    trade: dict,
    event: str,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Attach an LLM generated explanation for a trailing-stop action."""

    try:
        explanation = explain_trailing_action(event, trade, context or {})
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Failed to build trailing explanation for %s", event, exc_info=True)
        explanation = ""

    if not explanation:
        return

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    safe_context: Dict[str, Any] = {}
    if context:
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_context[key] = value
            else:
                safe_context[key] = repr(value)

    record: Dict[str, Any] = {
        "event": event,
        "ts": timestamp,
        "message": explanation,
    }
    if safe_context:
        record["context"] = safe_context

    history = trade.get("management_explanations")
    if not isinstance(history, list):
        history = []
    history.append(record)
    trade["management_explanations"] = history[-10:]

    log_event(
        logger,
        "trailing_management_explanation",
        symbol=trade.get("symbol"),
        event=event,
        explanation=explanation,
        context=safe_context,
    )


def create_new_trade(
    trade: dict,
    *,
    stop_price: Optional[float] = None,
    target_price: Optional[float] = None,
    auction_state: Optional[str] = None,
    lvn_entry_level: Optional[float] = None,
    orderflow_analysis: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Add a new trade to persistent storage if not already active.

    Returns
    -------
    bool
        ``True`` if the trade was stored, ``False`` if a trade with the
        same symbol was already active.
    """
    def _assign_float(key: str, value: Any) -> None:
        number = _to_float(value)
        if number is None or not math.isfinite(number):
            return
        trade[key] = number

    if auction_state is not None:
        trade["auction_state"] = str(auction_state)
    elif "auction_state" in trade and trade.get("auction_state") is not None:
        trade["auction_state"] = str(trade["auction_state"])

    if lvn_entry_level is not None:
        _assign_float("lvn_entry_level", lvn_entry_level)
    elif "lvn_entry_level" in trade:
        _assign_float("lvn_entry_level", trade.get("lvn_entry_level"))

    if stop_price is not None:
        _assign_float("lvn_stop", stop_price)
    elif "lvn_stop" in trade:
        _assign_float("lvn_stop", trade.get("lvn_stop"))

    if target_price is not None:
        _assign_float("poc_target", target_price)
    elif "poc_target" in trade:
        _assign_float("poc_target", trade.get("poc_target"))

    snapshot = orderflow_analysis
    if snapshot is None and isinstance(trade.get("orderflow_analysis"), Mapping):
        snapshot = trade.get("orderflow_analysis")  # type: ignore[assignment]
    if isinstance(snapshot, Mapping):
        state = str(snapshot.get("state", "neutral"))
        features: Dict[str, Optional[float]] = {}
        raw_features = snapshot.get("features", {})
        if isinstance(raw_features, Mapping):
            for key, value in raw_features.items():
                clean = _to_float(value)
                if clean is None or not math.isfinite(clean):
                    features[str(key)] = None
                else:
                    features[str(key)] = clean
        trade["orderflow_analysis"] = {"state": state, "features": features}

    symbol = trade.get("symbol")
    if symbol and is_trade_active(symbol):
        logger.info("Trade for %s already active; skipping new entry.", symbol)
        return False
    current_trades = load_active_trades()
    if len(current_trades) >= MAX_CONCURRENT_TRADES:
        logger.info(
            "Maximum concurrent trades (%d) already active; skipping %s.",
            MAX_CONCURRENT_TRADES,
            symbol or "unknown",
        )
        return False
    stored = store_trade(trade)
    if stored:
        entry_price = _to_float(trade.get("entry"))
        qty_val = _to_float(trade.get("position_size"))
        notional_val = _to_float(trade.get("size"))
        if notional_val is None and entry_price is not None and qty_val is not None:
            notional_val = entry_price * qty_val
        payload = {
            "symbol": symbol,
            "entry_price": entry_price,
            "quantity": qty_val,
            "notional_usd": notional_val,
            "atr_at_entry": _to_float(trade.get("atr_at_entry")),
            "sl": _to_float(trade.get("sl")),
            "tp1": _to_float(trade.get("tp1")),
            "tp2": _to_float(trade.get("tp2")),
            "tp3": _to_float(trade.get("tp3")),
            "atr_guard_multiplier": ATR_MULTIPLIER,
            "atr_guard_min_drawdown_pct": MIN_DRAWDOWN_PCT,
            "atr_guard_timeframe": EARLY_EXIT_ATR_TIMEFRAME,
        }
        logger.info("New trade entry recorded", extra=payload)
    return stored


def _latest_indicator_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "iloc"):
        try:
            return float(value.iloc[-1])
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _warn_atr_timeframe_once(message: str) -> None:
    global _ATR_TIMEFRAME_WARNING_EMITTED
    if _ATR_TIMEFRAME_WARNING_EMITTED:
        return
    logger.warning(message)
    _ATR_TIMEFRAME_WARNING_EMITTED = True


def _compute_atr_for_guard(price_data: pd.DataFrame, indicators: Any) -> Optional[float]:
    atr_value = _latest_indicator_value(indicators.get("atr")) if isinstance(indicators, Mapping) or hasattr(indicators, "get") else None
    freq = _ATR_TIMEFRAME_FREQ.get(EARLY_EXIT_ATR_TIMEFRAME)
    if EARLY_EXIT_ATR_TIMEFRAME == "1m" or freq is None:
        if freq is None and EARLY_EXIT_ATR_TIMEFRAME != "1m":
            _warn_atr_timeframe_once(
                f"Unsupported EARLY_EXIT_ATR_TIMEFRAME={EARLY_EXIT_ATR_TIMEFRAME}; falling back to 1m",
            )
        return atr_value
    if not isinstance(price_data.index, pd.DatetimeIndex):
        _warn_atr_timeframe_once(
            "Price data missing datetime index; ATR guard reverting to 1m timeframe",
        )
        return atr_value
    try:
        resampled = (
            price_data.resample(freq)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
    except Exception:
        _warn_atr_timeframe_once(
            f"Failed to resample price data for ATR timeframe {EARLY_EXIT_ATR_TIMEFRAME}; using 1m",
        )
        return atr_value
    if resampled.empty:
        return atr_value
    try:
        ht_indicators = calculate_indicators(resampled)
    except Exception:
        _warn_atr_timeframe_once(
            f"Unable to compute ATR for timeframe {EARLY_EXIT_ATR_TIMEFRAME}; using 1m",
        )
        return atr_value
    ht_atr = _latest_indicator_value(ht_indicators.get("atr")) if hasattr(ht_indicators, "get") else None
    if ht_atr is None:
        _warn_atr_timeframe_once(
            f"ATR unavailable for timeframe {EARLY_EXIT_ATR_TIMEFRAME}; using 1m",
        )
        return atr_value
    return ht_atr


def should_exit_early(
    trade: dict, observed_price: Optional[float], price_data
) -> Tuple[bool, Optional[str]]:
    """Determine if a trade should exit early based on price and indicators.

    ``observed_price`` represents the most adverse price seen during the
    polling interval (e.g., the candle low for long trades) so that sharp
    wicks can still trigger an exit. When ``None`` the ATR-based drawdown
    guard is bypassed for that evaluation cycle.
    """
    entry = trade.get('entry')
    direction = trade.get('direction')
    if entry is None or direction is None:
        return False, None

    indicators = calculate_indicators(price_data)
    current_price = price_data['close'].iloc[-1]
    atr_value = _compute_atr_for_guard(price_data, indicators)
    direction_token = str(direction).lower()
    entry_price = _to_float(entry)
    observed_val = _to_float(observed_price) if observed_price is not None else None
    drawdown_usd = 0.0
    drawdown_pct = 0.0
    atr_pct = 0.0
    atr_reason = None
    atr_triggered = False
    if (
        direction_token == "long"
        and entry_price is not None
        and observed_val is not None
    ):
        drawdown_usd = max(0.0, entry_price - observed_val)
        if entry_price > 0:
            drawdown_pct = drawdown_usd / entry_price
            if atr_value is not None and atr_value > 0:
                atr_pct = atr_value / entry_price
        if (
            atr_value is not None
            and atr_value > 0
            and atr_pct > 0
            and drawdown_pct >= MIN_DRAWDOWN_PCT
            and drawdown_pct >= atr_pct * ATR_MULTIPLIER
        ):
            atr_triggered = True
            atr_reason = (
                f"ATR drawdown {drawdown_pct:.3%} exceeds {ATR_MULTIPLIER:.2f}×ATR ({atr_pct:.3%})"
            )
    logger.debug(
        "ATR early-exit check",
        extra={
            "symbol": trade.get("symbol"),
            "entry": entry_price,
            "observed": observed_val,
            "drawdown_usd": drawdown_usd,
            "drawdown_pct": drawdown_pct,
            "atr": atr_value,
            "atr_pct": atr_pct,
            "atr_mult": ATR_MULTIPLIER,
            "min_dd_pct": MIN_DRAWDOWN_PCT,
            "triggered": atr_triggered,
        },
    )
    if atr_triggered and atr_reason:
        return True, atr_reason

    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd", 0)
    ema20 = indicators.get("ema_20", current_price)
    ema50 = indicators.get("ema_50", current_price)
    vwap = indicators.get("vwap") or indicators.get("vwma")

    if hasattr(rsi, 'iloc'):
        rsi = rsi.iloc[-1]
    if hasattr(macd_hist, 'iloc'):
        macd_hist = macd_hist.iloc[-1]
    if hasattr(ema20, 'iloc'):
        ema20 = ema20.iloc[-1]
    if hasattr(ema50, 'iloc'):
        ema50 = ema50.iloc[-1]
    if vwap is not None and hasattr(vwap, 'iloc'):
        vwap = vwap.iloc[-1]

    score = 0.0
    reasons = []
    if direction_token == "long":
        if rsi < 45:
            score += EARLY_EXIT_WEIGHTS["rsi"]
            reasons.append(f"RSI {rsi:.2f}")
        if macd_hist < 0:
            score += EARLY_EXIT_WEIGHTS["macd"]
            reasons.append(f"MACD {macd_hist:.4f}")
        if current_price < ema20:
            score += EARLY_EXIT_WEIGHTS["ema20"]
            reasons.append("Price below EMA20")
        if current_price < ema50:
            score += EARLY_EXIT_WEIGHTS["ema50"]
            reasons.append("Price below EMA50")
        if vwap is not None and current_price < vwap:
            score += EARLY_EXIT_WEIGHTS["vwap"]
            reasons.append("Close below VWAP")
    if score >= EXIT_SCORE_THRESHOLD:
        return True, f"Bearish signals (score {score:.2f}): {', '.join(reasons)}"

    macro = analyze_macro_sentiment()
    if macro.get('bias') == "bearish" and macro.get('confidence', 0) >= MACRO_CONFIDENCE_EXIT_THRESHOLD:
        return True, f"Macro sentiment shifted to bearish (Confidence: {macro.get('confidence')})"
    return False, None


def should_exit_position(
    trade: dict,
    *,
    current_price: Optional[float],
    recent_high: Optional[float],
    recent_low: Optional[float],
) -> List[dict]:
    """Return ordered exit signals triggered by TP/SL levels."""

    signals: List[dict] = []
    direction = str(trade.get("direction", "long")).lower()
    if direction != "long":
        return signals

    status = trade.setdefault("status", {})

    high_price = recent_high
    low_price = recent_low
    if high_price is None:
        high_price = current_price
    if low_price is None:
        low_price = current_price

    sl_value = _to_float(trade.get("sl"))
    if sl_value is not None:
        trade["sl"] = sl_value
    if sl_value is not None and low_price is not None and low_price <= sl_value:
        signals.append({"type": "stop_loss", "price": sl_value})
        return signals

    tp1_value = _to_float(trade.get("tp1"))
    tp2_value = _to_float(trade.get("tp2"))
    tp3_value = _to_float(trade.get("tp3"))
    if tp1_value is not None:
        trade["tp1"] = tp1_value
    if tp2_value is not None:
        trade["tp2"] = tp2_value
    if tp3_value is not None:
        trade["tp3"] = tp3_value

    high_enough = high_price is not None and tp1_value is not None and high_price >= tp1_value
    tp1_hit = bool(status.get("tp1"))
    tp2_hit = bool(status.get("tp2"))
    tp3_hit = bool(status.get("tp3"))

    if not tp1_hit and high_enough:
        signals.append({"type": "tp1", "price": tp1_value})
        tp1_hit = True

    if tp1_hit and not tp2_hit and tp2_value is not None:
        high_for_tp2 = high_price is not None and high_price >= tp2_value
        if high_for_tp2:
            signals.append({"type": "tp2", "price": tp2_value})
            tp2_hit = True

    if tp2_hit and not tp3_hit and tp3_value is not None:
        high_for_tp3 = high_price is not None and high_price >= tp3_value
        if high_for_tp3:
            signals.append({"type": "tp3", "price": tp3_value})

    return signals


def _manage_trades_body() -> None:
    raw_active = load_active_trades()
    if isinstance(raw_active, dict):
        logger.warning(
            "Active trades snapshot was a mapping; normalising to list layout",
        )
        active_trades = [value for value in raw_active.values() if isinstance(value, dict)]
        save_active_trades(active_trades)
    elif isinstance(raw_active, list):
        active_trades = list(raw_active)
    else:
        active_trades = []
    updated_trades: List[dict] = []
    for index, trade in enumerate(active_trades):
        symbol = trade.get("symbol")
        # Ensure original size (quantity) is tracked for partial profit-taking
        if "initial_size" not in trade:
            try:
                trade["initial_size"] = float(trade.get("position_size", 1))
            except Exception:
                trade["initial_size"] = 1.0
        trade.setdefault("realized_pnl", 0.0)
        trade.setdefault("realized_fees", 0.0)
        trade.setdefault("realized_slippage", 0.0)
        trade.setdefault("tp1_partial", False)
        trade.setdefault("tp2_partial", False)
        trade.setdefault("trailing_active", False)
        if trade.get("trailing_active") and not _to_float(trade.get("trail_multiplier")):
            trade["trail_multiplier"] = TRAIL_INITIAL_ATR
        fallback_exit_price: Optional[float] = None
        for price_key in ("last_price", "current_price", "entry"):
            price_val = trade.get(price_key)
            if price_val is None:
                continue
            try:
                fallback_exit_price = float(price_val)
                break
            except Exception:
                continue

        live_snapshot = _get_live_market(symbol)
        live_price = _to_float(live_snapshot.get("price"))
        live_high = _to_float(live_snapshot.get("high"))
        live_low = _to_float(live_snapshot.get("low"))
        live_bid = _to_float(live_snapshot.get("bid"))
        live_ask = _to_float(live_snapshot.get("ask"))
        if live_bid is not None:
            fallback_exit_price = live_bid
        elif live_price is not None:
            fallback_exit_price = live_price
        elif live_ask is not None:
            fallback_exit_price = live_ask

        direction = str(trade.get('direction', 'long')).lower()
        entry = _to_float(trade.get('entry'))
        if entry is not None:
            trade['entry'] = entry
        sl = _to_float(trade.get('sl'))
        if sl is not None:
            trade['sl'] = sl
        tp1 = _to_float(trade.get('tp1'))
        if tp1 is not None:
            trade['tp1'] = tp1
        tp2 = _to_float(trade.get('tp2'))
        if tp2 is not None:
            trade['tp2'] = tp2
        tp3 = _to_float(trade.get('tp3'))
        if tp3 is not None:
            trade['tp3'] = tp3
        status_flags = trade.setdefault('status', {})
        status_flags.setdefault('flow_break_even', False)
        actions = []
        # Time-based exit: close trades exceeding the maximum holding duration
        primary_entry_time = trade.get('entry_time')
        entry_sources = []
        if 'entry_time' in trade:
            entry_sources.append(('entry_time', primary_entry_time))
        for candidate_key in ("timestamp", "open_time", "created_at", "last_update"):
            candidate_value = trade.get(candidate_key)
            if candidate_value is not None:
                entry_sources.append((candidate_key, candidate_value))

        entry_dt = None
        entry_source_used: Optional[str] = None
        for source_name, source_value in entry_sources:
            entry_dt = _coerce_to_utc_datetime(source_value)
            if entry_dt is not None:
                entry_source_used = source_name
                break

        if entry_dt is None:
            logger.warning(
                "%s missing valid entry_time; assigning current time for hold duration tracking.",
                symbol,
            )
            entry_dt = datetime.utcnow()
            trade['entry_time'] = entry_dt.strftime("%Y-%m-%d %H:%M:%S")
        elif entry_source_used != 'entry_time':
            logger.warning(
                "%s entry_time unavailable; using %s for timeout check.",
                symbol,
                entry_source_used,
            )
            if not primary_entry_time:
                trade['entry_time'] = entry_dt.strftime("%Y-%m-%d %H:%M:%S")
        price_data = get_price_data(symbol)
        has_price_data = price_data is not None and not getattr(price_data, "empty", False)
        current_price: Optional[float] = None
        if has_price_data:
            try:
                current_price = float(price_data['close'].iloc[-1])
            except Exception:
                current_price = None
        if direction == "long":
            if live_bid is not None:
                current_price = live_bid
            elif live_price is not None:
                current_price = live_price
        elif live_price is not None:
            current_price = live_price
        if direction == "short" and live_ask is not None:
            current_price = live_ask
        price_for_exit = current_price if current_price is not None else fallback_exit_price

        if entry_dt and datetime.utcnow() - entry_dt >= MAX_HOLDING_TIME:
            actions.append("time_exit")
            logger.info("%s exceeded max holding time; exiting trade.", symbol)
            qty = _to_float(trade.get('position_size', 1)) or 0.0
            exit_price_candidate = price_for_exit
            if exit_price_candidate is None:
                exit_price_candidate = fallback_exit_price
            if exit_price_candidate is None:
                exit_price_candidate = current_price
            if exit_price_candidate is None:
                exit_price_candidate = entry if entry is not None else 0.0
            exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute_exit_trade(
                trade,
                exit_price=exit_price_candidate,
                reason="max_holding_time",
                outcome="time_exit",
                quantity=qty,
                exit_time=exit_time,
            )
            _log_trade_closed(trade, reason="max_holding_time", outcome="time_exit")
            _persist_active_snapshot(
                updated_trades, active_trades, index, include_current=False
            )
            _update_rl(trade, exit_price_candidate)
            send_email(f" Time Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
            logger.debug("%s actions: %s", symbol, actions)
            continue
        if not has_price_data:
            continue
        # Use the latest candle's high/low in addition to the close so that
        # intrabar moves that touch TP/SL levels are not missed if price
        # reverses before the next polling cycle. Restrict these values to
        # post-entry candles so newly opened trades do not inherit
        # pre-entry extremes.
        if current_price is None:
            current_price = price_data['close'].iloc[-1]
        observed_price: Optional[float] = None
        recent_high: float
        recent_low: float
        if entry_dt is not None:
            if isinstance(price_data.index, pd.DatetimeIndex):
                recent_candles = price_data[price_data.index >= entry_dt]
            else:
                recent_candles = price_data
            if not recent_candles.empty:
                recent_high = recent_candles['high'].iloc[-1]
                recent_low = recent_candles['low'].iloc[-1]
                try:
                    observed_price = float(recent_low)
                except Exception:
                    observed_price = None
            else:
                fallback_price = current_price
                if fallback_price is None:
                    fallback_price = trade.get('entry')
                try:
                    fallback_price = float(fallback_price)
                except Exception:
                    fallback_price = 0.0
                recent_high = fallback_price
                recent_low = fallback_price
        else:
            recent_high = price_data['high'].iloc[-1]
            recent_low = price_data['low'].iloc[-1]
            try:
                observed_price = float(recent_low)
            except Exception:
                observed_price = None
        if live_high is not None:
            if recent_high is None or live_high > recent_high:
                recent_high = live_high
        if live_low is not None:
            if recent_low is None or live_low < recent_low:
                recent_low = live_low
            try:
                observed_price = float(live_low)
            except Exception:
                pass
        trailing_mode = bool(trade.get('trailing_active'))
        profit_riding_mode = "🔒 Trail" if trailing_mode else "—"
        logger.debug(
            "Managing %s | Price=%s High=%s Low=%s SL=%s TP1=%s TP2=%s TP3=%s Status=%s Mode=%s",
            symbol,
            current_price,
            recent_high,
            recent_low,
            sl,
            tp1,
            tp2,
            tp3,
            status_flags,
            profit_riding_mode,
        )
        execution_plan = None
        sell_pressure_signal = None
        order_book_snapshot = None
        reference_price_for_plan = current_price if current_price is not None else fallback_exit_price
        plan_side: Optional[str] = None
        if direction == "long":
            plan_side = "sell"
        elif direction == "short":
            plan_side = "buy"
        if symbol:
            try:
                order_book_snapshot = get_order_book(symbol, limit=20)
            except Exception as exc:
                logger.debug("Order book lookup failed for %s: %s", symbol, exc, exc_info=True)
        if order_book_snapshot is not None:
            if plan_side:
                try:
                    execution_plan = plan_execution(
                        plan_side,
                        reference_price_for_plan,
                        order_book_snapshot,
                        depth=15,
                    )
                except Exception as exc:
                    logger.debug("Failed to derive execution plan for %s: %s", symbol, exc, exc_info=True)
                    execution_plan = None
            try:
                sell_pressure_signal = detect_sell_pressure(
                    order_book_snapshot,
                    reference_price=reference_price_for_plan,
                    depth=15,
                )
            except Exception as exc:
                logger.debug("Failed to evaluate sell pressure for %s: %s", symbol, exc, exc_info=True)
                sell_pressure_signal = None
        if execution_plan:
            execution_plan["observed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            trade["last_order_book_plan"] = execution_plan
        if sell_pressure_signal:
            sell_pressure_signal["observed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            trade["last_order_book_signal"] = sell_pressure_signal
        flow_analysis = None
        flow_features: dict[str, Any] = {}
        cvd_strength: Optional[float] = None
        flow_imbalance: Optional[float] = None
        strong_buying_flow = False
        try:
            flow_analysis = detect_aggression(
                price_data,
                order_book=order_book_snapshot,
                symbol=symbol,
                live_trades=price_data.attrs.get("live_trades"),
            )
        except Exception as exc:
            logger.debug("Order-flow analysis failed for %s: %s", symbol, exc, exc_info=True)
            flow_analysis = None
        if flow_analysis is not None:
            flow_features = flow_analysis.features or {}
            trade["last_flow_state"] = flow_analysis.state
            trade["last_flow_features"] = flow_features
            cvd_strength = _to_float(flow_features.get("cvd_change"))
            if cvd_strength is None:
                cvd_strength = _to_float(flow_features.get("cvd"))
            flow_imbalance = _to_float(flow_features.get("trade_imbalance"))
            if flow_imbalance is None:
                flow_imbalance = _to_float(flow_features.get("order_book_imbalance"))
            strong_buying_flow = (
                flow_analysis.state == "buyers in control"
                and cvd_strength is not None
                and cvd_strength >= 0.25
                and flow_imbalance is not None
                and flow_imbalance >= 0.2
            )
        if (
            direction == "long"
            and entry is not None
            and strong_buying_flow
            and not status_flags.get("flow_break_even")
        ):
            break_even_price = entry
            current_sl = _to_float(trade.get("sl"))
            if current_sl is None or current_sl < break_even_price:
                _update_stop_loss(trade, break_even_price)
                _persist_active_snapshot(updated_trades, active_trades, index)
                actions.append("flow_break_even")
                logger.info(
                    "%s strong order-flow buying detected; SL moved to break even (%.6f)",
                    symbol,
                    break_even_price,
                )
                cvd_display = (
                    f"{cvd_strength:.3f}" if cvd_strength is not None else "n/a"
                )
                flow_display = (
                    f"{flow_imbalance:.3f}" if flow_imbalance is not None else "n/a"
                )
                try:
                    send_email(
                        f"🔒 Break-even SL: {symbol}",
                        (
                            f"Order-flow buyers in control — stop moved to {break_even_price:.6f}.\n"
                            f"CVD strength: {cvd_display}\n"
                            f"Flow imbalance: {flow_display}"
                        ),
                    )
                except Exception:
                    logger.debug("Failed to send break-even notification for %s", symbol, exc_info=True)
            status_flags["flow_break_even"] = True
        # Evaluate early exit conditions using the candle low to capture
        # sharp drops within the interval
        exit_now, reason = should_exit_early(trade, observed_price, price_data)
        if exit_now:
            actions.append("early_exit")
            logger.info("Early exit triggered for %s: %s", symbol, reason)
            qty = _to_float(trade.get('position_size', 1)) or 0.0
            exit_price_candidate = None
            if execution_plan is not None:
                exit_price_candidate = _to_float(execution_plan.get("recommended_price"))
            if exit_price_candidate is None:
                exit_price_candidate = _to_float(current_price)
            if exit_price_candidate is None:
                exit_price_candidate = _to_float(fallback_exit_price)
            if exit_price_candidate is None and entry is not None:
                exit_price_candidate = entry
            if exit_price_candidate is None:
                exit_price_candidate = 0.0
            exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            maker_flag = False
            if execution_plan is not None:
                maker_flag = execution_plan.get("aggressiveness") == "passive"
            execute_exit_trade(
                trade,
                exit_price=exit_price_candidate,
                reason=reason,
                outcome="early_exit",
                quantity=qty,
                exit_time=exit_time,
                maker=maker_flag,
            )
            _log_trade_closed(trade, reason=reason, outcome="early_exit")
            _persist_active_snapshot(
                updated_trades, active_trades, index, include_current=False
            )
            _update_rl(trade, exit_price_candidate)
            send_email(f" Early Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
            logger.debug("%s actions: %s", symbol, actions)
            continue
        if (
            direction == "long"
            and sell_pressure_signal
            and sell_pressure_signal.get("sell_pressure")
        ):
            qty = _to_float(trade.get('position_size', 1)) or 0.0
            if qty > 0:
                reason_text = sell_pressure_signal.get("reason") or "Order book sell pressure"
                exit_price_candidate = None
                if execution_plan is not None:
                    exit_price_candidate = _to_float(execution_plan.get("recommended_price"))
                if exit_price_candidate is None:
                    exit_price_candidate = _to_float(current_price)
                if exit_price_candidate is None:
                    exit_price_candidate = _to_float(price_for_exit)
                if exit_price_candidate is None:
                    exit_price_candidate = _to_float(fallback_exit_price)
                if exit_price_candidate is None and entry is not None:
                    exit_price_candidate = entry
                if exit_price_candidate is None:
                    exit_price_candidate = 0.0
                proceed = True
                if entry is not None and exit_price_candidate is not None:
                    proceed = (
                        exit_price_candidate >= entry
                        or sell_pressure_signal.get("urgency") == "high"
                    )
                if proceed:
                    actions.append("orderbook_exit")
                    exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    maker_flag = False
                    if execution_plan is not None:
                        maker_flag = execution_plan.get("aggressiveness") == "passive"
                    execute_exit_trade(
                        trade,
                        exit_price=exit_price_candidate,
                        reason=reason_text,
                        outcome="orderbook_exit",
                        quantity=qty,
                        exit_time=exit_time,
                        maker=maker_flag,
                    )
                    _log_trade_closed(trade, reason=reason_text, outcome="orderbook_exit")
                    _persist_active_snapshot(
                        updated_trades, active_trades, index, include_current=False
                    )
                    _update_rl(trade, exit_price_candidate)
                    send_email(
                        f"⚠️ Order Book Exit: {symbol}",
                        f"{trade}\n\nSignal: {sell_pressure_signal}\n\nNarrative:\n{trade.get('narrative', 'N/A')}",
                    )
                    logger.info(
                        "%s exited due to order book pressure: %s",
                        symbol,
                        reason_text,
                    )
                    logger.debug("%s actions: %s", symbol, actions)
                    continue
        # Compute updated indicators for trailing stops and TP
        indicators = calculate_indicators(price_data)
        adx_series = indicators.get('adx')
        macd_line = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        kc_lower_series = indicators.get('kc_lower')
        atr = indicators.get('atr', 0.005)
        atr_latest: Optional[float]
        if hasattr(atr, 'iloc'):
            try:
                atr_latest = float(atr.iloc[-1])
            except Exception:
                atr_latest = None
        else:
            atr_latest = _to_float(atr)
        if atr_latest is not None and (not math.isfinite(atr_latest) or atr_latest <= 0):
            atr_latest = None

        if hasattr(adx_series, 'iloc'):
            adx = adx_series.iloc[-1]
            adx_prev = adx_series.iloc[-2] if len(adx_series) > 1 else adx
        else:
            adx = adx_prev = adx_series or 20

        if hasattr(macd_line, 'iloc') and hasattr(macd_signal, 'iloc'):
            macd_line_last = macd_line.iloc[-1]
            macd_line_prev = macd_line.iloc[-2] if len(macd_line) > 1 else macd_line_last
            macd_signal_last = macd_signal.iloc[-1]
            macd_signal_prev = macd_signal.iloc[-2] if len(macd_signal) > 1 else macd_signal_last
        else:
            macd_line_last = macd_line_prev = macd_line or 0
            macd_signal_last = macd_signal_prev = macd_signal or 0

        macd_hist = macd_line_last - macd_signal_last
        _ = kc_lower_series  # retained for compatibility with earlier logic
        # Only long trades are supported in spot mode
        # Trailing-first management logic
        trade_closed = False
        exit_signals = should_exit_position(
            trade,
            current_price=current_price,
            recent_high=recent_high,
            recent_low=recent_low,
        )
        if exit_signals:
            for signal in exit_signals:
                label = str(signal.get("type") or "").lower()
                target_price = _to_float(signal.get("price"))
                if label == "tp1" and direction == "long":
                    if status_flags.get("tp1") or target_price is None:
                        continue
                    status_flags["tp1"] = True
                    lock_price = _activate_trailing_mode(
                        trade,
                        target_price,
                        current_price=current_price,
                    )
                    actions.append("tp1_trailing_activate")
                    if lock_price is not None:
                        current_sl = _to_float(trade.get("sl"))
                        if current_sl is None or lock_price > current_sl:
                            _update_stop_loss(trade, lock_price)
                            sl = lock_price
                            _persist_active_snapshot(updated_trades, active_trades, index)
                    logger.info(
                        "%s TP1 threshold reached — trailing mode activated (SL %.6f)",
                        symbol,
                        lock_price if lock_price is not None else trade.get("sl"),
                    )
                    try:
                        send_email(
                            f"🔒 Trail armed: {symbol}",
                            (
                                f"TP threshold {target_price:.6f} reached. "
                                f"Stop now protecting {TRAIL_LOCK_IN_RATIO * 100:.0f}% of the move."
                            ),
                        )
                    except Exception:
                        logger.debug("Failed to send trailing activation email for %s", symbol, exc_info=True)
                    _append_trailing_explanation(
                        trade,
                        "tp1_trailing_activate",
                        {
                            "trigger_price": target_price,
                            "lock_price": lock_price,
                            "current_price": current_price,
                            "atr": atr_latest,
                            "adx": float(adx) if adx is not None else None,
                            "macd_histogram": macd_hist,
                        },
                    )
                elif label == "tp2" and direction == "long":
                    if status_flags.get("tp2") or target_price is None:
                        continue
                    status_flags["tp2"] = True
                    if _adjust_trailing_multiplier(trade, TRAIL_TIGHT_ATR):
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        logger.info(
                            "%s TP2 threshold — trailing distance tightened to %.2f ATR",
                            symbol,
                            TRAIL_TIGHT_ATR,
                        )
                        actions.append("tp2_trail_tighten")
                        try:
                            send_email(
                                f"🔒 Trail tightened: {symbol}",
                                (
                                    "Price extended to management level 2. "
                                    f"Stop distance now {TRAIL_TIGHT_ATR:.2f}×ATR."
                                ),
                            )
                        except Exception:
                            logger.debug("Failed to send TP2 tighten email for %s", symbol, exc_info=True)
                        _append_trailing_explanation(
                            trade,
                            "tp2_trail_tighten",
                            {
                                "new_multiplier": TRAIL_TIGHT_ATR,
                                "atr": atr_latest,
                                "adx": float(adx) if adx is not None else None,
                                "macd_histogram": macd_hist,
                                "orderflow_state": getattr(flow_analysis, "state", None),
                            },
                        )
                elif label == "tp3" and direction == "long":
                    if status_flags.get("tp3") or target_price is None:
                        continue
                    status_flags["tp3"] = True
                    if _adjust_trailing_multiplier(trade, TRAIL_FINAL_ATR):
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        logger.info(
                            "%s TP3 threshold — final trailing distance %.2f ATR",
                            symbol,
                            TRAIL_FINAL_ATR,
                        )
                        actions.append("tp3_trail_final")
                        try:
                            send_email(
                                f"🔒 Trail tightened: {symbol}",
                                (
                                    "Momentum extended to management level 3. "
                                    f"Stop distance now {TRAIL_FINAL_ATR:.2f}×ATR."
                                ),
                            )
                        except Exception:
                            logger.debug("Failed to send TP3 tighten email for %s", symbol, exc_info=True)
                        _append_trailing_explanation(
                            trade,
                            "tp3_trail_tighten",
                            {
                                "new_multiplier": TRAIL_FINAL_ATR,
                                "atr": atr_latest,
                                "adx": float(adx) if adx is not None else None,
                                "macd_histogram": macd_hist,
                                "orderflow_state": getattr(flow_analysis, "state", None),
                            },
                        )
                elif label == "stop_loss":
                    if target_price is None:
                        continue
                    qty = _to_float(trade.get('position_size', 1)) or 0.0
                    exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    trailing_flag = bool(trade.get("trailing_active"))
                    outcome_label = "trailing_sl" if trailing_flag else "sl"
                    reason_text = "Trailing stop hit" if trailing_flag else "Stop loss hit"
                    execute_exit_trade(
                        trade,
                        exit_price=target_price,
                        reason=reason_text,
                        outcome=outcome_label,
                        quantity=qty,
                        exit_time=exit_time,
                    )
                    _log_trade_closed(trade, reason=reason_text, outcome=outcome_label)
                    _persist_active_snapshot(
                        updated_trades, active_trades, index, include_current=False
                    )
                    _update_rl(trade, target_price)
                    send_email(
                        f" Stop Loss Hit: {symbol}",
                        f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}",
                    )
                    actions.append("stop_loss")
                    trade_closed = True
                    break
        if trade_closed:
            logger.debug("%s actions: %s", symbol, actions)
            continue

        sl = _to_float(trade.get('sl'))
        trailing_active_now = bool(trade.get("trailing_active"))
        if trailing_active_now:
            atr_value: Optional[float]
            if hasattr(atr, 'iloc'):
                atr_value = float(atr.iloc[-1])
            else:
                atr_value = _to_float(atr)
            if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
                atr_value = _get_trade_atr(trade)

            if atr_value is not None and atr_value > 0:
                anchor = _to_float(trade.get("trail_high"))
                if anchor is None:
                    anchor = _to_float(trade.get("tp1"))
                if current_price is not None:
                    if direction == "long":
                        anchor = max(anchor or current_price, current_price)
                    else:
                        anchor = min(anchor or current_price, current_price)
                if anchor is not None:
                    trade["trail_high"] = anchor
                    multiplier = _to_float(trade.get("trail_multiplier"))
                    if multiplier is None or multiplier <= 0:
                        multiplier = TRAIL_INITIAL_ATR
                        trade["trail_multiplier"] = multiplier
                    candidate = None
                    if direction == "long":
                        candidate = anchor - atr_value * multiplier
                        lock_price = _to_float(trade.get("locked_profit_price"))
                        if lock_price is not None:
                            candidate = max(candidate, lock_price)
                        if candidate is not None:
                            current_sl = sl
                            if current_sl is None or candidate > current_sl + 1e-6:
                                _update_stop_loss(trade, round(candidate, 6))
                                sl = candidate
                                _persist_active_snapshot(updated_trades, active_trades, index)
                                logger.info(
                                    "%s trailing SL advanced to %.6f (anchor %.6f, ATR %.4f, mult %.2f)",
                                    symbol,
                                    candidate,
                                    anchor,
                                    atr_value,
                                    multiplier,
                                )
                                actions.append("trail_sl")
                                _append_trailing_explanation(
                                    trade,
                                    "trail_sl_move",
                                    {
                                        "new_stop": round(candidate, 6) if candidate is not None else None,
                                        "anchor": anchor,
                                        "atr": atr_value,
                                        "multiplier": multiplier,
                                        "current_price": current_price,
                                        "locked_profit_price": lock_price,
                                        "orderflow_state": getattr(flow_analysis, "state", None),
                                        "cvd_strength": cvd_strength,
                                    },
                                )

            orderflow_weak = False
            if flow_analysis is not None:
                if flow_analysis.state == "sellers in control":
                    orderflow_weak = True
                if cvd_strength is not None and cvd_strength < 0:
                    orderflow_weak = True
                imbalance_val = _to_float(flow_features.get("trade_imbalance"))
                if imbalance_val is not None and imbalance_val < -0.15:
                    orderflow_weak = True
            momentum_soft = (
                (adx is not None and adx < 15)
                or (macd_hist < 0)
                or (macd_line_last < macd_signal_last)
            )
            tighten_needed = orderflow_weak or momentum_soft
            if tighten_needed and _adjust_trailing_multiplier(trade, TRAIL_FINAL_ATR):
                _persist_active_snapshot(updated_trades, active_trades, index)
                logger.info(
                    "%s momentum waning — trailing multiplier tightened to %.2f ATR",
                    symbol,
                    TRAIL_FINAL_ATR,
                )
                actions.append("trail_tight_flow")
                _append_trailing_explanation(
                    trade,
                    "trail_multiplier_tightened",
                    {
                        "new_multiplier": TRAIL_FINAL_ATR,
                        "atr": atr_value,
                        "adx": float(adx) if adx is not None else None,
                        "macd_histogram": macd_hist,
                        "orderflow_state": getattr(flow_analysis, "state", None),
                        "cvd_strength": cvd_strength,
                        "flow_imbalance": flow_imbalance,
                        "momentum_soft": momentum_soft,
                    },
                )
        # Add the trade back to the updated list if still active
        logger.debug("%s actions: %s", symbol, actions if actions else "none")
        updated_trades.append(trade)
    # Persist updated trades to storage
    _persist_active_snapshot(updated_trades)


def manage_trades() -> None:
    """Iterate over active trades and update or close them."""

    if not _ACTIVE_TRADES_LOCK.acquire(blocking=False):
        logger.debug("manage_trades already running; skipping concurrent call")
        return
    try:
        _manage_trades_body()
    finally:
        _ACTIVE_TRADES_LOCK.release()
