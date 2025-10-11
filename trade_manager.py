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
)
from rl_policy import RLPositionSizer
from microstructure import plan_execution, detect_sell_pressure
from orderflow import detect_aggression
from local_llm import generate_post_trade_summary

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
ATR_MULTIPLIER = 1.0
# Require fairly high confidence before exiting on bearish macro signals
MACRO_CONFIDENCE_EXIT_THRESHOLD = 7
# Maximum duration to hold a trade before forcing exit
MAX_HOLDING_TIME = timedelta(hours=6)

logger = setup_logger(__name__)
USE_RL_POSITION_SIZER = False
rl_sizer = RLPositionSizer() if USE_RL_POSITION_SIZER else None


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


def _calculate_leg_pnl(trade: dict, exit_price: float, quantity: float, fees: float, slippage: float) -> float:
    """Return realised PnL for a single trade leg."""

    try:
        entry_val = float(trade.get("entry"))
        exit_val = float(exit_price)
        qty_val = float(quantity)
    except Exception:
        return 0.0
    direction = str(trade.get("direction", "long")).lower()
    if direction == "short":
        pnl = (entry_val - exit_val) * qty_val
    else:
        pnl = (exit_val - entry_val) * qty_val
    try:
        pnl -= float(fees or 0.0)
    except Exception:
        pass
    try:
        pnl -= float(slippage or 0.0)
    except Exception:
        pass
    return pnl


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
    pnl = _calculate_leg_pnl(trade, exit_price, quantity, fee_val, slippage_val)
    trade["realized_pnl"] = trade.get("realized_pnl", 0.0) + pnl
    trade["realized_fees"] = trade.get("realized_fees", 0.0) + fee_val
    trade["realized_slippage"] = trade.get("realized_slippage", 0.0) + slippage_val
    trade["last_partial_exit"] = exit_time
    trade["last_partial_pnl"] = pnl
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
        trade["pnl_tp1"] = pnl
        if quantity_val is not None:
            trade["size_tp1"] = quantity_val
        if entry_val is not None and quantity_val is not None:
            trade["notional_tp1"] = entry_val * quantity_val
    elif label == "tp2":
        trade["tp2_partial"] = True
        trade["tp2_exit_price"] = exit_price
        trade["pnl_tp2"] = pnl
        if quantity_val is not None:
            trade["size_tp2"] = quantity_val
        if entry_val is not None and quantity_val is not None:
            trade["notional_tp2"] = entry_val * quantity_val
    else:
        trade["trail_partial_pnl"] = trade.get("trail_partial_pnl", 0.0) + pnl
        trade["trail_partial_count"] = trade.get("trail_partial_count", 0) + 1
        trade["last_trail_partial_pnl"] = pnl


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
    final_leg_pnl = _calculate_leg_pnl(trade, exit_price, quantity, fee_val, slippage_val)
    trade["final_leg_pnl"] = final_leg_pnl
    trade["final_exit_size"] = quantity
    total_pnl = trade.get("realized_pnl", 0.0) + final_leg_pnl
    total_fees = trade.get("realized_fees", 0.0) + fee_val
    total_slippage = trade.get("realized_slippage", 0.0) + slippage_val
    trade["realized_pnl"] = total_pnl
    trade["total_pnl"] = total_pnl
    trade["realized_fees"] = total_fees
    trade["total_fees"] = total_fees
    trade["realized_slippage"] = total_slippage
    trade["total_slippage"] = total_slippage
    trade["tp1_partial"] = bool(trade.get("tp1_partial"))
    trade["tp2_partial"] = bool(trade.get("tp2_partial"))
    entry_val = _to_float(trade.get("entry"))
    outcome_token = str(outcome or trade.get("outcome", "")).lower()
    if "tp3" in outcome_token and "_partial" not in outcome_token:
        trade["tp3_reached"] = True
        trade["pnl_tp3"] = trade.get("pnl_tp3", 0.0) + final_leg_pnl
        try:
            qty_val = float(quantity)
        except Exception:
            qty_val = None
        if qty_val is not None:
            trade["size_tp3"] = qty_val
        if entry_val is not None and qty_val is not None:
            trade["notional_tp3"] = entry_val * qty_val
    return total_fees, total_slippage


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
    if exit_time is None:
        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if fees is None:
        commission_rate = estimate_commission(symbol, quantity=qty_val, maker=maker)
        fees = exit_price_val * qty_val * commission_rate
    if slippage is None:
        slip_price = simulate_slippage(exit_price_val, direction=direction)
        slippage = abs(slip_price - exit_price_val)

    trade["exit_price"] = exit_price_val
    trade["exit_time"] = exit_time
    trade["outcome"] = outcome
    trade["exit_reason"] = reason

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
    if not status.get("tp1"):
        tp_price = trade.get("tp1")
    elif not status.get("tp2"):
        tp_price = trade.get("tp2")
    elif not status.get("tp3"):
        tp_price = trade.get("tp3")
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
    active_trades: List[dict],
    index: int,
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
        snapshot: List[dict] = list(updated_trades)
        if include_current and 0 <= index < len(active_trades):
            snapshot.append(active_trades[index])
        if index + 1 < len(active_trades):
            snapshot.extend(active_trades[index + 1 :])
        save_active_trades([trade for trade in snapshot if trade is not None])
    except Exception:
        logger.exception("Failed to persist active trades snapshot")


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
    return store_trade(trade)


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
    atr = indicators.get('atr', 0)
    if hasattr(atr, 'iloc'):
        atr = atr.iloc[-1]
    if direction == "long" and atr and observed_price is not None:
        drawdown = entry - observed_price
        if drawdown > atr * ATR_MULTIPLIER:
            return True, f"Drawdown {drawdown:.4f} exceeds {ATR_MULTIPLIER} ATR ({atr:.4f})"

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
    if direction == "long":
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


def manage_trades() -> None:
    """Iterate over active trades and update or close them."""
    active_trades = load_active_trades()
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
        profit_riding_mode = "🚀 TP4" if trade.get('profit_riding', False) else "—"
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
        kc_lower_val = kc_lower_series.iloc[-1] if hasattr(kc_lower_series, 'iloc') else kc_lower_series or 0
        # Only long trades are supported in spot mode
        trade_closed = False
        exit_signals = should_exit_position(
            trade,
            current_price=current_price,
            recent_high=recent_high,
            recent_low=recent_low,
        )
        if direction == "long":
            if exit_signals:
                for signal in exit_signals:
                    label = signal.get("type")
                    target_price = _to_float(signal.get("price"))
                    if label == "tp1":
                        if status_flags.get("tp1") or target_price is None:
                            continue
                        status_flags["tp1"] = True
                        initial_qty = _to_float(
                            trade.get('initial_size', trade.get('position_size', 1))
                        )
                        if initial_qty is None or initial_qty <= 0:
                            initial_qty = _to_float(trade.get('position_size', 1)) or 0.0
                        qty = _to_float(trade.get('position_size', 1)) or 0.0
                        sell_qty = min(initial_qty * 0.5, qty)
                        if sell_qty <= 0:
                            continue
                        commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                        fees = target_price * sell_qty * commission_rate
                        slip_price = simulate_slippage(target_price, direction=direction)
                        slippage_amt = abs(slip_price - target_price)
                        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        _record_partial_exit(
                            trade,
                            "tp1",
                            exit_price=target_price,
                            quantity=sell_qty,
                            fees=fees,
                            slippage=slippage_amt,
                            exit_time=exit_time,
                        )
                        partial_trade = trade.copy()
                        entry_price_val = entry
                        partial_trade['position_size'] = sell_qty
                        partial_trade['initial_size'] = sell_qty
                        if entry_price_val is not None:
                            partial_trade['size'] = sell_qty * entry_price_val
                            partial_trade['notional'] = sell_qty * entry_price_val
                        else:
                            partial_trade['size'] = sell_qty
                        partial_trade['exit_reason'] = "TP1 partial"
                        partial_trade['outcome'] = "tp1_partial"
                        partial_pnl = trade.get('last_partial_pnl')
                        if partial_pnl is not None:
                            partial_trade['realized_pnl'] = partial_pnl
                            partial_trade['total_pnl'] = partial_pnl
                        partial_trade['realized_fees'] = fees
                        partial_trade['total_fees'] = fees
                        partial_trade['realized_slippage'] = slippage_amt
                        partial_trade['total_slippage'] = slippage_amt
                        try:
                            log_trade_result(
                                partial_trade,
                                outcome="tp1_partial",
                                exit_price=target_price,
                                exit_time=exit_time,
                                fees=fees,
                                slippage=slippage_amt,
                            )
                        except Exception as error:
                            logger.error("Failed to log TP1 partial for %s: %s", symbol, error)
                        remaining_qty = max(0.0, qty - sell_qty)
                        trade['position_size'] = remaining_qty
                        if entry_price_val is not None:
                            trade['size'] = remaining_qty * entry_price_val
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        break_even_price = entry_price_val
                        current_sl = _to_float(trade.get('sl'))
                        if break_even_price is None:
                            break_even_price = current_sl if current_sl is not None else target_price
                        elif current_sl is not None:
                            break_even_price = max(break_even_price, current_sl)
                        if break_even_price is not None:
                            _update_stop_loss(trade, break_even_price)
                            _persist_active_snapshot(updated_trades, active_trades, index)
                        logger.info(
                            "%s hit TP1 — sold 50%% and moved SL to Break Even (%s)",
                            symbol,
                            break_even_price,
                        )
                        send_email(
                            f"✅ TP1 Partial: {symbol}",
                            f"Partial exit qty: {sell_qty:.6f} @ {target_price} (PnL: {trade.get('pnl_tp1', 0.0):.4f})\n\n"
                            f"Narrative:\n{trade.get('narrative', 'N/A')}",
                        )
                        actions.append("tp1_partial")
                    elif label == "tp2":
                        if status_flags.get("tp2") or target_price is None:
                            continue
                        status_flags["tp2"] = True
                        initial_qty = _to_float(
                            trade.get('initial_size', trade.get('position_size', 1))
                        )
                        if initial_qty is None or initial_qty <= 0:
                            initial_qty = _to_float(trade.get('position_size', 1)) or 0.0
                        qty = _to_float(trade.get('position_size', 1)) or 0.0
                        sell_qty = min(initial_qty * 0.3, qty)
                        if sell_qty <= 0:
                            continue
                        commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                        fees = target_price * sell_qty * commission_rate
                        slip_price = simulate_slippage(target_price, direction=direction)
                        slippage_amt = abs(slip_price - target_price)
                        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        _record_partial_exit(
                            trade,
                            "tp2",
                            exit_price=target_price,
                            quantity=sell_qty,
                            fees=fees,
                            slippage=slippage_amt,
                            exit_time=exit_time,
                        )
                        partial_trade = trade.copy()
                        entry_price_val = entry
                        partial_trade['position_size'] = sell_qty
                        partial_trade['initial_size'] = sell_qty
                        if entry_price_val is not None:
                            partial_trade['size'] = sell_qty * entry_price_val
                            partial_trade['notional'] = sell_qty * entry_price_val
                        else:
                            partial_trade['size'] = sell_qty
                        partial_trade['exit_reason'] = "TP2 partial"
                        partial_trade['outcome'] = "tp2_partial"
                        partial_pnl = trade.get('last_partial_pnl')
                        if partial_pnl is not None:
                            partial_trade['realized_pnl'] = partial_pnl
                            partial_trade['total_pnl'] = partial_pnl
                        partial_trade['realized_fees'] = fees
                        partial_trade['total_fees'] = fees
                        partial_trade['realized_slippage'] = slippage_amt
                        partial_trade['total_slippage'] = slippage_amt
                        try:
                            log_trade_result(
                                partial_trade,
                                outcome="tp2_partial",
                                exit_price=target_price,
                                exit_time=exit_time,
                                fees=fees,
                                slippage=slippage_amt,
                            )
                        except Exception as error:
                            logger.error("Failed to log TP2 partial for %s: %s", symbol, error)
                        remaining_qty = max(0.0, qty - sell_qty)
                        trade['position_size'] = remaining_qty
                        if entry_price_val is not None:
                            trade['size'] = remaining_qty * entry_price_val
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        desired_sl: Optional[float] = None
                        current_sl = _to_float(trade.get('sl'))
                        if trade.get('profit_riding') and target_price is not None:
                            desired_sl = target_price
                        elif tp1 is not None:
                            desired_sl = tp1
                        if (
                            desired_sl is not None
                            and (current_sl is None or desired_sl > current_sl)
                        ):
                            _update_stop_loss(trade, desired_sl)
                            _persist_active_snapshot(updated_trades, active_trades, index)
                            if trade.get('profit_riding'):
                                logger.info(
                                    "%s 🚀 TP4 prep — SL raised to %.6f",
                                    symbol,
                                    desired_sl,
                                )
                            else:
                                logger.info(
                                    "%s hit TP2 — sold 30%% and raised SL to %.6f",
                                    symbol,
                                    desired_sl,
                                )
                        else:
                            logger.info("%s hit TP2 — sold 30%%", symbol)
                        send_email(
                            f"✅ TP2 Partial: {symbol}",
                            f"Partial exit qty: {sell_qty:.6f} @ {target_price} (PnL: {trade.get('pnl_tp2', 0.0):.4f})\n\n"
                            f"Narrative:\n{trade.get('narrative', 'N/A')}",
                        )
                        actions.append("tp2_partial")
                    elif label == "tp3":
                        if status_flags.get("tp3") or target_price is None:
                            continue
                        status_flags["tp3"] = True
                        qty = _to_float(trade.get('position_size', trade.get('initial_size', 1)))
                        if qty is None or qty <= 0:
                            qty = _to_float(trade.get('initial_size', 1)) or 0.0
                        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        execute_exit_trade(
                            trade,
                            exit_price=target_price,
                            reason="tp3_hit",
                            outcome="tp3",
                            quantity=qty,
                            exit_time=exit_time,
                        )
                        _persist_active_snapshot(
                            updated_trades, active_trades, index, include_current=False
                        )
                        _update_rl(trade, target_price)
                        send_email(
                            f"✅ TP3 Exit: {symbol}",
                            f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}",
                        )
                        actions.append("tp3_exit")
                        trade_closed = True
                        break
                    elif label == "stop_loss":
                        if target_price is None:
                            continue
                        qty = _to_float(trade.get('position_size', 1)) or 0.0
                        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        outcome_label = "tp4_sl" if trade.get("profit_riding") else "sl"
                        reason_text = (
                            "Trailing stop loss"
                            if trade.get("profit_riding")
                            else "Stop loss hit"
                        )
                        execute_exit_trade(
                            trade,
                            exit_price=target_price,
                            reason=reason_text,
                            outcome=outcome_label,
                            quantity=qty,
                            exit_time=exit_time,
                        )
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
            # Refresh SL reference in case partial exits modified it
            sl = _to_float(trade.get('sl'))
            # Trailing logic after TP1 before entering TP4 mode
            if (
                trade['status'].get('tp1')
                and not trade.get('profit_riding')
                and current_price is not None
            ):
                trail_multiplier = 0.5 if adx < 15 or macd_hist < 0 else 1.0
                trail_candidate = current_price - atr * trail_multiplier
                base_price = entry if entry is not None else current_price
                if base_price is None:
                    base_price = trail_candidate
                trail_sl = round(max(base_price, trail_candidate), 6)
                current_sl = sl if sl is not None else _to_float(trade.get('sl'))
                if current_sl is None or trail_sl > current_sl:
                    _update_stop_loss(trade, trail_sl)
                    sl = trail_sl
                    _persist_active_snapshot(updated_trades, active_trades, index)
                    logger.info("%s TP1 trail: SL moved to %s", symbol, trail_sl)
                    actions.append("trail_sl")
            # TP4 profit riding logic
            if trade.get('profit_riding'):
                trail_pct = trade.get('trail_tp_pct')
                if trail_pct:
                    next_tp = trade.get('next_trail_tp')
                    if not next_tp:
                        next_tp = current_price * (1 + trail_pct)
                        trade['next_trail_tp'] = next_tp
                        _persist_active_snapshot(updated_trades, active_trades, index)
                    if recent_high >= next_tp:
                        qty = float(trade.get('position_size', 1))
                        sell_qty = qty * 0.1
                        commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                        fees = next_tp * sell_qty * commission_rate
                        slip_price = simulate_slippage(next_tp, direction=direction)
                        slippage_amt = abs(slip_price - next_tp)
                        exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        pct_reason = (
                            f"{trail_pct * 100:.1f}% trail"
                            if isinstance(trail_pct, (int, float))
                            else "Trail target"
                        )
                        _record_partial_exit(
                            trade,
                            "trail",
                            exit_price=next_tp,
                            quantity=sell_qty,
                            fees=fees,
                            slippage=slippage_amt,
                            exit_time=exit_time,
                        )
                        current_sl = _to_float(trade.get('sl'))
                        new_sl = max(current_sl or 0, next_tp)
                        _update_stop_loss(trade, new_sl)
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        logger.info("%s 🚀 TP4 trail: SL moved to %s", symbol, new_sl)
                        remaining_qty = qty - sell_qty
                        trade['position_size'] = remaining_qty
                        if entry is not None:
                            trade['size'] = remaining_qty * entry
                        trade['next_trail_tp'] = next_tp * (1 + trail_pct)
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        send_email(
                            f"✅ TP Trail: {symbol}",
                            f"Trailing partial qty: {sell_qty:.6f} @ {next_tp} (PnL: {trade.get('last_trail_partial_pnl', 0.0):.4f})\n\n"
                            f"Narrative:\n{trade.get('narrative', 'N/A')}",
                        )

                adx_drop = adx < 20 and adx < adx_prev
                macd_cross = macd_line_prev > macd_signal_prev and macd_line_last < macd_signal_last
                price_below_kc = current_price < kc_lower_val
                if current_price is not None and adx_drop and (macd_cross or price_below_kc):
                    logger.warning("%s 🚨 TP4 trailing exit triggered", symbol)
                    qty = _to_float(trade.get('position_size', 1)) or 0.0
                    exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    execute_exit_trade(
                        trade,
                        exit_price=current_price,
                        reason="TP4 trailing exit",
                        outcome="tp4",
                        quantity=qty,
                        exit_time=exit_time,
                    )
                    _persist_active_snapshot(
                        updated_trades, active_trades, index, include_current=False
                    )
                    _update_rl(trade, current_price)
                    send_email(
                        f"✅ TP4 Exit: {symbol}",
                        f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}",
                    )
                    actions.append("tp4_exit")
                    logger.debug("%s actions: %s", symbol, actions)
                    continue
                trail_multiplier = 0.7 if adx < 15 else 1.0
                trail_sl = round(current_price - atr * trail_multiplier, 6)
                if trail_sl > trade['sl']:
                    _update_stop_loss(trade, trail_sl)
                    _persist_active_snapshot(updated_trades, active_trades, index)
                    logger.info("%s 🚀 TP4 ride: SL trailed to %s", symbol, trail_sl)
                    actions.append("tp4_trail_sl")
        # Add the trade back to the updated list if still active
        logger.debug("%s actions: %s", symbol, actions if actions else "none")
        updated_trades.append(trade)
    # Persist updated trades to storage
    save_active_trades(updated_trades)
