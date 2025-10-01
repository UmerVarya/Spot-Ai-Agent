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
from typing import List, Tuple, Optional, Union

import pandas as pd

from trade_utils import (
    get_price_data,
    calculate_indicators,
    estimate_commission,
    simulate_slippage,
    update_stop_loss_order,
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
rl_sizer = RLPositionSizer()


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
    return total_fees, total_slippage

def _update_rl(trade: dict, exit_price: float) -> None:
    """Update RL position sizer based on trade outcome."""
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


def create_new_trade(trade: dict) -> bool:
    """Add a new trade to persistent storage if not already active.

    Returns
    -------
    bool
        ``True`` if the trade was stored, ``False`` if a trade with the
        same symbol was already active.
    """
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

        direction = trade.get('direction')
        entry = trade.get('entry')
        sl = trade.get('sl')
        tp1 = trade.get('tp1')
        tp2 = trade.get('tp2')
        tp3 = trade.get('tp3')
        status_flags = trade.get('status', {})
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
            qty = float(trade.get('position_size', 1))
            commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
            price_for_fees = price_for_exit if price_for_exit is not None else 0.0
            fees = price_for_fees * qty * commission_rate
            if price_for_exit is not None:
                slip_price = simulate_slippage(price_for_exit, direction=direction)
                slippage = abs(slip_price - price_for_exit)
            else:
                slippage = 0.0
            trade['exit_price'] = price_for_exit
            trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            trade['outcome'] = "time_exit"
            trade['exit_reason'] = "max_holding_time"
            total_fees, total_slippage = _finalize_trade_result(
                trade,
                exit_price=price_for_exit if price_for_exit is not None else 0.0,
                quantity=qty,
                fees=fees,
                slippage=slippage,
            )
            log_trade_result(
                trade,
                outcome="time_exit",
                exit_price=price_for_exit,
                exit_time=trade['exit_time'],
                fees=total_fees,
                slippage=total_slippage,
            )
            _persist_active_snapshot(
                updated_trades, active_trades, index, include_current=False
            )
            if price_for_exit is not None:
                _update_rl(trade, price_for_exit)
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
        logger.debug(
            "Managing %s | Price=%s High=%s Low=%s SL=%s TP1=%s TP2=%s TP3=%s Status=%s ProfitRiding=%s",
            symbol,
            current_price,
            recent_high,
            recent_low,
            sl,
            tp1,
            tp2,
            tp3,
            status_flags,
            trade.get('profit_riding', False),
        )
        # Evaluate early exit conditions using the candle low to capture
        # sharp drops within the interval
        exit_now, reason = should_exit_early(trade, observed_price, price_data)
        if exit_now:
            actions.append("early_exit")
            logger.info("Early exit triggered for %s: %s", symbol, reason)
            # Compute fees and slippage on exit
            qty = float(trade.get('position_size', 1))
            commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
            fees = current_price * qty * commission_rate
            slip_price = simulate_slippage(current_price, direction=direction)
            slippage = abs(slip_price - current_price)
            # Record exit details
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            trade['outcome'] = "early_exit"
            trade['exit_reason'] = reason
            total_fees, total_slippage = _finalize_trade_result(
                trade,
                exit_price=current_price,
                quantity=qty,
                fees=fees,
                slippage=slippage,
            )
            # Log trade result with fees and slippage
            log_trade_result(
                trade,
                outcome="early_exit",
                exit_price=current_price,
                exit_time=trade['exit_time'],
                fees=total_fees,
                slippage=total_slippage,
            )
            _persist_active_snapshot(
                updated_trades, active_trades, index, include_current=False
            )
            _update_rl(trade, current_price)
            send_email(f" Early Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
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
        if direction == "long":
            # Take profit logic with partial exits
            if not trade['status'].get('tp1') and recent_high >= tp1:
                trade['status']['tp1'] = True
                initial_qty = float(trade.get('initial_size', trade.get('position_size', 1)))
                sell_qty = initial_qty * 0.5
                qty = float(trade.get('position_size', 1))
                sell_qty = min(sell_qty, qty)
                commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                fees = tp1 * sell_qty * commission_rate
                slip_price = simulate_slippage(tp1, direction=direction)
                slippage_amt = abs(slip_price - tp1)
                exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                _record_partial_exit(
                    trade,
                    "tp1",
                    exit_price=tp1,
                    quantity=sell_qty,
                    fees=fees,
                    slippage=slippage_amt,
                    exit_time=exit_time,
                )
                partial_trade = trade.copy()
                try:
                    entry_price_val = float(entry) if entry is not None else None
                except Exception:
                    entry_price_val = None
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
                        exit_price=tp1,
                        exit_time=exit_time,
                        fees=fees,
                        slippage=slippage_amt,
                    )
                except Exception as error:
                    logger.error("Failed to log TP1 partial for %s: %s", symbol, error)
                remaining_qty = qty - sell_qty
                trade['position_size'] = remaining_qty
                trade['size'] = remaining_qty * entry
                _persist_active_snapshot(updated_trades, active_trades, index)
                break_even_price = max(entry, trade.get('sl', entry))
                _update_stop_loss(trade, break_even_price)
                _persist_active_snapshot(updated_trades, active_trades, index)
                logger.info(
                    "%s hit TP1 — sold 50%% and moved SL to Break Even (%s)",
                    symbol,
                    break_even_price,
                )
                send_email(
                    f"✅ TP1 Partial: {symbol}",
                    f"Partial exit qty: {sell_qty:.6f} @ {tp1} (PnL: {trade.get('pnl_tp1', 0.0):.4f})\n\n"
                    f"Narrative:\n{trade.get('narrative', 'N/A')}",
                )
                actions.append("tp1_partial")

            if trade['status'].get('tp1') and not trade['status'].get('tp2') and recent_high >= tp2:
                trade['status']['tp2'] = True
                initial_qty = float(trade.get('initial_size', trade.get('position_size', 1)))
                sell_qty = initial_qty * 0.3
                qty = float(trade.get('position_size', 1))
                sell_qty = min(sell_qty, qty)
                commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                fees = tp2 * sell_qty * commission_rate
                slip_price = simulate_slippage(tp2, direction=direction)
                slippage_amt = abs(slip_price - tp2)
                exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                _record_partial_exit(
                    trade,
                    "tp2",
                    exit_price=tp2,
                    quantity=sell_qty,
                    fees=fees,
                    slippage=slippage_amt,
                    exit_time=exit_time,
                )
                partial_trade = trade.copy()
                try:
                    entry_price_val = float(entry) if entry is not None else None
                except Exception:
                    entry_price_val = None
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
                        exit_price=tp2,
                        exit_time=exit_time,
                        fees=fees,
                        slippage=slippage_amt,
                    )
                except Exception as error:
                    logger.error("Failed to log TP2 partial for %s: %s", symbol, error)
                remaining_qty = qty - sell_qty
                trade['position_size'] = remaining_qty
                trade['size'] = remaining_qty * entry
                _persist_active_snapshot(updated_trades, active_trades, index)
                _update_stop_loss(trade, tp1)
                _persist_active_snapshot(updated_trades, active_trades, index)
                logger.info("%s hit TP2 — sold 30% and moved SL to TP1", symbol)
                send_email(
                    f"✅ TP2 Partial: {symbol}",
                    f"Partial exit qty: {sell_qty:.6f} @ {tp2} (PnL: {trade.get('pnl_tp2', 0.0):.4f})\n\n"
                    f"Narrative:\n{trade.get('narrative', 'N/A')}",
                )
                actions.append("tp2_partial")

            if trade['status'].get('tp2') and not trade['status'].get('tp3') and recent_high >= tp3:
                trade['status']['tp3'] = True
                trade['profit_riding'] = True  # enable TP4 mode
                _update_stop_loss(trade, tp2)
                _persist_active_snapshot(updated_trades, active_trades, index)
                logger.info("%s hit TP3 — Entering TP4 Profit Riding Mode", symbol)
                send_email(f"✅ TP3 Hit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                actions.append("tp3_hit")
                logger.debug("%s actions: %s", symbol, actions)
                updated_trades.append(trade)
                continue
            if recent_low <= sl:
                # Stop loss hit (use intrabar low so quick wicks trigger)
                logger.info("%s hit Stop Loss!", symbol)
                qty = float(trade.get('position_size', 1))
                commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
                fees = sl * qty * commission_rate
                slip_price = simulate_slippage(sl, direction=direction)
                slippage_amt = abs(slip_price - sl)
                trade['exit_price'] = sl
                trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "tp4_sl" if trade.get("profit_riding") else "sl"
                trade['exit_reason'] = (
                    "Trailing stop loss"
                    if trade.get("profit_riding")
                    else "Stop loss hit"
                )
                total_fees, total_slippage = _finalize_trade_result(
                    trade,
                    exit_price=sl,
                    quantity=qty,
                    fees=fees,
                    slippage=slippage_amt,
                )
                log_trade_result(
                    trade,
                    outcome=trade['outcome'],
                    exit_price=sl,
                    exit_time=trade['exit_time'],
                    fees=total_fees,
                    slippage=total_slippage,
                )
                _persist_active_snapshot(
                    updated_trades, active_trades, index, include_current=False
                )
                _update_rl(trade, sl)
                send_email(f" Stop Loss Hit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                actions.append("stop_loss")
                logger.debug("%s actions: %s", symbol, actions)
                continue
            # Trailing logic after TP1 before entering TP4 mode
            if trade['status'].get('tp1') and not trade.get('profit_riding'):
                trail_multiplier = 0.5 if adx < 15 or macd_hist < 0 else 1.0
                trail_sl = round(max(entry, current_price - atr * trail_multiplier), 6)
                if trail_sl > trade['sl']:
                    _update_stop_loss(trade, trail_sl)
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
                        new_sl = max(trade.get('sl', 0), next_tp)
                        _update_stop_loss(trade, new_sl)
                        _persist_active_snapshot(updated_trades, active_trades, index)
                        logger.info("%s TP Trail: SL moved to %s", symbol, new_sl)
                        remaining_qty = qty - sell_qty
                        trade['position_size'] = remaining_qty
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
                if adx_drop and (macd_cross or price_below_kc):
                    logger.warning("%s momentum reversal — exiting TP4", symbol)
                    qty = float(trade.get('position_size', 1))
                    commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
                    fees = current_price * qty * commission_rate
                    slip_price = simulate_slippage(current_price, direction=direction)
                    slippage_amt = abs(slip_price - current_price)
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    trade['outcome'] = "tp4"
                    trade['exit_reason'] = "TP4 trailing exit"
                    total_fees, total_slippage = _finalize_trade_result(
                        trade,
                        exit_price=current_price,
                        quantity=qty,
                        fees=fees,
                        slippage=slippage_amt,
                    )
                    log_trade_result(
                        trade,
                        outcome="tp4",
                        exit_price=current_price,
                        exit_time=trade['exit_time'],
                        fees=total_fees,
                        slippage=total_slippage,
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
                    logger.info("%s TP4 ride: SL trailed to %s", symbol, trail_sl)
                    actions.append("tp4_trail_sl")
        # Add the trade back to the updated list if still active
        logger.debug("%s actions: %s", symbol, actions if actions else "none")
        updated_trades.append(trade)
    # Persist updated trades to storage
    save_active_trades(updated_trades)
