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

from datetime import datetime, timedelta
from typing import List, Tuple, Optional

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

# Exit thresholds
EARLY_EXIT_THRESHOLD = 0.015  # 1.5% move against entry
# Require fairly high confidence before exiting on bearish macro signals
MACRO_CONFIDENCE_EXIT_THRESHOLD = 7
# Maximum duration to hold a trade before forcing exit
MAX_HOLDING_TIME = timedelta(hours=1)

logger = setup_logger(__name__)
rl_sizer = RLPositionSizer()

def _update_rl(trade: dict, exit_price: float) -> None:
    """Update RL position sizer based on trade outcome."""
    try:
        entry = trade.get('entry')
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
        qty = float(trade.get("size", trade.get("position_size", 1)))
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


def should_exit_early(trade: dict, observed_price: float, price_data) -> Tuple[bool, Optional[str]]:
    """Determine if a trade should exit early based on price and indicators.

    ``observed_price`` represents the most adverse price seen during the
    polling interval (e.g., the candle low for long trades) so that sharp
    wicks can still trigger an exit.
    """
    entry = trade.get('entry')
    direction = trade.get('direction')
    if entry is None or direction is None:
        return False, None
    # 1. Price reversed significantly
    if direction == "long" and observed_price < entry * (1 - EARLY_EXIT_THRESHOLD):
        return True, "Price dropped beyond early exit threshold"
    # 2. Indicator weakness
    indicators = calculate_indicators(price_data)
    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd", 0)
    if hasattr(rsi, 'iloc'):
        rsi = rsi.iloc[-1]
    if hasattr(macd_hist, 'iloc'):
        macd_hist = macd_hist.iloc[-1]
    if direction == "long" and rsi < 45:
        return True, f"Weak RSI: {rsi:.2f}"
    if direction == "long" and macd_hist < 0:
        return True, f"MACD histogram reversed: {macd_hist:.4f}"
    # 3. Macro shift
    macro = analyze_macro_sentiment()
    if macro.get('bias') == "bearish" and macro.get('confidence', 0) >= MACRO_CONFIDENCE_EXIT_THRESHOLD:
        return True, f"Macro sentiment shifted to bearish (Confidence: {macro.get('confidence')})"
    return False, None


def manage_trades() -> None:
    """Iterate over active trades and update or close them."""
    active_trades = load_active_trades()
    updated_trades: List[dict] = []
    for trade in active_trades:
        symbol = trade.get("symbol")
        # Ensure original size is tracked for partial profit-taking
        if "initial_size" not in trade:
            try:
                trade["initial_size"] = float(trade.get("size", trade.get("position_size", 1)))
            except Exception:
                trade["initial_size"] = 1.0
        price_data = get_price_data(symbol)
        if price_data is None or price_data.empty:
            continue
        # Use the latest candle's high/low in addition to the close so that
        # intrabar moves that touch TP/SL levels are not missed if price
        # reverses before the next polling cycle.
        current_price = price_data['close'].iloc[-1]
        recent_high = price_data['high'].iloc[-1]
        recent_low = price_data['low'].iloc[-1]
        direction = trade.get('direction')
        entry = trade.get('entry')
        sl = trade.get('sl')
        tp1 = trade.get('tp1')
        tp2 = trade.get('tp2')
        tp3 = trade.get('tp3')
        status_flags = trade.get('status', {})
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
        actions = []
        # Time-based exit: close trades exceeding the maximum holding duration
        entry_time_str = trade.get('entry_time')
        try:
            entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S") if entry_time_str else None
        except Exception:
            entry_dt = None
        if entry_dt and datetime.utcnow() - entry_dt >= MAX_HOLDING_TIME:
            actions.append("time_exit")
            logger.info("%s exceeded max holding time; exiting trade.", symbol)
            qty = float(trade.get('size', trade.get('position_size', 1)))
            commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
            fees = current_price * qty * commission_rate
            slip_price = simulate_slippage(current_price, direction=direction)
            slippage = abs(slip_price - current_price)
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            trade['outcome'] = "time_exit"
            trade['exit_reason'] = "max_holding_time"
            log_trade_result(
                trade,
                outcome="time_exit",
                exit_price=current_price,
                exit_time=trade['exit_time'],
                fees=fees,
                slippage=slippage,
            )
            _update_rl(trade, current_price)
            send_email(f" Time Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
            logger.debug("%s actions: %s", symbol, actions)
            continue
        # Evaluate early exit conditions using the candle low to capture
        # sharp drops within the interval
        exit_now, reason = should_exit_early(trade, recent_low, price_data)
        if exit_now:
            actions.append("early_exit")
            logger.info("Early exit triggered for %s: %s", symbol, reason)
            # Compute fees and slippage on exit
            qty = float(trade.get('size', trade.get('position_size', 1)))
            commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
            fees = current_price * qty * commission_rate
            slip_price = simulate_slippage(current_price, direction=direction)
            slippage = abs(slip_price - current_price)
            # Record exit details
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            trade['outcome'] = "early_exit"
            trade['exit_reason'] = reason
            # Log trade result with fees and slippage
            log_trade_result(
                trade,
                outcome="early_exit",
                exit_price=current_price,
                exit_time=trade['exit_time'],
                fees=fees,
                slippage=slippage,
            )
            _update_rl(trade, current_price)
            send_email(f" Early Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
            logger.debug("%s actions: %s", symbol, actions)
            continue
        # Compute updated indicators for trailing stops and TP
        indicators = calculate_indicators(price_data)
        adx = indicators.get('adx', 20)
        macd_hist = indicators.get('macd', 0)
        atr = indicators.get('atr', 0.005)
        if hasattr(adx, 'iloc'):
            adx = adx.iloc[-1]
        if hasattr(macd_hist, 'iloc'):
            macd_hist = macd_hist.iloc[-1]
        # Only long trades are supported in spot mode
        if direction == "long":
            # Take profit logic with partial exits
            if not trade['status'].get('tp1') and recent_high >= tp1:
                trade['status']['tp1'] = True
                initial_qty = float(trade.get('initial_size', trade.get('size', trade.get('position_size', 1))))
                sell_qty = initial_qty * 0.5
                qty = float(trade.get('size', trade.get('position_size', 1)))
                sell_qty = min(sell_qty, qty)
                commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                fees = tp1 * sell_qty * commission_rate
                slip_price = simulate_slippage(tp1, direction=direction)
                slippage_amt = abs(slip_price - tp1)
                partial_trade = trade.copy()
                partial_trade['size'] = sell_qty
                exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                partial_trade['exit_price'] = tp1
                partial_trade['exit_time'] = exit_time
                log_trade_result(
                    partial_trade,
                    outcome="tp1_partial",
                    exit_price=tp1,
                    exit_time=exit_time,
                    fees=fees,
                    slippage=slippage_amt,
                )
                send_email(f"✅ TP1 Partial: {symbol}", f"{partial_trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                remaining_qty = qty - sell_qty
                trade['size'] = remaining_qty
                trade['position_size'] = remaining_qty
                break_even_price = max(entry, trade.get('sl', entry))
                _update_stop_loss(trade, break_even_price)
                logger.info(
                    "%s hit TP1 — sold 50%% and moved SL to Break Even (%s)",
                    symbol,
                    break_even_price,
                )
                actions.append("tp1_partial")

            if trade['status'].get('tp1') and not trade['status'].get('tp2') and recent_high >= tp2:
                trade['status']['tp2'] = True
                initial_qty = float(trade.get('initial_size', trade.get('size', trade.get('position_size', 1))))
                sell_qty = initial_qty * 0.3
                qty = float(trade.get('size', trade.get('position_size', 1)))
                sell_qty = min(sell_qty, qty)
                commission_rate = estimate_commission(symbol, quantity=sell_qty, maker=False)
                fees = tp2 * sell_qty * commission_rate
                slip_price = simulate_slippage(tp2, direction=direction)
                slippage_amt = abs(slip_price - tp2)
                partial_trade = trade.copy()
                partial_trade['size'] = sell_qty
                exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                partial_trade['exit_price'] = tp2
                partial_trade['exit_time'] = exit_time
                log_trade_result(
                    partial_trade,
                    outcome="tp2_partial",
                    exit_price=tp2,
                    exit_time=exit_time,
                    fees=fees,
                    slippage=slippage_amt,
                )
                send_email(f"✅ TP2 Partial: {symbol}", f"{partial_trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                remaining_qty = qty - sell_qty
                trade['size'] = remaining_qty
                trade['position_size'] = remaining_qty
                _update_stop_loss(trade, tp1)
                logger.info("%s hit TP2 — sold 30% and moved SL to TP1", symbol)
                actions.append("tp2_partial")

            if trade['status'].get('tp2') and not trade['status'].get('tp3') and recent_high >= tp3:
                trade['status']['tp3'] = True
                trade['profit_riding'] = True  # enable TP4 mode
                _update_stop_loss(trade, tp2)
                logger.info("%s hit TP3 — Entering TP4 Profit Riding Mode", symbol)
                send_email(f"✅ TP3 Hit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                actions.append("tp3_hit")
                logger.debug("%s actions: %s", symbol, actions)
                updated_trades.append(trade)
                continue
            if recent_low <= sl:
                # Stop loss hit (use intrabar low so quick wicks trigger)
                logger.info("%s hit Stop Loss!", symbol)
                qty = float(trade.get('size', trade.get('position_size', 1)))
                commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
                fees = sl * qty * commission_rate
                slip_price = simulate_slippage(sl, direction=direction)
                slippage_amt = abs(slip_price - sl)
                trade['exit_price'] = sl
                trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "tp4_sl" if trade.get("profit_riding") else "sl"
                log_trade_result(
                    trade,
                    outcome=trade['outcome'],
                    exit_price=sl,
                    exit_time=trade['exit_time'],
                    fees=fees,
                    slippage=slippage_amt,
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
                    logger.info("%s TP1 trail: SL moved to %s", symbol, trail_sl)
                    actions.append("trail_sl")
            # TP4 profit riding logic
            if trade.get('profit_riding'):
                if macd_hist < 0:
                    logger.warning("%s momentum reversal — exiting TP4", symbol)
                    qty = float(trade.get('size', trade.get('position_size', 1)))
                    commission_rate = estimate_commission(symbol, quantity=qty, maker=False)
                    fees = current_price * qty * commission_rate
                    slip_price = simulate_slippage(current_price, direction=direction)
                    slippage_amt = abs(slip_price - current_price)
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    trade['outcome'] = "tp4"
                    log_trade_result(
                        trade,
                        outcome="tp4",
                        exit_price=current_price,
                        exit_time=trade['exit_time'],
                        fees=fees,
                        slippage=slippage_amt,
                    )
                    _update_rl(trade, current_price)
                    send_email(f"✅ TP4 Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                    actions.append("tp4_exit")
                    logger.debug("%s actions: %s", symbol, actions)
                    continue
                trail_multiplier = 0.7 if adx < 15 else 1.0
                trail_sl = round(current_price - atr * trail_multiplier, 6)
                if trail_sl > trade['sl']:
                    _update_stop_loss(trade, trail_sl)
                    logger.info("%s TP4 ride: SL trailed to %s", symbol, trail_sl)
                    actions.append("tp4_trail_sl")
        # Add the trade back to the updated list if still active
        logger.debug("%s actions: %s", symbol, actions if actions else "none")
        updated_trades.append(trade)
    # Persist updated trades to storage
    save_active_trades(updated_trades)
