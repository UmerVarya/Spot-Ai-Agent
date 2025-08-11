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

from datetime import datetime
from typing import List, Tuple, Optional

from trade_utils import get_price_data, calculate_indicators, estimate_commission, simulate_slippage
from macro_sentiment import analyze_macro_sentiment
from notifier import send_email
from log_utils import setup_logger
from trade_storage import (
    log_trade_result,
    store_trade,
    remove_trade,
    load_active_trades,
    save_active_trades,
)

# === Constants ===

# Exit thresholds
EARLY_EXIT_THRESHOLD = 0.015  # 1.5% move against entry
MACRO_CONFIDENCE_EXIT_THRESHOLD = 4

logger = setup_logger(__name__)


def create_new_trade(trade: dict) -> None:
    """Add a new trade to persistent storage."""
    store_trade(trade)


def should_exit_early(trade: dict, current_price: float, price_data) -> Tuple[bool, Optional[str]]:
    """Determine if a trade should exit early based on price and indicators."""
    entry = trade.get('entry')
    direction = trade.get('direction')
    if entry is None or direction is None:
        return False, None
    # 1. Price reversed significantly
    if direction == "long" and current_price < entry * (1 - EARLY_EXIT_THRESHOLD):
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
    if macro.get('bias') == "bearish" and macro.get('confidence', 0) < MACRO_CONFIDENCE_EXIT_THRESHOLD:
        return True, f"Macro sentiment shifted to bearish (Confidence: {macro.get('confidence')})"
    return False, None


def manage_trades() -> None:
    """Iterate over active trades and update or close them."""
    active_trades = load_active_trades()
    updated_trades: List[dict] = []
    for trade in active_trades:
        symbol = trade.get("symbol")
        price_data = get_price_data(symbol)
        if price_data is None or price_data.empty:
            continue
        current_price = price_data['close'].iloc[-1]
        direction = trade.get('direction')
        entry = trade.get('entry')
        sl = trade.get('sl')
        tp1 = trade.get('tp1')
        tp2 = trade.get('tp2')
        tp3 = trade.get('tp3')
        # Evaluate early exit conditions
        exit_now, reason = should_exit_early(trade, current_price, price_data)
        if exit_now:
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
            send_email(f" Early Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
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
            # Take profit logic
            if not trade['status'].get('tp1') and current_price >= tp1:
                trade['status']['tp1'] = True
                # Move stop loss to entry on TP1 hit
                trade['sl'] = entry
                logger.info("%s hit TP1 — SL moved to Entry", symbol)
            elif trade['status'].get('tp1') and not trade['status'].get('tp2') and current_price >= tp2:
                trade['status']['tp2'] = True
                trade['sl'] = tp1
                logger.info("%s hit TP2 — SL moved to TP1", symbol)
            elif trade['status'].get('tp2') and not trade['status'].get('tp3') and current_price >= tp3:
                trade['status']['tp3'] = True
                trade['profit_riding'] = True  # enable TP4 mode
                trade['sl'] = tp2
                logger.info("%s hit TP3 — Entering TP4 Profit Riding Mode", symbol)
                updated_trades.append(trade)
                continue
            elif current_price <= sl:
                # Stop loss hit
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
                send_email(f" Stop Loss Hit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                continue
            # Tighten stop-loss after TP1 if momentum fades (no TP4 mode)
            if trade['status'].get('tp1') and not trade.get('profit_riding'):
                if adx < 15 or macd_hist < 0:
                    tightened_sl = round(current_price - atr, 6)
                    if tightened_sl > trade['sl']:
                        trade['sl'] = tightened_sl
                        logger.info("SL tightened for %s to %s", symbol, trade['sl'])
            # TP4 profit riding logic
            if trade.get('profit_riding'):
                if adx > 25 and macd_hist > 0:
                    trail_sl = round(current_price - atr, 6)
                    if trail_sl > trade['sl']:
                        trade['sl'] = trail_sl
                        logger.info("%s TP4 ride: SL trailed to %s", symbol, trail_sl)
                else:
                    logger.warning("%s momentum weakening — exiting TP4", symbol)
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
                    send_email(f"✅ TP4 Exit: {symbol}", f"{trade}\n\n Narrative:\n{trade.get('narrative', 'N/A')}")
                    continue
        # Add the trade back to the updated list if still active
        updated_trades.append(trade)
    # Persist updated trades to storage
    save_active_trades(updated_trades)
