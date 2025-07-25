"""
Trade management utilities for the Spot AI Super Agent.

This module maintains the lifecycle of open trades, including creation,
trailing stop logic, take‑profit rules and early exits based on macro
conditions.  It persists all open trades in a JSON file so that the
dashboard and agent can share state.  By default, the file lives
alongside this module.  If you move this file, the constant
``ACTIVE_TRADES_FILE`` will ensure the path is consistent.

Functions
---------
load_active_trades()
    Load the current open trades from disk.
save_active_trades(trades)
    Persist the open trades to disk.
create_new_trade(trade)
    Append a newly opened trade to the active trades file.
should_exit_early(trade, current_price, price_data)
    Determine if a trade should exit early based on price and indicators.
manage_trades()
    Iterate through active trades and update stop‑losses or close trades.

"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

from trade_utils import get_price_data, calculate_indicators
from macro_sentiment import analyze_macro_sentiment
from notifier import send_email
from trade_logger import log_trade_result


# === Constants ===

# Exit thresholds
EARLY_EXIT_THRESHOLD = 0.015  # 1.5% move against entry
MACRO_CONFIDENCE_EXIT_THRESHOLD = 4

# Path to the JSON file storing active trades.  Use a fixed absolute path
# based on this module's location.  This ensures that both the agent and
# dashboard processes read and write the same file regardless of their
# current working directory.
ACTIVE_TRADES_FILE = os.path.join(os.path.dirname(__file__), "active_trades.json")


def load_active_trades() -> Dict[str, dict]:
    """Load all currently active trades from disk.

    Returns
    -------
    dict
        A mapping of symbol to trade dictionaries.  If the file does not
        exist or is invalid, an empty dict is returned.
    """
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_active_trades(trades: Dict[str, dict]) -> None:
    """Persist the current active trades to disk.

    Parameters
    ----------
    trades : dict
        Mapping of symbol to trade info.  This function will overwrite the
        entire ``active_trades.json`` file.  It uses ``ACTIVE_TRADES_FILE``
        to ensure the location is consistent across the project.
    """
    try:
        with open(ACTIVE_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=4)
    except Exception as e:
        print(f"⚠️ Unable to save active trades: {e}")


def create_new_trade(trade: dict) -> None:
    """Add a new trade to the active trades file.

    Parameters
    ----------
    trade : dict
        The trade record to add.  Must include at least ``symbol``.

    This helper loads existing trades, inserts the new trade, then writes
    them back to disk using the shared file path.  If the file is missing
    or malformed, it will be recreated.
    """
    active_trades = load_active_trades()
    active_trades[trade["symbol"]] = trade
    save_active_trades(active_trades)


def should_exit_early(trade: dict, current_price: float, price_data) -> Tuple[bool, Optional[str]]:
    """Evaluate if a trade should exit early before reaching TP or SL.

    Parameters
    ----------
    trade : dict
        The trade dictionary containing at least ``entry`` and ``direction``.
    current_price : float
        The latest close price for the symbol.
    price_data : pd.DataFrame
        Recent price data used to compute indicators.

    Returns
    -------
    Tuple[bool, Optional[str]]
        A tuple of ``(exit_now, reason)`` indicating whether to exit and why.
    """
    entry = trade.get('entry')
    direction = trade.get('direction')
    if entry is None or direction is None:
        return False, None
    # 1. Price reversed significantly
    if direction == "long" and current_price < entry * (1 - EARLY_EXIT_THRESHOLD):
        return True, "Price dropped beyond early exit threshold"
    # 2. Indicator Weakness
    indicators = calculate_indicators(price_data)
    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd", 0)
    # Unpack pandas Series to scalar if needed
    if hasattr(rsi, 'iloc'):
        rsi = rsi.iloc[-1]
    if hasattr(macd_hist, 'iloc'):
        macd_hist = macd_hist.iloc[-1]
    if direction == "long" and rsi < 45:
        return True, f"Weak RSI: {rsi:.2f}"
    if direction == "long" and macd_hist < 0:
        return True, f"MACD histogram reversed: {macd_hist:.4f}"
    # 3. Macro Shift
    macro = analyze_macro_sentiment()
    if macro.get('bias') == "bearish" and macro.get('confidence', 0) < MACRO_CONFIDENCE_EXIT_THRESHOLD:
        return True, f"Macro sentiment shifted to bearish (Confidence: {macro.get('confidence')})"
    return False, None


def manage_trades() -> None:
    """Iterate over active trades and update or close them.

    The function reads the current ``active_trades.json``, processes each
    trade using price and indicator data, and writes the updated trades
    back to disk.  Trades that hit stop‑loss or profit targets are
    removed from the file and logged.
    """
    active_trades = load_active_trades()
    updated_trades = {}
    for symbol, trade in active_trades.items():
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
            print(f"🚨 Early exit triggered for {symbol}: {reason}")
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade['outcome'] = "early_exit"
            trade['exit_reason'] = reason
            log_trade_result(trade, outcome="early_exit", exit_price=current_price)
            send_email(f"🚨 Early Exit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
            continue
        # Compute updated indicators for trailing stops and TP
        indicators = calculate_indicators(price_data)
        adx = indicators.get("adx", 20)
        macd_hist = indicators.get("macd", 0)
        atr = indicators.get("atr", 0.005)
        if hasattr(adx, 'iloc'):
            adx = adx.iloc[-1]
        if hasattr(macd_hist, 'iloc'):
            macd_hist = macd_hist.iloc[-1]
        # Only long trades are supported in spot mode
        if direction == "long":
            # Take profit logic
            if not trade['status'].get('tp1') and current_price >= tp1:
                trade['status']['tp1'] = True
                trade['sl'] = entry
                print(f"🎯 {symbol} hit TP1 — SL moved to Entry")
            elif trade['status'].get('tp1') and not trade['status'].get('tp2') and current_price >= tp2:
                trade['status']['tp2'] = True
                trade['sl'] = tp1
                print(f"🎯 {symbol} hit TP2 — SL moved to TP1")
            elif trade['status'].get('tp2') and not trade['status'].get('tp3') and current_price >= tp3:
                trade['status']['tp3'] = True
                trade['profit_riding'] = True  # enable TP4 mode
                trade['sl'] = tp2
                print(f"🚀 {symbol} hit TP3 — Entering TP4 Profit Riding Mode")
                # Keep trade in active trades but no further TP checks until next iteration
                updated_trades[symbol] = trade
                continue
            elif current_price <= sl:
                # Stop loss hit
                print(f"🛑 {symbol} hit Stop Loss!")
                trade['exit_price'] = sl
                trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "tp4_sl" if trade.get("profit_riding") else "sl"
                log_trade_result(trade, outcome=trade['outcome'], exit_price=sl)
                send_email(f"🛑 Stop Loss Hit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                continue
            # Tighten stop‑loss after TP1 if momentum fades (no TP4 mode)
            if trade['status'].get('tp1') and not trade.get('profit_riding'):
                if adx < 15 or macd_hist < 0:
                    tightened_sl = round(current_price - atr, 6)
                    if tightened_sl > trade['sl']:
                        trade['sl'] = tightened_sl
                        print(f"🔒 SL tightened for {symbol} to {trade['sl']}")
            # TP4 profit riding logic
            if trade.get('profit_riding'):
                if adx > 25 and macd_hist > 0:
                    trail_sl = round(current_price - atr, 6)
                    if trail_sl > trade['sl']:
                        trade['sl'] = trail_sl
                        print(f"🏇 {symbol} TP4 ride: SL trailed to {trail_sl}")
                else:
                    print(f"⚠️ {symbol} momentum weakening — exiting TP4")
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trade['outcome'] = "tp4"
                    log_trade_result(trade, outcome="tp4", exit_price=current_price)
                    send_email(f"✅ TP4 Exit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                    continue
        # Add trade back to updated trades if still active
        updated_trades[symbol] = trade
    # Persist updated trades to disk
    save_active_trades(updated_trades)
