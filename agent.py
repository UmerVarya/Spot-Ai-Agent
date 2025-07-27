"""
    Extended Spot AI Super Agent loop with enhanced trade logging.

    This version of ``agent.py`` builds upon the refined agent loop by
    recording additional metadata when a trade is opened.  In
    particular it stamps each new trade with an ``entry_time``, tags it
    with the detected strategy (chart pattern) and current session, and
    stores the position size and leverage explicitly.  These fields are
    written into the persistent storage via ``create_new_trade`` and
    ``log_trade_result`` so that the dashboard can compute accurate
    performance statistics.

    Only the sections relating to trade construction and logging have
    been modified; all other logic (news fetching, macro gating,
    technical scoring, LLM vetting, ML probability blending, and
    dynamic take‑profit/stop‑loss calculation) remains identical to
    the refined agent.
    """

import logging
logging.basicConfig(level=logging.INFO)
logging.info("agent.py starting...")

import time
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import json
import threading
from fetch_news import fetch_news, run_news_fetcher
from trade_utils import simulate_slippage, estimate_commission
from trade_utils import get_top_symbols, get_price_data, evaluate_signal, get_market_session, calculate_indicators
from trade_manager import manage_trades, load_active_trades, save_active_trades, create_new_trade
from notifier import send_email, log_rejection
from trade_storage import log_trade_result  # import from trade_storage instead of trade_logger
from brain import should_trade
from sentiment import get_macro_sentiment
from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index
from orderflow import detect_aggression
from diversify import select_diversified_signals
from ml_model import predict_success_probability
from drawdown_guard import is_trading_blocked
import numpy as np

# Maximum concurrent open trades
MAX_ACTIVE_TRADES = 2
# Interval between scans (in seconds)
SCAN_INTERVAL = 15
# Interval between news fetches (in seconds)
NEWS_INTERVAL = 3600


def auto_run_news():
    while True:
        print("🗞️ Running scheduled news fetcher...")
        run_news_fetcher()
        time.sleep(NEWS_INTERVAL)


def run_streamlit():
    port = os.environ.get("PORT", "10000")
    # Launch the dashboard as a daemon thread
    os.system(f"streamlit run dashboard.py --server.port {port} --server.headless true")


def run_agent_loop():
    print("\n🤖 Spot AI Super Agent running in paper trading mode...\n")
    # start news and dashboard threads
    threading.Thread(target=auto_run_news, daemon=True).start()
    threading.Thread(target=run_streamlit, daemon=True).start()
    # Ensure symbol_scores.json exists
    if not os.path.exists("symbol_scores.json"):
        with open("symbol_scores.json", "w") as f:
            json.dump({}, f)
        print("ℹ️ Initialized empty symbol_scores.json")
    while True:
        try:
            print(f"=== Scan @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            # Check drawdown guard
            if is_trading_blocked():
                print("⛔ Drawdown limit reached. Skipping trading for today.\n")
                time.sleep(SCAN_INTERVAL)
                continue
            # Macro signals
            btc_d = get_btc_dominance()
            fg = get_fear_greed_index()
            sentiment = get_macro_sentiment()
            try:
                sentiment_confidence = float(sentiment.get("confidence", 5.0))
            except Exception:
                sentiment_confidence = 5.0
            sentiment_bias = str(sentiment.get("bias", "neutral"))
            # Convert types safely
            try:
                btc_d = float(btc_d)
            except Exception:
                btc_d = 0.0
            try:
                fg = int(fg)
            except Exception:
                fg = 0
            print(
                f"🌐 BTC Dominance: {btc_d:.2f}% | Fear & Greed: {fg} | Sentiment: {sentiment_bias} (Confidence: {sentiment_confidence})"
            )
            # Macro gating
            if (sentiment_bias == "bearish" and sentiment_confidence >= 7) or fg < 20 or btc_d > 60:
                reasons = []
                if sentiment_bias == "bearish" and sentiment_confidence >= 7:
                    reasons.append("strong bearish sentiment")
                if fg < 20:
                    reasons.append("extreme Fear & Greed index")
                if btc_d > 60:
                    reasons.append("very high BTC dominance")
                reason_text = " + ".join(reasons) if reasons else "unfavorable conditions"
                print(f"🚫 Market unfavorable ({reason_text}). Skipping scan.\n")
                time.sleep(SCAN_INTERVAL)
                continue
            # Load active trades and ensure only long trades remain (spot mode)
            active_trades = load_active_trades()
            for sym, trade in list(active_trades.items()):
                if trade.get("direction") == "short":
                    print(f"⚠️ Removing non-long trade {sym} from active trades (spot-only mode).")
                    del active_trades[sym]
            save_active_trades(active_trades)
            # Get top symbols to scan
            top_symbols = get_top_symbols(limit=30)
            session = get_market_session()
            potential_trades = []
            symbol_scores = {}
            # Evaluate each symbol
            for symbol in top_symbols:
                if symbol in active_trades:
                    print(f"⚠️ Skipping {symbol}: already in an active trade.")
                    continue
                try:
                    price_data = get_price_data(symbol)
                    if price_data is None or price_data.empty or len(price_data) < 40:
                        print(f"⚠️ Skipping {symbol} due to insufficient data.\n")
                        continue
                    score, direction, position_size, pattern_name = evaluate_signal(price_data, symbol)
                    symbol_scores[symbol] = {"score": score, "direction": direction}
                    # Force long direction if high score but no direction
                    if direction is None and score >= 4.5:
                        print(f"⚠️ No clear direction for {symbol} despite score={score:.2f}. Forcing 'long' direction.")
                        direction = "long"
                    # Skip non-long or invalid position sizes
                    if direction != "long" or position_size <= 0:
                        continue
                    # Order flow caution
                    flow_status = detect_aggression(price_data)
                    if flow_status == "sellers in control":
                        print(
                            f"⚠️ Bearish order flow detected in {symbol}. Proceeding with caution (penalized score handled in evaluate_signal)."
                        )
                    potential_trades.append({
                        "symbol": symbol,
                        "score": score,
                        "direction": "long",
                        "position_size": position_size,
                        "pattern": pattern_name,
                        "price_data": price_data,
                    })
                except Exception as e:
                    print(f"❌ Error evaluating {symbol}: {e}")
                    continue
            # Sort by score and select diversified signals
            potential_trades.sort(key=lambda x: x['score'], reverse=True)
            allowed_new = MAX_ACTIVE_TRADES - len(active_trades)
            opened_count = 0
            selected = select_diversified_signals(potential_trades, allowed_new)
            # Iterate over selected trade candidates and open trades
            for trade_candidate in selected:
                if opened_count >= allowed_new:
                    break
                symbol = trade_candidate['symbol']
                score = trade_candidate['score']
                position_size = trade_candidate['position_size']
                pattern_name = trade_candidate['pattern']
                price_data = trade_candidate['price_data']
                # Build indicator snapshot for LLM and brain
                try:
                    indicators_df = calculate_indicators(price_data)
                    indicators = {
                        "rsi": float(indicators_df['rsi'].iloc[-1] if 'rsi' in indicators_df else 50.0),
                        "macd": float(indicators_df['macd'].iloc[-1] if 'macd' in indicators_df else 0.0),
                        "adx": float(indicators_df['adx'].iloc[-1] if 'adx' in indicators_df else 20.0),
                    }
                except Exception:
                    indicators = {"rsi": 50.0, "macd": 0.0, "adx": 20.0}
                # Ask the brain whether to take the trade
                decision_obj = should_trade(
                    symbol=symbol,
                    score=score,
                    direction="long",
                    indicators=indicators,
                    session=session,
                    pattern_name=pattern_name,
                    orderflow="buyers" if detect_aggression(price_data) == "buyers in control" else "sellers" if detect_aggression(price_data) == "sellers in control" else "neutral",
                    sentiment=sentiment,
                    macro_news={"safe": True, "reason": ""},
                )
                decision = decision_obj.get("decision", False)
                final_conf = decision_obj.get("confidence", score)
                narrative = decision_obj.get("narrative", "")
                if not decision:
                    reason = decision_obj.get("reason", "Unknown reason")
                    print(f"🧠 Brain Decision for {symbol}: False | Confidence: {final_conf:.2f}\nReason: {reason}\n")
                    log_rejection(symbol, reason)
                    continue
                # ML model veto
                ml_prob = predict_success_probability(
                    score=score,
                    confidence=final_conf,
                    session=session,
                    btc_d=btc_d,
                    fg=fg,
                    sentiment_conf=sentiment_confidence,
                    pattern=pattern_name,
                )
                if ml_prob < 0.5:
                    print(f"🤖 ML model predicted low success probability ({ml_prob:.2f}) for {symbol}. Skipping trade.")
                    log_rejection(symbol, f"ML prob {ml_prob:.2f} too low")
                    continue
                # Blend ML probability into final confidence
                final_conf = round((final_conf + ml_prob * 10) / 2.0, 2)
                # Proceed if we still have room for a new trade
                if position_size > 0:
                    entry_price = round(price_data['close'].iloc[-1], 6)
                    # Determine dynamic SL/TP using ATR if available
                    try:
                        atr_val = indicators_df['atr'].iloc[-1] if 'atr' in indicators_df else None
                        if atr_val is not None and not np.isnan(atr_val) and atr_val > 0:
                            sl = round(entry_price - atr_val * 2.0, 6)
                            tp1 = round(entry_price + atr_val * 2.0, 6)
                            tp2 = round(entry_price + atr_val * 3.0, 6)
                            tp3 = round(entry_price + atr_val * 4.0, 6)
                        else:
                            raise ValueError("Invalid ATR")
                    except Exception:
                        # Fallback to static percentages
                        sl = round(entry_price - entry_price * 0.01, 6)
                        tp1 = round(entry_price + entry_price * 0.01, 6)
                        tp2 = round(entry_price + entry_price * 0.015, 6)
                        tp3 = round(entry_price + entry_price * 0.025, 6)
                    # Compose the new trade dictionary with extra metadata
                    new_trade = {
                        "symbol": symbol,
                        "direction": "long",
                        "entry": entry_price,
                        "entry_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),  # record open time
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "position_size": position_size,
                        "size": position_size,  # duplicate for dashboard convenience
                        "leverage": 1,  # default leverage (spot)
                        "confidence": final_conf,
                        "score": score,
                        "session": session,
                        "btc_dominance": btc_d,
                        "fear_greed": fg,
                        "sentiment_bias": sentiment_bias,
                        "sentiment_confidence": sentiment_confidence,
                        "sentiment_summary": sentiment.get("summary", ""),
                        "pattern": pattern_name,
                        "strategy": pattern_name,  # tag strategy by pattern
                        "narrative": narrative,
                        "ml_prob": ml_prob,
                        "status": {"tp1": False, "tp2": False, "tp3": False, "sl": False},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    print(f"📝 Narrative:\n{narrative}\n")
                    print(f"✅ Trade: {symbol} | Score: {score:.2f} | Position Size: ${position_size}")
                    # Add to active trades and persist
                    active_trades[symbol] = new_trade
                    create_new_trade(new_trade)
                    # Log the new trade result (open).  Provide exit_price equal to entry for completeness
                    try:
                        log_trade_result(new_trade, outcome="open", exit_price=entry_price)
                    except Exception as e:
                        print(f"⚠️ Failed to log new trade {symbol}: {e}")
                    save_active_trades(active_trades)
                    send_email(f"New Trade Opened: {symbol}", str(new_trade))
                    opened_count += 1
            # Manage existing trades after opening new ones
            manage_trades()
            # Persist symbol scores to disk
            try:
                with open("symbol_scores.json", "r") as f:
                    old_data = json.load(f)
            except FileNotFoundError:
                old_data = {}
            old_data.update(symbol_scores)
            with open("symbol_scores.json", "w") as f:
                json.dump(old_data, f, indent=4)
            print("💾 Saved symbol scores (persistent memory updated).")
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    logging.info("🚀 Starting Spot AI Super Agent loop...")
    run_agent_loop()
