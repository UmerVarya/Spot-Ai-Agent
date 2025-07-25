import logging
logging.basicConfig(level=logging.INFO)
logging.info("agent.py starting...")

import time
import os
from dotenv import load_dotenv
load_dotenv()
import json
import threading
from fetch_news import fetch_news, run_news_fetcher
from trade_utils import simulate_slippage, estimate_commission
from trade_utils import get_top_symbols, get_price_data, evaluate_signal, get_market_session
from trade_manager import manage_trades, load_active_trades, save_active_trades, create_new_trade
from notifier import send_email, log_rejection
from trade_logger import log_trade_result  # record entries
from brain import should_trade
from sentiment import get_macro_sentiment
from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index
from orderflow import detect_aggression
from diversify import select_diversified_signals  # diversification helper
from ml_model import predict_success_probability  # ML success predictor
from drawdown_guard import is_trading_blocked  # ✅ Drawdown Guard

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
    port = os.environ.get("PORT", "10000")  # Render provides a dynamic port
    os.system(f"streamlit run dashboard.py --server.port {port} --server.headless true")


def run_agent_loop():
    print("\n🤖 Spot AI Super Agent running in paper trading mode...\n")
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

            # Check if trading is temporarily blocked (e.g., drawdown limit hit)
            if is_trading_blocked():
                print("⛔ Drawdown limit reached. Skipping trading for today.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            # Get macro context indicators
            btc_d = get_btc_dominance()
            fg = get_fear_greed_index()
            sentiment = get_macro_sentiment()
            try:
                sentiment_confidence = float(sentiment.get("confidence", 5.0))
            except:
                sentiment_confidence = 5.0
            sentiment_bias = str(sentiment.get("bias", "neutral"))
            try:
                btc_d = float(btc_d)
            except:
                btc_d = 0.0
            try:
                fg = int(fg)
            except:
                fg = 0

            print(f"🌐 BTC Dominance: {btc_d:.2f}% | Fear & Greed: {fg} | Sentiment: {sentiment_bias} (Confidence: {sentiment_confidence})")

            # Market sentiment gating – skip trading if conditions indicate a strongly bearish environment
            if (sentiment_bias == "bearish" and sentiment_confidence >= 7) or fg < 20 or btc_d > 60:
                reasons = []
                if sentiment_bias == "bearish" and sentiment_confidence >= 7:
                    reasons.append("strong bearish sentiment")
                if fg < 20:
                    reasons.append("extreme Fear & Greed index")
                if btc_d > 60:
                    reasons.append("very high BTC dominance")
                reason_text = " + ".join(reasons) if reasons else "unfavorable conditions"
                print(f"🛑 Market unfavorable ({reason_text}). Skipping scan.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            active_trades = load_active_trades()
            # Purge any remaining short trades from active trades (spot-only mode)
            for sym, trade in list(active_trades.items()):
                if trade.get("direction") == "short":
                    print(f"⛔ Removing non-long trade {sym} from active trades (spot-only mode).")
                    del active_trades[sym]
            save_active_trades(active_trades)

            top_symbols = get_top_symbols(limit=30)
            # Determine the current market session for logging and learning
            session = get_market_session()
            potential_trades = []  # collect potential trade signals (unsorted)
            symbol_scores = {}     # save scores for context boosting

            for symbol in top_symbols:
                # Enforce max concurrency but still evaluate all symbols for logging
                if symbol in active_trades:
                    print(f"⚠️ Skipping {symbol}: already in an active trade.")
                    continue

                try:
                    price_data = get_price_data(symbol)
                    if price_data is None or price_data.empty or len(price_data) < 20:
                        print(f"⚠️ Skipping {symbol} due to insufficient data.\n")
                        continue

                    score, direction, position_size, pattern_name = evaluate_signal(price_data, symbol)
                    symbol_scores[symbol] = {"score": score, "direction": direction}
                    # ✅ If no clear direction but score meets threshold, assume long (neutral sentiment fallback)
                    if direction is None and score >= 4.5:  # lower fallback threshold to align with evaluate_signal
                        print(f"⚠️ No clear direction for {symbol} despite score={score:.2f}. Forcing 'long' direction.")
                        direction = "long"

                    # Skip if signal does not qualify (we only trade long spot positions)
                    if direction != "long" or position_size <= 0:
                        # Skip reason already logged in evaluate_signal
                        continue

                    # Order flow check – treat bearish aggression as penalty, not veto
                    flow_status = detect_aggression(price_data)
                    if flow_status == "sellers in control":
                        print(f"⚠️ Bearish order flow detected in {symbol}. Proceeding with caution (penalized score handled in evaluate_signal).")
                    # No explicit skip for neutral flow; rely on evaluate_signal/brain to decide

                    # If we reach here, we have a valid potential **long** trade signal
                    potential_trades.append({
                        "symbol": symbol,
                        "score": score,
                        "direction": "long",
                        "position_size": position_size,
                        "pattern": pattern_name,
                        "price_data": price_data
                    })

                except Exception as e:
                    print(f"❌ Error evaluating {symbol}: {e}")
                    continue

            # Rank potential trades by score (highest first)
            potential_trades.sort(key=lambda x: x['score'], reverse=True)
            # Determine how many new trades we can open this cycle
            allowed_new = MAX_ACTIVE_TRADES - len(active_trades)
            opened_count = 0

            # Diversify signals: avoid opening highly correlated trades
            # Only select up to ``allowed_new`` trades.
            diversified_signals = select_diversified_signals(potential_trades, max_corr=0.8, max_trades=allowed_new)

            for signal in diversified_signals:
                symbol = signal["symbol"]
                score = signal["score"]
                direction = signal["direction"]
                position_size = signal["position_size"]
                pattern_name = signal["pattern"]
                price_data = signal["price_data"]

                if opened_count >= allowed_new:
                    print(f"⚠️ Skipping {symbol} due to concurrency cap (max {MAX_ACTIVE_TRADES} trades).")
                    continue

                # Prepare indicators for decision and consult the brain module
                indicators = {
                    "rsi": price_data["rsi"].iloc[-1] if "rsi" in price_data else 50,
                    "macd": price_data["macd"].iloc[-1] if "macd" in price_data else 0,
                    "adx": price_data["adx"].iloc[-1] if "adx" in price_data else 20,
                    "volume": price_data["volume"].iloc[-1] if "volume" in price_data else 0,
                }
                orderflow_tag = "buy-side aggression" if detect_aggression(price_data) == "buyers in control" else "neutral flow"

                # Pass the actual session to the brain for session-specific adjustments
                decision_obj = should_trade(
                    symbol=symbol,
                    score=score,
                    direction=direction,
                    indicators=indicators,
                    session=session,
                    pattern_name=pattern_name,
                    orderflow=orderflow_tag,
                    sentiment=sentiment,
                    macro_news=sentiment  # using macro sentiment as macro_news context
                )
                decision = decision_obj.get("decision", False)
                final_conf = decision_obj.get("confidence", 0.0)
                narrative = decision_obj.get("narrative", "")

                # --- Machine Learning Adjustment ---
                # Use the trained ML model to predict probability of success
                try:
                    ml_prob = predict_success_probability(
                        score=score,
                        confidence=final_conf,
                        session=session,
                        btc_d=btc_d,
                        fg=fg,
                        sentiment_conf=sentiment_confidence,
                        pattern=pattern_name
                    )
                except Exception:
                    ml_prob = 0.5
                # If predicted probability is low, skip the trade
                if ml_prob < 0.5:
                    print(f"🤖 ML model predicted low success probability ({ml_prob:.2f}) for {symbol}. Skipping trade.")
                    log_rejection(symbol, f"ML prob {ml_prob:.2f} too low")
                    continue
                # Blend the ML probability into the final confidence
                final_conf = round((final_conf + ml_prob * 10) / 2.0, 2)

                # Re-evaluate decision: we rely on brain's LLM decision as gate
                if not decision:
                    reason = decision_obj.get("reason", "Unknown reason")
                    print(f"🤖 Brain Decision for {symbol}: False | Confidence: {final_conf:.2f}\nReason: {reason}\n")
                    log_rejection(symbol, reason)
                    continue
                else:
                    print(f"🤖 Brain Decision for {symbol}: True | Confidence: {final_conf:.2f}\n{narrative}\n")

                # If brain approves and ML confidence is acceptable, execute trade
                if direction == "long" and position_size > 0:
                    entry_price = round(price_data['close'].iloc[-1], 6)
                    sl = round(entry_price - entry_price * 0.01, 6)
                    tp1 = round(entry_price + entry_price * 0.01, 6)
                    tp2 = round(entry_price + entry_price * 0.015, 6)
                    tp3 = round(entry_price + entry_price * 0.025, 6)

                    new_trade = {
                        "symbol": symbol,
                        "direction": "long",
                        "entry": entry_price,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "position_size": position_size,
                        # Store the final blended confidence for learning log
                        "confidence": final_conf,
                        # Persist the normalized technical score for learning log
                        "score": score,
                        # Add session to allow session-specific learning
                        "session": session,
                        "btc_dominance": btc_d,
                        "fear_greed": fg,
                        "sentiment_bias": sentiment_bias,
                        "sentiment_confidence": sentiment_confidence,
                        "sentiment_summary": sentiment.get("summary", ""),
                        "pattern": pattern_name,
                        "narrative": narrative,
                        "ml_prob": ml_prob,
                        "status": {"tp1": False, "tp2": False, "tp3": False},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    print(f"📖 Narrative:\n{narrative}\n")
                    print(f"✅ Trade: {symbol} | Score: {score:.2f} | Position Size: ${position_size}")
                    active_trades[symbol] = new_trade
                    create_new_trade(new_trade)
                    try:
                        log_trade_result(new_trade, outcome="open")
                    except Exception as e:
                        print(f"⚠️ Failed to log new trade {symbol}: {e}")
                    save_active_trades(active_trades)
                    send_email(f"New Trade Opened: {symbol}", str(new_trade))
                    opened_count += 1

            # Update and manage any open trades, then pause
            manage_trades()
            # Save symbol_scores to JSON (persistent)
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
