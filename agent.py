"""
Entry point for the Spot AI Super Agent.

This module orchestrates the scanning of market symbols, evaluates
potential trades using technical indicators, sentiment analysis, a
language model (LLM) and optional machine‑learning predictions.  When
conditions are satisfied, the agent opens paper trades and logs them
for subsequent learning.  Active trades are persisted to a JSON file
whose path is determined by the ``ACTIVE_TRADES_FILE`` environment
variable.  The agent also periodically fetches macro news and drives
the Streamlit dashboard in a separate thread.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from btc_dominance import get_btc_dominance
from drawdown_guard import is_trading_blocked
from fear_greed import get_fear_greed_index
from fetch_news import run_news_fetcher
from notifier import send_email, log_rejection
from orderflow import detect_aggression
from sentiment import get_macro_sentiment
from trade_logger import log_trade_result
from trade_manager import load_active_trades, save_active_trades, manage_trades
from trade_utils import get_top_symbols, get_price_data, evaluate_signal, get_market_session
from brain import should_trade

# Optional machine‑learning model for success probability
try:
    from ml_model import predict_success_probability
except Exception:
    predict_success_probability = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_ACTIVE_TRADES = 2  # Maximum number of concurrent trades
SCAN_INTERVAL = 15     # Seconds between scans
NEWS_INTERVAL = 3600   # Seconds between macro news fetches


def auto_run_news() -> None:
    """Background thread to periodically fetch crypto and macro news."""
    while True:
        logger.info("🗞️ Running scheduled news fetcher...")
        try:
            run_news_fetcher()
        except Exception as e:
            logger.exception(f"News fetcher error: {e}")
        time.sleep(NEWS_INTERVAL)


def start_dashboard() -> None:
    """Launch the Streamlit dashboard on the configured port in a thread."""
    # Streamlit picks up the PORT environment variable if provided
    port = os.environ.get("PORT", "10000")
    cmd = f"streamlit run dashboard.py --server.port {port} --server.headless true"
    os.system(cmd)


def run_agent_loop() -> None:
    logger.info("🚀 Starting Spot AI Super Agent loop...")
    # Start background tasks
    threading.Thread(target=auto_run_news, daemon=True).start()
    threading.Thread(target=start_dashboard, daemon=True).start()

    # Initialise persistent symbol scores file if missing
    if not os.path.exists("symbol_scores.json"):
        with open("symbol_scores.json", "w") as f:
            json.dump({}, f)
        logger.info("ℹ️ Initialized empty symbol_scores.json")

    while True:
        try:
            now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"=== Scan @ {now_str} ===")
            # Check drawdown guard
            if is_trading_blocked():
                logger.warning("⛔ Drawdown limit reached. Skipping trading for today.")
                time.sleep(SCAN_INTERVAL)
                continue
            # Get macro indicators
            btc_d = get_btc_dominance() or 0.0
            fg = get_fear_greed_index() or 0
            sentiment = get_macro_sentiment() or {}
            sentiment_bias = sentiment.get("bias", "neutral")
            sentiment_conf = float(sentiment.get("confidence", 5.0))
            logger.info(
                f"🌐 BTC Dominance: {btc_d:.2f}% | Fear & Greed: {fg} | Sentiment: {sentiment_bias} (Confidence: {sentiment_conf})"
            )
            # Apply basic macro gating: skip if extreme fear or high BTC dominance
            if (sentiment_bias == "bearish" and sentiment_conf >= 7) or fg < 20 or btc_d > 65:
                logger.warning("🛑 Market conditions unfavorable. Skipping this cycle.")
                time.sleep(SCAN_INTERVAL)
                continue
            # Load current active trades
            active_trades = load_active_trades()
            # Remove non‑long trades if any (spot‑only)
            for sym in list(active_trades.keys()):
                if active_trades[sym].get("direction") != "long":
                    logger.info(f"Removing non‑long trade {sym} from active trades (spot only).")
                    del active_trades[sym]
            save_active_trades(active_trades)
            # Evaluate symbols
            top_symbols = get_top_symbols(limit=30)
            potential: List[Dict[str, Any]] = []
            symbol_scores: Dict[str, Any] = {}
            session = get_market_session()
            for symbol in top_symbols:
                # Skip if already active
                if symbol in active_trades:
                    continue
                try:
                    price_data = get_price_data(symbol)
                    if price_data is None or price_data.empty or len(price_data) < 20:
                        continue
                    score, direction, position_size, pattern_name = evaluate_signal(price_data, symbol)
                    symbol_scores[symbol] = {"score": score, "direction": direction}
                    # Default to long if score high but no direction specified
                    if direction is None and score >= 4.5:
                        direction = "long"
                    if direction != "long" or position_size <= 0:
                        continue
                    # Penalty for bearish flow handled in evaluate_signal; no hard skip
                    potential.append({
                        "symbol": symbol,
                        "score": score,
                        "direction": direction,
                        "position_size": position_size,
                        "pattern": pattern_name,
                        "price_data": price_data,
                    })
                except Exception as e:
                    logger.exception(f"Error evaluating {symbol}: {e}")
            # Sort by score descending
            potential.sort(key=lambda x: x['score'], reverse=True)
            # Determine number of new trades we can open
            allowed_new = MAX_ACTIVE_TRADES - len(active_trades)
            opened = 0
            for sig in potential:
                if opened >= allowed_new:
                    break
                symbol = sig["symbol"]
                score = sig["score"]
                direction = sig["direction"]
                size = sig["position_size"]
                pattern = sig["pattern"]
                data = sig["price_data"]
                # Build indicator snapshot for brain
                indicators = {
                    "rsi": data["rsi"].iloc[-1] if "rsi" in data else 50,
                    "macd": data["macd"].iloc[-1] if "macd" in data else 0,
                    "adx": data["adx"].iloc[-1] if "adx" in data else 20,
                    "volume": data["volume"].iloc[-1] if "volume" in data else 0,
                }
                flow_tag = "buy-side aggression" if detect_aggression(data) == "buyers in control" else "neutral flow"
                brain_decision = should_trade(
                    symbol=symbol,
                    score=score,
                    direction=direction,
                    indicators=indicators,
                    session=session,
                    pattern_name=pattern,
                    orderflow=flow_tag,
                    sentiment=sentiment,
                    macro_news=sentiment  # reuse sentiment for macro
                )
                decision = brain_decision.get("decision", False)
                conf = brain_decision.get("confidence", 0.0)
                narrative = brain_decision.get("narrative", "")
                # Apply ML probability if model available
                ml_prob = 0.5
                if predict_success_probability is not None:
                    try:
                        # Feature vector: score, conf, session id (0=Asia,1=Europe,2=US), btc_d, fg, sentiment_conf
                        session_map = {"Asia": 0, "Europe": 1, "US": 2}
                        features = [
                            score,
                            conf,
                            session_map.get(session, 2),
                            btc_d,
                            fg,
                            sentiment_conf,
                        ]
                        ml_prob = predict_success_probability(features)
                    except Exception as e:
                        logger.warning(f"ML prediction error: {e}")
                        ml_prob = 0.5
                # If ML model strongly negative, veto trade
                if ml_prob < 0.4:
                    log_rejection(symbol, f"ML model probability too low ({ml_prob:.2f})")
                    continue
                # Veto if brain declines
                if not decision:
                    log_rejection(symbol, brain_decision.get("reason", "Brain veto"))
                    continue
                # Open trade
                entry_price = round(data["close"].iloc[-1], 6)
                sl = round(entry_price * 0.99, 6)
                tp1 = round(entry_price * 1.01, 6)
                tp2 = round(entry_price * 1.015, 6)
                tp3 = round(entry_price * 1.025, 6)
                new_trade = {
                    "symbol": symbol,
                    "direction": "long",
                    "entry": entry_price,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3,
                    "position_size": size,
                    "score": score,
                    "confidence": conf,
                    "ml_prob": ml_prob,
                    "session": session,
                    "btc_dominance": btc_d,
                    "fear_greed": fg,
                    "sentiment_bias": sentiment_bias,
                    "sentiment_confidence": sentiment_conf,
                    "sentiment_summary": sentiment.get("summary", ""),
                    "pattern": pattern,
                    "narrative": narrative,
                    "status": {"tp1": False, "tp2": False, "tp3": False},
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Persist trade and log open
                active_trades[symbol] = new_trade
                save_active_trades(active_trades)
                log_trade_result(new_trade, outcome="open")
                send_email(f"New Trade Opened: {symbol}", json.dumps(new_trade, indent=2))
                logger.info(f"✅ Opened trade {symbol} @ {entry_price} | ML={ml_prob:.2f} | Conf={conf:.2f}")
                opened += 1
            # Save symbol scores for context boosting
            try:
                existing = {}
                if os.path.exists("symbol_scores.json"):
                    with open("symbol_scores.json", "r") as f:
                        existing = json.load(f)
                existing.update(symbol_scores)
                with open("symbol_scores.json", "w") as f:
                    json.dump(existing, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update symbol_scores.json: {e}")
            # Manage existing trades (update status/exit) and sleep
            manage_trades()
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_agent_loop()
