"""
Improved main entry point for the Spot AI Super Agent.

This version incorporates several enhancements:

* Dynamic stop‑loss and take‑profit levels based on Average True Range (ATR) rather
  than fixed percentages.  ATR multipliers can be adjusted via environment
  variables at runtime, allowing flexible risk management without code changes.
* Background threads launch both the news fetcher and the Streamlit dashboard,
  binding the dashboard to the port specified by the ``PORT`` environment
  variable (default 10000).  This makes the service compatible with hosts
  like Render that require explicit port binding.
* Logging uses Python's ``logging`` module for structured output and easier
  monitoring.
* The agent enforces spot‑only (long) trading and filters out symbols with
  insufficient volume or price history.

To run the agent locally:

    python agent.py

The dashboard will be available at ``http://localhost:<PORT>`` where ``PORT``
defaults to 10000 unless overridden by the environment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, List

from dotenv import load_dotenv
from ta.volatility import AverageTrueRange

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
    predict_success_probability = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Maximum number of concurrent trades
MAX_ACTIVE_TRADES = int(os.getenv("MAX_ACTIVE_TRADES", 2))
# Seconds between scans
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
# Seconds between macro news fetches
NEWS_INTERVAL = int(os.getenv("NEWS_INTERVAL", 3600))

# ATR multipliers for dynamic risk management
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", 1.0))
ATR_TP1_MULT = float(os.getenv("ATR_TP1_MULT", 1.0))
ATR_TP2_MULT = float(os.getenv("ATR_TP2_MULT", 1.5))
ATR_TP3_MULT = float(os.getenv("ATR_TP3_MULT", 2.0))


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
    port = os.environ.get("PORT", "10000")
    # Use headless mode and explicit port binding.  Streamlit will bind to
    # 0.0.0.0 automatically when running with --server.headless true.
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
                f"🌐 BTC Dominance: {btc_d:.2f}% | Fear & Greed: {fg} | "
                f"Sentiment: {sentiment_bias} (Confidence: {sentiment_conf})"
            )
            # Macro gating
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
                        pattern_len = len(pattern) if pattern else 0
                        ml_prob = predict_success_probability(
                            score,
                            conf,
                            session,
                            btc_d,
                            fg,
                            sentiment_conf,
                            pattern_len,
                        )
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
                # Determine dynamic SL/TP using ATR
                try:
                    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14)
                    atr_series = atr_indicator.average_true_range()
                    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
                except Exception:
                    atr_val = data['close'].iloc[-1] * 0.01  # fallback 1% of price
                entry_price = round(data["close"].iloc[-1], 6)
                sl = round(entry_price - atr_val * ATR_SL_MULT, 6)
                tp1 = round(entry_price + atr_val * ATR_TP1_MULT, 6)
                tp2 = round(entry_price + atr_val * ATR_TP2_MULT, 6)
                tp3 = round(entry_price + atr_val * ATR_TP3_MULT, 6)
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
                try:
                    save_active_trades(active_trades)
                except Exception as e:
                    fallback_path = os.getenv("ACTIVE_TRADES_FILE", "/tmp/active_trades.json")
                    try:
                        with open(fallback_path, "w") as f:
                            json.dump(active_trades, f, indent=2)
                    except Exception as e2:
                        logger.warning(f"Unable to save active trades: {e2}")
                log_trade_result(new_trade, outcome="open")
                send_email(f"New Trade Opened: {symbol}", new_trade)
                logger.info(
                    f"✅ Opened trade {symbol} @ {entry_price} | ATR={atr_val:.6f} | ML={ml_prob:.2f} | Conf={conf:.2f}"
                )
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
