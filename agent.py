"""
Main trading agent loop for the Spot AI Super Agent.

This script orchestrates periodic market scans, evaluates signals, asks the
brain whether to take trades, applies an ML-based veto and finally opens
trades in paper trading mode.  It also handles diversification to avoid
overexposure to correlated symbols, enforces a maximum number of open trades
and logs decisions and outcomes.

The call to ``select_diversified_signals`` was updated to pass the
``max_trades`` parameter explicitly.  Without this fix, the previously
computed ``allowed_new`` value was being treated as a correlation threshold,
resulting in at most two signals being selected regardless of available slots.
"""

from log_utils import setup_logger, LOG_FILE

logger = setup_logger(__name__)

import time
import os
import sys
import asyncio
from datetime import datetime

# Centralized configuration loader
import config
import json
import threading
from fetch_news import fetch_news, run_news_fetcher  # noqa: F401
from trade_utils import simulate_slippage, estimate_commission  # noqa: F401
from trade_utils import (
    get_top_symbols,
    get_price_data_async,
    evaluate_signal,
    get_market_session,
    calculate_indicators,
    compute_performance_metrics,
    summarise_technical_score,
)
from trade_manager import manage_trades, create_new_trade  # trade logic
from trade_storage import (
    load_active_trades,
    save_active_trades,
    ACTIVE_TRADES_FILE,
    TRADE_HISTORY_FILE,
    log_trade_result,
)
from notifier import send_email, log_rejection, REJECTED_TRADES_FILE
from trade_logger import TRADE_LEARNING_LOG_FILE
from brain import should_trade
from sentiment import get_macro_sentiment
from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index
from orderflow import detect_aggression
from diversify import select_diversified_signals
from ml_model import predict_success_probability
from sequence_model import predict_next_return, train_sequence_model, SEQ_PKL
from drawdown_guard import is_trading_blocked
import numpy as np
from rl_policy import RLPositionSizer
from trade_utils import get_rl_state
from volatility_regime import atr_percentile
from realtime_signal_cache import RealTimeSignalCache


def handle_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions with stack traces."""
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

# Maximum concurrent open trades
MAX_ACTIVE_TRADES = 2
# Interval between scans (in seconds).  Default lowered to tighten reaction time.
SCAN_INTERVAL = float(os.getenv("SCAN_INTERVAL", "3"))
# Background refresh cadence for the real-time signal cache.
SIGNAL_REFRESH_INTERVAL = float(os.getenv("SIGNAL_REFRESH_INTERVAL", "2.0"))
SIGNAL_STALE_AFTER = float(os.getenv("SIGNAL_STALE_AFTER", str(SIGNAL_REFRESH_INTERVAL * 3)))
# Interval between news fetches (in seconds)
NEWS_INTERVAL = 3600
RUN_DASHBOARD = os.getenv("RUN_DASHBOARD", "0") == "1"
rl_sizer = RLPositionSizer()

# Explicit USD bounds for trade sizing
MIN_TRADE_USD = 100.0
MAX_TRADE_USD = 200.0

logger.info(
    "Paths: LOG_FILE=%s TRADE_HISTORY=%s ACTIVE_TRADES=%s REJECTED_TRADES=%s LEARNING_LOG=%s",
    LOG_FILE,
    TRADE_HISTORY_FILE,
    ACTIVE_TRADES_FILE,
    REJECTED_TRADES_FILE,
    TRADE_LEARNING_LOG_FILE,
)


def calculate_dynamic_trade_size(confidence: float, ml_prob: float, score: float) -> float:
    """Return trade notional in USDT based on confidence signals.

    The agent expresses confidence on a 0-10 scale, the ML model supplies a
    success probability in ``[0, 1]`` and the technical analysis module
    outputs a normalized score on a 0-10 scale.  We map these inputs to a
    dollar *position size* between 100 and 200 USDT.  Values above the upper
    thresholds saturate at 200 and values below the lower bounds default to
    100.

    Parameters
    ----------
    confidence : float
        Final blended confidence score on a 0-10 scale.
    ml_prob : float
        Predicted success probability from the ML model.
    score : float
        Technical analysis score on a 0-10 scale.

    Returns
    -------
    float
        Trade size in USDT.
    """

    # Define the ranges over which size scales
    min_conf, max_conf = 4.5, 8.0
    min_prob, max_prob = 0.5, 0.7
    min_score, max_score = 4.5, 8.0

    conf_norm = (confidence - min_conf) / (max_conf - min_conf)
    prob_norm = (ml_prob - min_prob) / (max_prob - min_prob)
    score_norm = (score - min_score) / (max_score - min_score)
    conf_norm = max(0.0, min(1.0, conf_norm))
    prob_norm = max(0.0, min(1.0, prob_norm))
    score_norm = max(0.0, min(1.0, score_norm))

    edge = (conf_norm + prob_norm + score_norm) / 3.0
    trade_usd = MIN_TRADE_USD + edge * (MAX_TRADE_USD - MIN_TRADE_USD)
    return max(MIN_TRADE_USD, min(MAX_TRADE_USD, trade_usd))


def macro_filter_decision(btc_dom: float, fear_greed: int, bias: str, conf: float):
    """Return macro gating decision.

    Parameters
    ----------
    btc_dom : float
        Bitcoin dominance percentage.
    fear_greed : int
        Fear & Greed index value.
    bias : str
        Macro sentiment bias.
    conf : float
        Confidence in the sentiment assessment.

    Returns
    -------
    tuple
        ``(skip_all, skip_alt, reasons)`` where ``reasons`` is a list of
        explanatory strings.
    """
    skip_all = False
    skip_alt = False
    reasons: list[str] = []

    if fear_greed < 10:
        skip_all = True
        reasons.append("extreme fear (FG < 10)")
    elif bias == "bearish" and conf >= 8.0 and fear_greed < 15:
        skip_all = True
        reasons.append("very bearish sentiment with deep fear")

    alt_risk_score = 0
    if btc_dom > 60.0:
        alt_risk_score += 1
    if fear_greed < 20:
        alt_risk_score += 1
    if bias == "bearish" and conf >= 6.0:
        alt_risk_score += 1

    if not skip_all and alt_risk_score >= 2:
        skip_alt = True
        if btc_dom > 60.0:
            reasons.append("very high BTC dominance")
        if fear_greed < 20:
            reasons.append("low Fear & Greed")
        if bias == "bearish" and conf >= 6.0:
            reasons.append("bearish sentiment")

    return skip_all, skip_alt, reasons


def dynamic_max_active_trades(fear_greed: int, bias: str, volatility: float | None) -> int:
    """Determine allowable concurrent trades based on macro conditions.

    The baseline is ``MAX_ACTIVE_TRADES``.  During fearful or bearish
    environments, the cap is reduced to 1.  When conditions are both
    bullish and confident, one extra slot is granted.  Volatility acts as
    a secondary modifier: very high volatility removes a slot while very
    low volatility can add one, within bounds.
    """

    max_trades = MAX_ACTIVE_TRADES
    if fear_greed < 25 or bias == "bearish":
        max_trades = 1
    elif fear_greed > 70 and bias == "bullish":
        max_trades = MAX_ACTIVE_TRADES + 1

    if volatility is not None and not np.isnan(volatility):
        if volatility > 0.75:
            max_trades = max(1, max_trades - 1)
        elif volatility < 0.25:
            max_trades = min(max_trades + 1, MAX_ACTIVE_TRADES + 1)

    return max_trades


def auto_run_news() -> None:
    """Background thread that periodically fetches news."""
    while True:
        logger.info("Running scheduled news fetcher...")
        run_news_fetcher()
        time.sleep(NEWS_INTERVAL)


def run_streamlit() -> None:
    """Launch the Streamlit dashboard using Streamlit's Python API."""
    port = os.environ.get("PORT", "10000")
    import sys
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        "dashboard.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    stcli.main()


def run_agent_loop() -> None:
    """Main loop that scans the market, evaluates signals and opens trades."""
    logger.info("Spot AI Super Agent running in paper trading mode...")
    # start news and dashboard threads
    threading.Thread(target=auto_run_news, daemon=True).start()
    if RUN_DASHBOARD:
        threading.Thread(target=run_streamlit, daemon=True).start()
    # Ensure symbol_scores.json exists
    if not os.path.exists("symbol_scores.json"):
        with open("symbol_scores.json", "w") as f:
            json.dump({}, f)
        logger.info("Initialized empty symbol_scores.json")
    signal_cache = RealTimeSignalCache(
        price_fetcher=get_price_data_async,
        evaluator=evaluate_signal,
        refresh_interval=SIGNAL_REFRESH_INTERVAL,
        stale_after=SIGNAL_STALE_AFTER,
    )
    signal_cache.start()
    while True:
        try:
            logger.info("=== Scan @ %s ===", time.strftime('%Y-%m-%d %H:%M:%S'))
            # Check drawdown guard
            if is_trading_blocked():
                logger.warning("Drawdown limit reached. Skipping trading for today.")
                time.sleep(SCAN_INTERVAL)
                continue
            perf = compute_performance_metrics()
            if perf.get("max_drawdown", 0) < -0.25:
                logger.warning("Max drawdown exceeded 25%. Halting trading.")
                time.sleep(SCAN_INTERVAL)
                continue
            # Macro signals
            try:
                btc_d = get_btc_dominance()
                fg = get_fear_greed_index()
                sentiment = get_macro_sentiment()
            except Exception as e:
                logger.error("Macro data fetch error: %s", e, exc_info=True)
                time.sleep(SCAN_INTERVAL)
                continue
            # Extract sentiment bias and confidence safely
            try:
                sentiment_confidence = float(sentiment.get("confidence", 5.0))
            except Exception:
                sentiment_confidence = 5.0
            sentiment_bias = str(sentiment.get("bias", "neutral"))
            # Convert BTC dominance and Fear & Greed to numeric values
            try:
                btc_d = float(btc_d)
            except Exception:
                btc_d = 0.0
            try:
                fg = int(fg)
            except Exception:
                fg = 0
            logger.info(
                "BTC Dominance: %.2f%% | Fear & Greed: %s | Sentiment: %s (Confidence: %s)",
                btc_d,
                fg,
                sentiment_bias,
                sentiment_confidence,
            )
            skip_all, skip_alt, macro_reasons = macro_filter_decision(
                btc_d, fg, sentiment_bias, sentiment_confidence
            )
            signal_cache.update_context(sentiment_bias=sentiment_bias)
            if skip_all:
                reason_text = " + ".join(macro_reasons) if macro_reasons else "unfavorable conditions"
                logger.warning("Market unfavorable (%s). Skipping scan.", reason_text)
                time.sleep(SCAN_INTERVAL)
                continue
            btc_vol = float("nan")
            try:
                btc_df = asyncio.run(get_price_data_async("BTCUSDT"))
                if btc_df is not None:
                    btc_vol = atr_percentile(
                        btc_df["high"], btc_df["low"], btc_df["close"]
                    )
            except Exception as e:
                logger.warning("Could not compute BTC volatility: %s", e)
            max_active_trades = dynamic_max_active_trades(
                fg, sentiment_bias, btc_vol
            )
            logger.info(
                "Max active trades dynamically set to %d (BTC vol=%.2f)",
                max_active_trades,
                btc_vol,
            )
            # If we are only skipping altcoins, filter top_symbols down to BTCUSDT
            if skip_alt:
                # We will filter top_symbols later after fetching them
                macro_reason_text = " + ".join(macro_reasons) if macro_reasons else "macro caution"
            else:
                macro_reason_text = ""
            # Load active trades and ensure only long trades remain (spot mode)
            active_trades_raw = load_active_trades()
            active_trades = []
            for t in active_trades_raw:
                if t.get("direction") == "short":
                    logger.warning(
                        "Removing non-long trade %s from active trades (spot-only mode).",
                        t.get("symbol"),
                    )
                    continue
                active_trades.append(t)
            save_active_trades(active_trades)
            # Get top symbols to scan
            try:
                top_symbols = get_top_symbols(limit=30)
            except Exception as e:
                logger.error("Error fetching top symbols: %s", e, exc_info=True)
                top_symbols = []
            if not top_symbols:
                logger.warning(
                    "No symbols fetched from Binance. Check your python-binance installation and network connectivity."
                )
            # Apply macro filtering to symbols: if macro filter indicated to skip altcoins,
            # restrict the universe to BTCUSDT only.  We do this after fetching the symbols
            # to avoid unnecessary API calls during the gating step.
            if skip_alt:
                # keep BTCUSDT and potentially stablecoins if you wish; here we only keep BTCUSDT
                top_symbols = [sym for sym in top_symbols if sym.upper() == "BTCUSDT"]
                if macro_reason_text:
                    logger.warning("Macro gating (%s). Scanning only BTCUSDT.", macro_reason_text)
            session = get_market_session()
            potential_trades: list[dict] = []
            symbol_scores: dict[str, dict[str, float | None]] = {}
            symbols_to_fetch = [
                sym for sym in top_symbols if not any(t.get("symbol") == sym for t in active_trades)
            ]
            signal_cache.update_universe(symbols_to_fetch)
            for symbol in symbols_to_fetch:
                try:
                    cached_signal = signal_cache.get(symbol)
                    if cached_signal is None:
                        logger.debug("No fresh cache entry for %s yet; skipping this cycle.", symbol)
                        continue
                    price_data = cached_signal.price_data
                    if price_data is None or price_data.empty or len(price_data) < 40:
                        logger.warning("Skipping %s due to insufficient data.", symbol)
                        continue
                    score = cached_signal.score
                    direction = cached_signal.direction
                    position_size = cached_signal.position_size
                    pattern_name = cached_signal.pattern
                    logger.info(
                        "%s: Score=%.2f, Direction=%s, Pattern=%s, PosSize=%s (age=%.2fs)",
                        symbol,
                        score,
                        direction,
                        pattern_name,
                        position_size,
                        cached_signal.age(),
                    )
                    symbol_scores[symbol] = {"score": score, "direction": direction}
                    if direction is None and score >= 4.5:
                        logger.warning(
                            "No clear direction for %s despite score=%.2f. Forcing 'long' direction.",
                            symbol,
                            score,
                        )
                        direction = "long"
                    if direction != "long" or position_size <= 0:
                        skip_reasons: list[str] = []
                        if direction != "long":
                            if direction is None:
                                skip_reasons.append("no long signal (score below cutoff)")
                            else:
                                skip_reasons.append("direction not long")
                        if position_size <= 0:
                            skip_reasons.append("zero position (low confidence)")
                        reason_text = " and ".join(skip_reasons) if skip_reasons else "not eligible"
                        logger.info(
                            "[SKIP] %s: direction=%s, size=%s – %s, Score=%.2f",
                            symbol,
                            direction,
                            position_size,
                            reason_text,
                            score,
                        )
                        continue
                    flow_analysis = detect_aggression(
                        price_data,
                        symbol=symbol,
                        live_trades=price_data.attrs.get("live_trades"),
                    )
                    flow_status = getattr(flow_analysis, "state", "neutral")
                    if flow_status == "sellers in control":
                        logger.warning(
                            "Bearish order flow detected in %s. Proceeding with caution (penalized score handled in evaluate_signal).",
                            symbol,
                        )
                    potential_trades.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "direction": "long",
                            "position_size": position_size,
                            "pattern": pattern_name,
                            "price_data": price_data,
                            "orderflow": flow_analysis,
                        }
                    )
                    logger.info(
                        "[Potential Trade] %s | Score=%.2f | Direction=long | Size=%s",
                        symbol,
                        score,
                        position_size,
                    )
                except Exception as e:
                    logger.error("Error evaluating %s: %s", symbol, e, exc_info=True)
                    continue
            # Sort by score and select diversified signals
            potential_trades.sort(key=lambda x: x['score'], reverse=True)
            allowed_new = max_active_trades - len(active_trades)
            opened_count = 0
            # Pass allowed_new as max_trades to avoid misassigning correlation threshold
            selected = select_diversified_signals(potential_trades, max_trades=allowed_new)
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
                    indicators_df = price_data
                    indicators = {"rsi": 50.0, "macd": 0.0, "adx": 20.0}
                next_ret = 0.0
                try:
                    if not os.path.exists(SEQ_PKL):
                        train_sequence_model(indicators_df)
                    next_ret = predict_next_return(indicators_df.tail(10))
                except Exception:
                    pass
                indicators['next_return'] = next_ret
                # Precompute order flow once for reuse
                flow_analysis = trade_candidate.get("orderflow")
                if flow_analysis is None:
                    flow_analysis = detect_aggression(
                        price_data,
                        symbol=symbol,
                        live_trades=price_data.attrs.get("live_trades"),
                    )
                of_state = getattr(flow_analysis, "state", "neutral")
                orderflow = (
                    "buyers" if of_state == "buyers in control" else
                    "sellers" if of_state == "sellers in control" else
                    "neutral"
                )
                signal_snapshot = price_data.attrs.get("signal_features", {}) or {}
                flow_features = getattr(flow_analysis, "features", {}) or {}
                order_imb_feature = signal_snapshot.get("order_book_imbalance")
                if order_imb_feature is None:
                    order_imb_feature = flow_features.get("order_book_imbalance")
                if order_imb_feature != order_imb_feature or order_imb_feature is None:
                    order_imb_feature = flow_features.get("trade_imbalance")
                try:
                    order_imb_ratio = float(order_imb_feature)
                except (TypeError, ValueError):
                    order_imb_ratio = 0.0
                if order_imb_ratio != order_imb_ratio:
                    order_imb_ratio = 0.0
                order_imb = float(order_imb_ratio) * 100.0
                macro_ind = (
                    100.0
                    if sentiment_bias == "bullish"
                    else -100.0 if sentiment_bias == "bearish" else 0.0
                )
                # Symbol-specific volatility
                try:
                    sym_vol_pct = atr_percentile(
                        price_data["high"], price_data["low"], price_data["close"]
                    )
                    sym_vol = sym_vol_pct * 100.0
                except Exception:
                    sym_vol_pct = float("nan")
                    sym_vol = 0.0
                try:
                    ema20 = indicators_df['ema_20'].iloc[-1]
                    ema50 = indicators_df['ema_50'].iloc[-1]
                    last_close_price = float(price_data['close'].iloc[-1])
                    if last_close_price:
                        htf_trend_pct = ((ema20 - ema50) / last_close_price) * 100.0
                    else:
                        htf_trend_pct = 0.0
                except Exception:
                    htf_trend_pct = 0.0
                micro_feature_payload = {
                    'volatility': sym_vol,
                    'htf_trend': htf_trend_pct,
                    'order_imbalance': order_imb,
                    'macro_indicator': macro_ind,
                    'sent_bias': sentiment_bias,
                    'order_flow_score': signal_snapshot.get('order_flow_score'),
                    'order_flow_flag': signal_snapshot.get('order_flow_flag'),
                    'cvd': signal_snapshot.get('cvd'),
                    'cvd_change': signal_snapshot.get('cvd_change'),
                    'taker_buy_ratio': signal_snapshot.get('taker_buy_ratio'),
                    'trade_imbalance': signal_snapshot.get('trade_imbalance'),
                    'aggressive_trade_rate': signal_snapshot.get('aggressive_trade_rate'),
                    'spoofing_intensity': signal_snapshot.get('spoofing_intensity'),
                    'spoofing_alert': signal_snapshot.get('spoofing_alert'),
                    'volume_ratio': signal_snapshot.get('volume_ratio'),
                    'price_change_pct': signal_snapshot.get('price_change_pct'),
                    'spread_bps': signal_snapshot.get('spread_bps'),
                }
                # Ask the brain whether to take the trade
                # Call the brain with error handling.  If the brain throws an
                # exception, record it as the reason for skipping so users
                # understand why a potential trade was vetoed.
                try:
                    decision_obj = should_trade(
                        symbol=symbol,
                        score=score,
                        direction="long",
                        indicators=indicators,
                        session=session,
                        pattern_name=pattern_name,
                        orderflow=orderflow,
                        sentiment=sentiment,
                        macro_news={"safe": True, "reason": ""},
                        volatility=sym_vol_pct,
                        fear_greed=fg,
                    )
                except Exception as e:
                    logger.error("Error in should_trade for %s: %s", symbol, e, exc_info=True)
                    decision_obj = {
                        "decision": False,
                        "confidence": 0.0,
                        "reason": f"Error in should_trade(): {e}",
                    }
                decision = bool(decision_obj.get("decision", False))
                final_conf = float(decision_obj.get("confidence", score))
                narrative = decision_obj.get("narrative", "")
                reason = decision_obj.get("reason", "")
                llm_signal = decision_obj.get("llm_decision")
                llm_approval = decision_obj.get("llm_approval")
                technical_score = decision_obj.get("technical_indicator_score")
                if technical_score is None:
                    technical_score = summarise_technical_score(indicators, "long")
                logger.info(
                    "[Brain] %s -> %s | Confidence: %.2f | Reason: %s",
                    symbol,
                    decision,
                    final_conf,
                    reason,
                )
                if not decision:
                    log_rejection(symbol, reason or "Unknown reason")
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
                    llm_approval=bool(llm_approval) if llm_approval is not None else True,
                    llm_confidence=decision_obj.get("llm_confidence", 5.0),
                    micro_features=micro_feature_payload,
                )
                if ml_prob < 0.5:
                    logger.info(
                        "ML model predicted low success probability (%.2f) for %s. Skipping trade.",
                        ml_prob,
                        symbol,
                    )
                    log_rejection(symbol, f"ML prob {ml_prob:.2f} too low")
                    continue
                # Blend ML probability into final confidence
                final_conf = round((final_conf + ml_prob * 10) / 2.0, 2)
                # Proceed if we still have room for a new trade
                if position_size > 0:
                    try:
                        entry_price = round(price_data['close'].iloc[-1], 6)
                        try:
                            atr_val = indicators_df['atr'].iloc[-1] if 'atr' in indicators_df else None
                        except Exception:
                            atr_val = None
                        trade_usd = calculate_dynamic_trade_size(final_conf, ml_prob, score)
                        if atr_val is not None and not np.isnan(atr_val) and atr_val > 0:
                            sl_dist = atr_val * 2.0
                        else:
                            atr_val = entry_price * 0.02
                            sl_dist = atr_val * 2.0
                        state = get_rl_state(sym_vol_pct)
                        mult = rl_sizer.select_multiplier(state)
                        trade_usd *= mult
                        trade_usd = max(MIN_TRADE_USD, min(MAX_TRADE_USD, trade_usd))
                        position_size = round(max(trade_usd / entry_price, 0), 6)
                        sl = round(entry_price - sl_dist, 6)
                        tp1 = round(entry_price + atr_val * 1.0, 6)
                        tp2 = round(entry_price + atr_val * 2.0, 6)
                        tp3 = round(entry_price + atr_val * 3.0, 6)
                        htf_trend = htf_trend_pct
                        risk_amount = position_size * sl_dist
                        # Determine the high-level strategy label.  The
                        # historical trade log expects ``strategy`` to describe
                        # the approach (e.g. purely technical vs. LLM filtered)
                        # while ``pattern`` captures the specific signal.  The
                        # previous implementation copied ``pattern_name`` into
                        # both fields which obscured whether an LLM gate was
                        # involved.  We now derive a descriptive label that
                        # preserves that distinction.
                        strategy_label = "PatternTrade"
                        llm_decision_token = str(llm_signal or "").strip().lower()
                        if llm_approval is True:
                            strategy_label = "PatternTrade+LLM"
                        elif llm_approval is False:
                            strategy_label = "PatternTrade-LLM"
                        elif llm_decision_token in {"approved", "greenlight", "yes"}:
                            strategy_label = "PatternTrade+LLM"
                        elif llm_decision_token in {"vetoed", "rejected", "no"}:
                            strategy_label = "PatternTrade-LLM"

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
                            "size": trade_usd,
                            "initial_size": position_size,  # track original size for partial exits
                            "risk_amount": risk_amount,
                            "rl_state": state,
                            "rl_multiplier": mult,
                            "leverage": 1,  # default leverage (spot)
                            "confidence": final_conf,
                            "score": score,
                            "session": session,
                            "btc_dominance": btc_d,
                            "fear_greed": fg,
                            "sentiment_bias": sentiment_bias,
                            "sentiment_confidence": sentiment_confidence,
                            "sentiment_summary": sentiment.get("summary", ""),
                            "volatility": sym_vol,
                            "htf_trend": htf_trend,
                            "order_imbalance": order_imb,
                            "order_flow_score": signal_snapshot.get("order_flow_score"),
                            "order_flow_flag": signal_snapshot.get("order_flow_flag"),
                            "order_flow_state": signal_snapshot.get("order_flow_state"),
                            "cvd": signal_snapshot.get("cvd"),
                            "cvd_change": signal_snapshot.get("cvd_change"),
                            "taker_buy_ratio": signal_snapshot.get("taker_buy_ratio"),
                            "trade_imbalance": signal_snapshot.get("trade_imbalance"),
                            "aggressive_trade_rate": signal_snapshot.get("aggressive_trade_rate"),
                            "spoofing_intensity": signal_snapshot.get("spoofing_intensity"),
                            "spoofing_alert": signal_snapshot.get("spoofing_alert"),
                            "volume_ratio": signal_snapshot.get("volume_ratio"),
                            "price_change_pct": signal_snapshot.get("price_change_pct"),
                            "spread_bps": signal_snapshot.get("spread_bps"),
                            "macro_indicator": macro_ind,
                            "pattern": pattern_name,
                            "strategy": strategy_label,
                            "narrative": narrative,
                            "ml_prob": ml_prob,
                            "llm_decision": llm_signal,
                            "llm_approval": llm_approval,
                            "llm_confidence": decision_obj.get("llm_confidence"),
                            "llm_error": decision_obj.get("llm_error"),
                            "technical_indicator_score": technical_score,
                            "status": {"tp1": False, "tp2": False, "tp3": False, "sl": False},
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "news_summary": decision_obj.get("news_summary", ""),
                        }
                        logger.info("Narrative:\n%s\n", narrative)
                        logger.info(
                            "Trade Opened %s @ %s | Notional=%s USD | Qty=%s | TP1 %s / TP2 %s / TP3 %s",
                            symbol,
                            entry_price,
                            trade_usd,
                            position_size,
                            tp1,
                            tp2,
                            tp3,
                        )
                        # Add to active trades and persist if not a duplicate
                        if create_new_trade(new_trade):
                            active_trades.append(new_trade)
                            # Do NOT log the open trade as a completed trade.  It will be logged upon exit by trade_manager.py.
                            save_active_trades(active_trades)
                            send_email(
                                f"New Trade Opened: {symbol}",
                                f"{new_trade}\n\n Narrative:\n{narrative}\n\nNews Summary:\n{decision_obj.get('news_summary', '')}",
                            )
                            opened_count += 1
                        else:
                            logger.info("Trade for %s already active; skipping new entry", symbol)
                    except Exception as e:
                        logger.error("Error opening trade for %s: %s", symbol, e, exc_info=True)
            # Manage existing trades after opening new ones
            try:
                manage_trades()
            except Exception as e:
                logger.error("Error managing trades: %s", e, exc_info=True)
            # Persist symbol scores to disk
            try:
                with open("symbol_scores.json", "r") as f:
                    old_data = json.load(f)
            except FileNotFoundError:
                old_data = {}
            except Exception as e:
                logger.error("Error reading symbol_scores.json: %s", e, exc_info=True)
                old_data = {}
            try:
                old_data.update(symbol_scores)
                with open("symbol_scores.json", "w") as f:
                    json.dump(old_data, f, indent=4)
                logger.info("Saved symbol scores (persistent memory updated).")
            except Exception as e:
                logger.error("Error saving symbol scores: %s", e, exc_info=True)
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            logger.error("Main Loop Error: %s", e, exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    logger.info("Starting Spot AI Super Agent loop...")
    run_agent_loop()
