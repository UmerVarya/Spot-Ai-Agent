"""
Decision‑making brain of the Spot AI Super Agent.

This improved version enhances prompt engineering for the LLM, enforces
structured JSON responses and integrates additional contextual data.  It
leverages ``groq_llm.get_llm_judgment`` which sanitises prompts and
returns a dictionary containing the advisor's decision, confidence and
reason.  Baseline confidence scoring is preserved but extended with
adaptive thresholds and retrieval augmentation.
"""

import re
from typing import Dict, Any

from sentiment import get_macro_sentiment
from groq_llm import get_llm_judgment
from confidence_guard import get_adaptive_conf_threshold
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence
from memory_retriever import get_recent_trade_summary  # retrieval augmentation

# Global symbol context memory
symbol_context_cache: Dict[str, Dict[str, Any]] = {}


def should_trade(symbol: str, score: float, direction: str | None, indicators: Dict[str, float], session: str,
                 pattern_name: str, orderflow: str, sentiment: Dict[str, Any], macro_news: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate signals and decide whether to take a long trade.

    Returns a dictionary containing ``decision`` (bool), ``confidence`` (0–10), ``reason`` (string) and an
    optional ``narrative``.  The function considers quantitative metrics,
    macro conditions, pattern memory, historical performance and an
    LLM advisor.
    """
    try:
        # === Sentiment interpretation ===
        sentiment_bias = sentiment.get("bias", "neutral")
        sentiment_confidence = float(sentiment.get("confidence", 5.0))
        # === Macro news filter ===
        if not macro_news.get("safe", True):
            return {
                "decision": False,
                "confidence": 0.0,
                "reason": "Macro news unsafe: " + macro_news.get("reason", "unknown"),
            }
        # === Adaptive Score Threshold ===
        base_threshold = get_adaptive_conf_threshold() or 4.5
        if sentiment_bias == "bullish":
            score_threshold = base_threshold - 0.3
        elif sentiment_bias == "bearish":
            score_threshold = base_threshold + 0.3
        else:
            score_threshold = base_threshold
        score_threshold = round(score_threshold, 2)
        # === Direction fallback logic ===
        if direction is None and score >= score_threshold and sentiment_bias != "bearish":
            direction = "long"
        # === Base confidence ===
        confidence = float(score)
        # === Sentiment bias adjustment ===
        if sentiment_bias == "bullish":
            confidence += 1.0
        elif sentiment_bias == "bearish":
            confidence -= 1.0
        confidence += (sentiment_confidence - 5.0) * 0.3
        # === Indicator-based adjustments (long only) ===
        if direction == "long":
            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            adx = indicators.get("adx", 20)
            if rsi > 70:
                confidence -= 1.0
            elif rsi < 30:
                confidence += 1.0
            if macd > 0:
                confidence += 0.5
            if adx > 25:
                confidence += 0.5
        # === Pattern Memory Boost ===
        memory_boost = recall_pattern_confidence(symbol, pattern_name)
        confidence += memory_boost
        # === Historical Performance Boost ===
        hist_result = calculate_historical_confidence(symbol, score, direction, session, pattern_name)
        confidence += (hist_result.get("confidence", 50) - 50) / 10.0
        # === Order Flow Boost ===
        if "buy" in orderflow.lower():
            confidence += 0.5
        elif "sell" in orderflow.lower():
            confidence -= 0.5
        # === Multi-Symbol Context Awareness ===
        symbol_context_cache[symbol] = {
            "bias": sentiment_bias,
            "direction": direction,
            "confidence": score,
        }
        if symbol != "BTCUSDT" and direction == "long":
            btc_ctx = symbol_context_cache.get("BTCUSDT", {})
            if (
                btc_ctx.get("bias") == "bullish"
                and btc_ctx.get("direction") == "long"
                and btc_ctx.get("confidence", 0) >= 6
            ):
                confidence += 0.8
        # === Normalize confidence to [0,10] ===
        final_confidence = round(max(0.0, min(confidence, 10.0)), 2)
        # === Preliminary checks ===
        if direction != "long":
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Trade direction is not long (spot only)",
            }
        if score < score_threshold:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"Score {score:.2f} below threshold {score_threshold:.2f}",
            }
        if final_confidence < 4.5:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Low confidence",
            }
        # === Retrieval Augmentation ===
        recent_summary = get_recent_trade_summary(symbol=symbol, pattern=pattern_name, max_entries=3)
        # === Build structured prompt ===
        advisor_prompt = (
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Technical Score: {score:.2f}\n"
            f"Pre‑LLM Confidence: {final_confidence:.2f}\n"
            f"Macro Sentiment: {sentiment_bias} (Confidence: {sentiment_confidence})\n"
            f"Pattern: {pattern_name}\n"
            f"Indicators: RSI {indicators.get('rsi', 0):.1f}, MACD {indicators.get('macd', 0):.4f}, ADX {indicators.get('adx', 0):.1f}\n"
            f"Orderflow: {orderflow}\n"
            f"Recent similar trades: {recent_summary}\n"
        )
        # Query LLM for structured judgment
        llm_result = get_llm_judgment(advisor_prompt)
        advisor_decision = bool(llm_result.get("decision", False))
        advisor_rating = float(llm_result.get("confidence", 0.0))
        advisor_reason = llm_result.get("reason", "")
        if advisor_rating:
            advisor_rating = max(0.0, min(advisor_rating, 10.0))
            final_confidence = round((final_confidence + advisor_rating) / 2.0, 2)
        if not advisor_decision:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"LLM advisor vetoed trade: {advisor_reason}",
            }
        # Generate narrative using same LLM helper (could use separate function)
        narrative = advisor_reason
        return {
            "decision": True,
            "confidence": final_confidence,
            "reason": "All filters passed",
            "narrative": narrative,
        }
    except Exception as e:
        return {
            "decision": False,
            "confidence": 0.0,
            "reason": f"Error in should_trade(): {e}",
        }
