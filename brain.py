"""
Decision logic for Spot AI Super Agent with structured LLM integration.

This revised ``brain.py`` retains the core confidence aggregation and
filtering pipeline while adding support for parsing JSON responses from
the LLM advisor.  The LLM is expected to return a JSON object with
``decision`` (Yes/No), ``confidence`` (0–10) and ``reason``.  If JSON
parsing fails, the module falls back to the legacy behaviour of
extracting a numeric rating via regex and inferring a yes/no from the
response prefix.

Other enhancements include minor adjustments to weighting and verbose
reason logging.
"""

import re
import json
from sentiment import get_macro_sentiment
from groq_llm import get_llm_judgment
from confidence_guard import get_adaptive_conf_threshold
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence
from narrative_builder import generate_trade_narrative
from memory_retriever import get_recent_trade_summary

# Cache for BTC context awareness
symbol_context_cache = {}


def _parse_llm_response(resp: str):
    """Attempt to parse a JSON response from the LLM.

    Returns a tuple (decision_bool, advisor_rating, reason).  If parsing
    fails, returns (None, None, raw_response).
    """
    try:
        data = json.loads(resp)
        decision = data.get("decision", "No")
        rating = data.get("confidence", None)
        reason = data.get("reason", "")
        decision_bool = str(decision).strip().lower().startswith('y')
        rating_val = None
        if isinstance(rating, (int, float, str)):
            try:
                rating_val = float(rating)
            except Exception:
                rating_val = None
        return decision_bool, rating_val, reason
    except Exception:
        return None, None, resp


def should_trade(symbol, score, direction, indicators, session, pattern_name, orderflow, sentiment, macro_news):
    """Determine whether to take a trade based on quantitative metrics and LLM guidance."""
    try:
        sentiment_bias = sentiment.get("bias", "neutral")
        sentiment_confidence = sentiment.get("score", 5.0)
        # Macro news safety
        if not macro_news.get("safe", True):
            return {"decision": False, "confidence": 0.0, "reason": "Macro news unsafe: " + macro_news.get("reason", "unknown")}
        base_threshold = get_adaptive_conf_threshold() or 4.5
        if sentiment_bias == "bullish":
            score_threshold = base_threshold - 0.3
        elif sentiment_bias == "bearish":
            score_threshold = base_threshold + 0.3
        else:
            score_threshold = base_threshold
        score_threshold = round(score_threshold, 2)
        if direction is None and score >= score_threshold and sentiment_bias != "bearish":
            direction = "long"
            print(f"🧠 Fallback direction applied: long (Sentiment: {sentiment_bias})")
        confidence = float(score)
        # Sentiment adjustments
        if sentiment_bias == "bullish":
            confidence += 1.0
        elif sentiment_bias == "bearish":
            confidence -= 1.0
        try:
            confidence += (float(sentiment_confidence) - 5.0) * 0.3
        except Exception:
            pass
        # Indicator adjustments
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
        # Pattern memory
        memory_boost = recall_pattern_confidence(symbol, pattern_name)
        confidence += memory_boost
        # Historical performance boost
        hist_result = calculate_historical_confidence(symbol, score, direction, session, pattern_name)
        try:
            confidence += (hist_result.get("confidence", 50) - 50) / 10.0
        except Exception:
            pass
        # Order flow adjustments
        if isinstance(orderflow, str):
            if "buy" in orderflow.lower():
                confidence += 0.5
            elif "sell" in orderflow.lower():
                confidence -= 0.5
        # Symbol context awareness
        symbol_context_cache[symbol] = {
            "bias": sentiment_bias,
            "direction": direction,
            "confidence": score
        }
        if symbol != "BTCUSDT" and direction == "long":
            btc_ctx = symbol_context_cache.get("BTCUSDT", {})
            if btc_ctx.get("bias") == "bullish" and btc_ctx.get("direction") == "long" and btc_ctx.get("confidence", 0) >= 6:
                confidence += 0.8
        final_confidence = round(max(0.0, min(confidence, 10.0)), 2)
        # Base gating
        if direction != "long":
            return {"decision": False, "confidence": final_confidence, "reason": "Trade direction is not long (spot-only mode)"}
        if score < score_threshold:
            return {"decision": False, "confidence": final_confidence, "reason": f"Score {score:.2f} below threshold {score_threshold:.2f}"}
        if final_confidence < 4.5:
            return {"decision": False, "confidence": final_confidence, "reason": "Low confidence"}
        # Build prompt for LLM
        recent_summary = get_recent_trade_summary(symbol=symbol, pattern=pattern_name, max_entries=3)
        advisor_prompt = (
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Technical Score: {score:.2f}\n"
            f"Current Confidence (pre‑LLM): {final_confidence:.2f}\n"
            f"Macro Sentiment: {sentiment_bias} (Confidence: {sentiment_confidence})\n"
            f"Pattern: {pattern_name}\n"
            f"Indicators: RSI {indicators.get('rsi', 0):.1f}, MACD {indicators.get('macd', 0):.4f}, ADX {indicators.get('adx', 0):.1f}\n"
            f"Recent similar trades: {recent_summary}\n\n"
            "Please perform the following analysis:\n"
            "1. Summarise the macro sentiment and any relevant macro news (if provided).\n"
            "2. Discuss any conflicting technical indicators or signals.\n"
            "3. Provide your overall trading thesis for this setup."
        )
        llm_response = get_llm_judgment(advisor_prompt)
        parsed_decision, advisor_rating, advisor_reason = _parse_llm_response(llm_response)
        if parsed_decision is None:
            # Fallback: use regex to extract a number and yes/no at start
            match = re.search(r'(\d+(?:\.\d+)?)', llm_response)
            if match:
                try:
                    advisor_rating = float(match.group(1))
                except Exception:
                    advisor_rating = None
            parsed_decision = llm_response.strip().lower().startswith("yes")
            advisor_reason = llm_response.strip()
        # Blend advisor rating
        if advisor_rating is not None:
            advisor_rating = max(0.0, min(advisor_rating, 10.0))
            final_confidence = round((final_confidence + advisor_rating) / 2.0, 2)
        if not parsed_decision:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"LLM advisor vetoed trade: {advisor_reason}"
            }
        # Generate narrative
        narrative = generate_trade_narrative(
            symbol=symbol,
            direction=direction,
            score=score,
            confidence=final_confidence,
            indicators=indicators,
            sentiment_bias=sentiment_bias,
            sentiment_confidence=sentiment_confidence,
            orderflow=orderflow,
            pattern=pattern_name,
            macro_reason=macro_news.get("reason", "")
        ) or f"No major pattern, but macro/sentiment context favors {direction} setup."
        return {
            "decision": True,
            "confidence": final_confidence,
            "reason": "All filters passed",
            "narrative": narrative
        }
    except Exception as e:
        return {"decision": False, "confidence": 0.0, "reason": f"Error in should_trade(): {e}"}
