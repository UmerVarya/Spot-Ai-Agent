import random
import re
from sentiment import get_macro_sentiment
from groq_llm import get_llm_judgment
from confidence_guard import get_adaptive_conf_threshold
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence
from narrative_builder import generate_trade_narrative  # ✅ Corrected import

# Global symbol context memory
symbol_context_cache = {}


def should_trade(symbol, score, direction, indicators, session, pattern_name, orderflow, sentiment, macro_news):
    """
    Decide whether to take a long trade based on quantitative metrics and LLM guidance.

    This function aggregates various signals—technical indicator scores, sentiment bias,
    pattern memory, historical win rates, and order‑flow cues—into a unified
    confidence score.  It then queries a language model for a sanity check,
    optionally blending the model's confidence rating with its own.  A narrative
    explaining the rationale is generated on success.
    """
    try:
        # === Sentiment interpretation ===
        sentiment_bias = sentiment.get("bias", "neutral")
        sentiment_confidence = sentiment.get("score", 5.0)

        # === Macro news filter ===
        if not macro_news.get("safe", True):
            return {
                "decision": False,
                "confidence": 0.0,
                "reason": "Macro news unsafe: " + macro_news.get("reason", "unknown")
            }

        # === Adaptive Score Threshold ===
        # Use historical performance to adapt the base threshold.  If the learning log
        # contains enough data, get_adaptive_conf_threshold() will return a
        # context‑aware value.  Otherwise fall back to a lower default (4.5).  A
        # slightly lower base allows more trades to be considered while still
        # leaving room for further confidence gating later.  Sentiment bias
        # adjusts this threshold up/down modestly.
        base_threshold = get_adaptive_conf_threshold() or 4.5
        if sentiment_bias == "bullish":
            score_threshold = base_threshold - 0.3  # easier if bullish sentiment
        elif sentiment_bias == "bearish":
            score_threshold = base_threshold + 0.3  # stricter if bearish sentiment
        else:
            score_threshold = base_threshold
        score_threshold = round(score_threshold, 2)

        # === Direction fallback logic ===
        if direction is None and score >= score_threshold and sentiment_bias != "bearish":
            direction = "long"
            print(f"🧠 Fallback direction applied: long (Sentiment: {sentiment_bias})")

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
            "confidence": score
        }
        if symbol != "BTCUSDT" and direction == "long":
            btc_ctx = symbol_context_cache.get("BTCUSDT", {})
            if btc_ctx.get("bias") == "bullish" and btc_ctx.get("direction") == "long" and btc_ctx.get("confidence", 0) >= 6:
                confidence += 0.8

        # === Normalize confidence to [0,10] ===
        final_confidence = round(max(0.0, min(confidence, 10.0)), 2)

        # === Final checks before decision ===
        if direction != "long":
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Trade direction is not long (spot-only mode)"
            }
        if score < score_threshold:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"Score {score:.2f} below threshold {score_threshold:.2f}"
            }
        # Allow lower confidence trades to progress to the LLM if they still
        # exceed a more permissive floor.  This encourages exploration of
        # promising but not perfect setups.  Trades below 4.5 are vetoed here.
        if final_confidence < 4.5:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Low confidence"
            }

        # === LLM Advisor Check ===
        advisor_prompt = (
            f"Symbol: {symbol}\n"
            f"Direction: {direction}\n"
            f"Technical Score: {score:.2f}\n"
            f"Confidence: {final_confidence:.2f}\n"
            f"Sentiment: {sentiment_bias} (Confidence: {sentiment_confidence})\n"
            f"Pattern: {pattern_name}\n"
            f"Indicators: RSI {indicators.get('rsi', 0):.1f}, MACD {indicators.get('macd', 0):.4f}, ADX {indicators.get('adx', 0):.1f}\n"
            "Should we take this trade?"
        )
        llm_response = get_llm_judgment(advisor_prompt)

        # Extract numeric rating from LLM response if provided
        advisor_rating = None
        if llm_response:
            match = re.search(r'(\d+(?:\.\d+)?)', llm_response)
            if match:
                try:
                    advisor_rating = float(match.group(1))
                except ValueError:
                    advisor_rating = None
        if advisor_rating is not None:
            # Clamp rating to [0,10] and blend with existing confidence
            advisor_rating = max(0.0, min(advisor_rating, 10.0))
            final_confidence = round((final_confidence + advisor_rating) / 2.0, 2)

        # Check LLM decision (must start with "yes")
        if not llm_response or not llm_response.strip().lower().startswith("yes"):
            reason_text = llm_response.strip() if llm_response else "No response from LLM advisor"
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"LLM advisor vetoed trade: {reason_text}"
            }
        # If LLM says "Yes", proceed to generate narrative

        # === Generate Narrative ===
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
        )
        if not narrative:
            narrative = f"No major pattern, but macro/sentiment context favors {direction} setup."

        print(f"[DEBUG] Final decision for {symbol}: direction={direction}, confidence={final_confidence}, reason='All filters passed'")

        return {
            "decision": True,
            "confidence": final_confidence,
            "reason": "All filters passed",
            "narrative": narrative
        }

    except Exception as e:
        return {
            "decision": False,
            "confidence": 0.0,
            "reason": f"Error in should_trade(): {e}"
        }
