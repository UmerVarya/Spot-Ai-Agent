import random
from sentiment import get_macro_sentiment
from groq_llm import get_llm_judgment
from confidence_guard import get_adaptive_conf_threshold
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence
from narrative_builder import generate_trade_narrative  # ✅ Corrected import

# Global symbol context memory
symbol_context_cache = {}

def should_trade(symbol, score, direction, indicators, session, pattern_name, orderflow, sentiment, macro_news):
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

        # === Adaptive Score Threshold (base threshold 5.5) ===
        base_threshold = 5.5
        if sentiment_bias == "bullish":
            score_threshold = base_threshold - 0.3  # slightly easier if bullish sentiment
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
        if final_confidence < 5.5:
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
            "Should we take this trade? Answer with Yes or No and a brief reason."
        )
        llm_response = get_llm_judgment(advisor_prompt)
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
