import random
from sentiment import get_macro_sentiment
from groq_llm import get_llm_judgment
from confidence_guard import get_adaptive_conf_threshold
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence
from narrative_builder import generate_trade_narrative  # ✅ Corrected import

# Global symbol context memory (shared between decisions)
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

        # === Adaptive Score Threshold (adjusted base threshold to 5.5) ===
        base_threshold = 5.5
        if sentiment_bias == "bullish":
            score_threshold = base_threshold - 0.3  # slightly easier threshold if bullish sentiment
        elif sentiment_bias == "bearish":
            score_threshold = base_threshold + 0.3  # slightly stricter threshold if bearish sentiment
        else:
            score_threshold = base_threshold
        score_threshold = round(score_threshold, 2)

        # === Direction fallback logic ===
        if direction is None and score >= score_threshold:
            # Permit long trades even on neutral sentiment; only use short if strongly bearish bias
            direction = "long" if sentiment_bias != "bearish" else "short"
            print(f"🧠 Fallback direction applied: {direction} (Sentiment: {sentiment_bias})")

        # === Base confidence ===
        confidence = float(score)

        # === Sentiment bias boost ===
        if sentiment_bias == "bullish":
            confidence += 1
        elif sentiment_bias == "bearish":
            confidence -= 1

        confidence += (sentiment_confidence - 5) * 0.3

        # === Indicator-based adjustments ===
        if direction == "long":
            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            adx = indicators.get("adx", 20)

            if rsi > 70:
                confidence -= 1
            elif rsi < 30:
                confidence += 1

            if macd > 0:
                confidence += 0.5

            if adx > 25:
                confidence += 0.5

        # === Pattern Memory Boost ===
        memory_boost = recall_pattern_confidence(symbol, pattern_name)
        confidence += memory_boost

        # === Historical Performance Boost ===
        historical_result = calculate_historical_confidence(symbol, score, direction, session, pattern_name)
        confidence += (historical_result.get("confidence", 50) - 50) / 10

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

        # === Normalize and clamp confidence ===
        final_confidence = round(max(0.0, min(confidence, 10.0)), 2)

        # === Final score and confidence checks ===
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
        # ✅ Fallback narrative
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
