"""
Decision logic for Spot AI Super Agent with structured LLM integration.

This module implements the core confidence aggregation and filtering pipeline
for deciding whether to open a trade.  It supports parsing JSON responses
from a large language model (LLM) advisor and gracefully falls back to
reasonable defaults when the advisor is unavailable or returns an error.

When the LLM returns an error (for example due to missing API keys),
the function automatically approves the trade based on quantitative
metrics alone.  This ensures that paper trading mode continues to
function even without LLM access.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, Tuple
from log_utils import setup_logger
import logging

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Optional imports for external modules.
#
# The Spot AI brain logic integrates several helper modules such as
# ``sentiment``, ``groq_llm``, ``confidence_guard``, ``pattern_memory``,
# ``confidence``, ``narrative_builder`` and ``memory_retriever``.  In
# environments where these modules are missing, we provide simple
# fallback implementations so that this file can still be imported
# without crashing.  These fallbacks return neutral or default values.

try:
    from sentiment import get_macro_sentiment  # type: ignore
except Exception:
    def get_macro_sentiment() -> Dict[str, Any]:  # type: ignore
        """Fallback macro sentiment: neutral with medium confidence."""
        return {"bias": "neutral", "score": 5.0, "confidence": 5.0}

try:
    from groq_llm import get_llm_judgment  # type: ignore
except Exception:
    def get_llm_judgment(prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> str:  # type: ignore
        """Fallback LLM judgment: always allow trade with 7.0 confidence."""
        return json.dumps({"decision": "Yes", "confidence": 7.0, "reason": "Fallback approval"})

try:
    from confidence_guard import get_adaptive_conf_threshold  # type: ignore
except Exception:
    def get_adaptive_conf_threshold() -> float:  # type: ignore
        return 4.5

try:
    from pattern_memory import recall_pattern_confidence  # type: ignore
except Exception:
    def recall_pattern_confidence(symbol: str, pattern_name: str) -> float:  # type: ignore
        return 0.0

try:
    from confidence import calculate_historical_confidence  # type: ignore
except Exception:
    def calculate_historical_confidence(symbol: str, score: float, direction: str, session: str, pattern: str) -> Dict[str, Any]:  # type: ignore
        return {"confidence": 50.0}

try:
    from narrative_builder import generate_trade_narrative  # type: ignore
except Exception:
    def generate_trade_narrative(**kwargs: Any) -> Any:  # type: ignore
        return None

try:
    from memory_retriever import get_recent_trade_summary  # type: ignore
except Exception:
    def get_recent_trade_summary(symbol: str, pattern: str, max_entries: int = 3) -> str:  # type: ignore
        return ""

# Cache for BTC context awareness
symbol_context_cache: Dict[str, Dict[str, Any]] = {}


def _parse_llm_response(resp: str) -> Tuple[bool | None, float | None, str]:
    """Attempt to parse a JSON response from the LLM.

    Returns a tuple ``(decision_bool, advisor_rating, reason)``.  If parsing
    fails, returns ``(None, None, raw_response)``.
    """
    try:
        data = json.loads(resp)
        decision = data.get("decision", "No")
        rating = data.get("confidence", None)
        reason = data.get("reason", "")
        decision_bool = str(decision).strip().lower().startswith("y")
        rating_val: float | None = None
        if isinstance(rating, (int, float, str)):
            try:
                rating_val = float(rating)  # type: ignore[arg-type]
            except Exception:
                rating_val = None
        return decision_bool, rating_val, reason
    except Exception:
        return None, None, resp


def should_trade(
    symbol: str,
    score: float,
    direction: str | None,
    indicators: Dict[str, float],
    session: str,
    pattern_name: str,
    orderflow: str,
    sentiment: Dict[str, Any],
    macro_news: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine whether to take a trade based on quantitative metrics and LLM guidance.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., ``"BTCUSDT"``).
    score : float
        Technical score for the trade setup.
    direction : str or None
        Desired trade direction (only "long" trades are allowed).
    indicators : dict
        Dictionary with technical indicator values such as RSI, MACD and ADX.
    session : str
        Market session identifier.
    pattern_name : str
        Name of the candlestick pattern detected.
    orderflow : str
        Order flow state (e.g., "buyers", "sellers" or "neutral").
    sentiment : dict
        Macro sentiment dictionary with ``bias`` and ``score`` keys.
    macro_news : dict
        Macro news analysis result with ``safe`` and ``reason`` keys.

    Returns
    -------
    dict
        A dictionary with keys ``decision`` (bool), ``confidence`` (float),
        ``reason`` (str) and optionally ``narrative`` (str).
    """
    try:
        sentiment_bias: str = sentiment.get("bias", "neutral")  # type: ignore[arg-type]
        sentiment_confidence: float = sentiment.get("score", 5.0)  # type: ignore[arg-type]
        # Macro news safety check
        if not macro_news.get("safe", True):
            return {
                "decision": False,
                "confidence": 0.0,
                "reason": "Macro news unsafe: " + macro_news.get("reason", "unknown"),
            }

        # Determine adaptive score threshold based on sentiment
        base_threshold: float = get_adaptive_conf_threshold() or 4.5
        if sentiment_bias == "bullish":
            score_threshold = base_threshold - 0.3
        elif sentiment_bias == "bearish":
            score_threshold = base_threshold + 0.3
        else:
            score_threshold = base_threshold
        score_threshold = round(score_threshold, 2)

        # -----------------------------------------------------------------
        # Quant‑style dynamic threshold adjustments
        # Certain bullish candlestick patterns historically produce strong
        # follow‑through.  Lower the score threshold for these patterns
        # to ensure good setups are not discarded purely due to a high
        # default threshold.  This is especially important when the
        # confidence_guard module returns a strict cutoff (e.g., 5.5).
        # We cap the minimum threshold at 4.0 to avoid approving
        # marginal setups.
        try:
            pattern_lower = pattern_name.lower() if isinstance(pattern_name, str) else ""
        except Exception:
            pattern_lower = ""
        strong_patterns = {
            "three_white_soldiers",
            "marubozu_bullish",
            "bullish_engulfing",
            "piercing_line",
            "hammer",
            "inverted_hammer",
            "tweezer_bottom",
        }
        if pattern_lower in strong_patterns:
            # Reduce threshold by 0.5 for strong bullish patterns
            score_threshold -= 0.5

        # Additional adjustment: if sentiment is neutral or bullish and the
        # raw technical score already exceeds (base_threshold - 0.5), nudge the
        # threshold down slightly.  This encourages trades in moderately
        # positive environments.
        if sentiment_bias in {"bullish", "neutral"}:
            try:
                base_thr = get_adaptive_conf_threshold() or 4.5
            except Exception:
                base_thr = 4.5
            if score >= (base_thr - 0.5):
                score_threshold -= 0.2

        # Ensure threshold does not fall below 4.0
        score_threshold = round(max(score_threshold, 4.0), 2)

        # Set default direction based on score and sentiment
        if direction is None and score >= score_threshold and sentiment_bias != "bearish":
            direction = "long"
            logger.info("Fallback direction applied: long (Sentiment: %s)", sentiment_bias)

        confidence: float = float(score)

        # Sentiment adjustments
        if sentiment_bias == "bullish":
            confidence += 1.0
        elif sentiment_bias == "bearish":
            confidence -= 1.0
        try:
            confidence += (float(sentiment_confidence) - 5.0) * 0.3
        except Exception:
            pass

        # Indicator adjustments (RSI, MACD, ADX)
        if direction == "long":
            rsi = indicators.get("rsi", 50.0)
            macd = indicators.get("macd", 0.0)
            adx = indicators.get("adx", 20.0)
            if rsi > 70:
                confidence -= 1.0
            elif rsi < 30:
                confidence += 1.0
            if macd > 0:
                confidence += 0.5
            if adx > 25:
                confidence += 0.5

        # Pattern memory boost
        memory_boost: float = recall_pattern_confidence(symbol, pattern_name)
        confidence += memory_boost

        # Historical performance boost
        hist_result = calculate_historical_confidence(symbol, score, direction, session, pattern_name)
        try:
            confidence += (hist_result.get("confidence", 50) - 50) / 10.0  # type: ignore[arg-type]
        except Exception:
            pass

        # Order flow adjustments
        if isinstance(orderflow, str):
            flow = orderflow.lower()
            if "buy" in flow:
                confidence += 0.5
            elif "sell" in flow:
                confidence -= 0.5

        # Symbol context awareness (boost if BTC is strongly long and bullish)
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

        exp_ret = indicators.get('next_return')
        if exp_ret is not None:
            try:
                adj = max(min(float(exp_ret) * 50.0, 2.0), -2.0)
                confidence += adj
            except Exception:
                pass

        # Clamp confidence into [0, 10]
        final_confidence = round(max(0.0, min(confidence, 10.0)), 2)

        # Basic gating conditions
        if direction != "long":
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Trade direction is not long (spot-only mode)",
            }
        if score < score_threshold:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"Score {score:.2f} below threshold {score_threshold:.2f}",
            }
        # Require a minimum confidence but allow more flexibility for scalping.
        # Original implementation rejected trades below 4.5; we relax this to 4.0
        # to avoid discarding potentially profitable setups when indicators and
        # macro conditions are supportive.
        if final_confidence < 4.0:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Low confidence",
            }

        # Build prompt for LLM advisor
        recent_summary: str = get_recent_trade_summary(symbol=symbol, pattern=pattern_name, max_entries=3)
        advisor_prompt: str = (
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

        # Query the LLM advisor
        llm_response: Any = get_llm_judgment(advisor_prompt)

        # -------------------------------------------------------------------
        # Handle LLM failures: if the response contains the word "error", auto-approve
        # The LLM can fail if API keys are missing or other issues occur.  In such
        # cases, we skip the LLM gating and approve the trade based on quantitative
        # metrics alone.
        try:
            resp_lc = llm_response.lower() if isinstance(llm_response, str) else ""
        except Exception:
            resp_lc = ""
        if "error" in resp_lc:
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
                macro_reason=macro_news.get("reason", ""),
            ) or f"No major pattern, but macro/sentiment context favors {direction} setup."
            return {
                "decision": True,
                "confidence": final_confidence,
                "reason": "LLM unavailable or returned error; auto-approval",
                "narrative": narrative,
            }

        # Parse the LLM response (JSON or fallback)
        parsed_decision, advisor_rating, advisor_reason = _parse_llm_response(str(llm_response))
        if parsed_decision is None:
            # Fallback: use regex to extract a number and yes/no at start
            match = re.search(r"(\d+(?:\.\d+)?)", str(llm_response))
            if match:
                try:
                    advisor_rating = float(match.group(1))
                except Exception:
                    advisor_rating = None
            parsed_decision = str(llm_response).strip().lower().startswith("yes")
            advisor_reason = str(llm_response).strip()

        # Blend advisor rating into final confidence
        if advisor_rating is not None:
            advisor_rating = max(0.0, min(float(advisor_rating), 10.0))
            final_confidence = round((final_confidence + advisor_rating) / 2.0, 2)

        # If the LLM advises against the trade, veto it
        if not parsed_decision:
            return {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"LLM advisor vetoed trade: {advisor_reason}",
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
            macro_reason=macro_news.get("reason", ""),
        ) or f"No major pattern, but macro/sentiment context favors {direction} setup."

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
