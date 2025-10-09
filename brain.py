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

import asyncio
from dataclasses import dataclass
import re
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from log_utils import setup_logger
import logging
import math

from trade_utils import summarise_technical_score

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
    from pattern_memory import (  # type: ignore
        recall_pattern_confidence,
        get_pattern_posterior_stats,
    )
except Exception:
    def recall_pattern_confidence(symbol: str, pattern_name: str) -> float:  # type: ignore
        return 0.0

    def get_pattern_posterior_stats(symbol: str, pattern_name: str) -> Dict[str, float]:  # type: ignore
        return {"mean": 0.5, "variance": 1.0 / 12.0, "alpha": 1.0, "beta": 1.0, "trades": 0.0}

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

try:
    from fetch_news import (
        run_news_fetcher,
        analyze_news_with_llm,
        run_news_fetcher_async,
        analyze_news_with_llm_async,
    )  # type: ignore
except Exception:
    def run_news_fetcher() -> list:  # type: ignore
        return []

    def analyze_news_with_llm(events: list) -> Dict[str, Any]:  # type: ignore
        return {"safe": True, "reason": ""}

    async def run_news_fetcher_async(path: str = "news_events.json") -> list:  # type: ignore
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, run_news_fetcher)

    async def analyze_news_with_llm_async(events: list) -> Dict[str, Any]:  # type: ignore
        return analyze_news_with_llm(events)

# Cache for BTC context awareness
symbol_context_cache: Dict[str, Dict[str, Any]] = {}


@dataclass(slots=True)
class PreparedTradeDecision:
    """Container holding deterministic context before querying the LLM."""

    symbol: str
    direction: str
    session: str
    setup_type: Optional[str]
    score: float
    indicators: Dict[str, float]
    sentiment_bias: str
    sentiment_confidence: float
    fear_greed: Optional[int]
    macro_news: Dict[str, Any]
    news_summary: str
    orderflow: str
    auction_state: Optional[str]
    pattern_name: str
    pattern_memory_context: Dict[str, Any]
    technical_score: float
    final_confidence: float
    advisor_prompt: str


async def summarize_recent_news_async() -> str:
    """Asynchronously fetch recent events and summarise them via the LLM."""

    try:
        events: list
        try:
            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(
                None,
                _load_cached_events,
            )
        except Exception:
            events = await run_news_fetcher_async()
        analysis = await analyze_news_with_llm_async(events)
        return str(analysis.get("reason", ""))
    except Exception:
        logger.debug("Async news summary failed", exc_info=True)
        return ""


def summarize_recent_news() -> str:
    """Synchronous helper that wraps :func:`summarize_recent_news_async`."""

    try:
        return asyncio.run(summarize_recent_news_async())
    except RuntimeError:
        # Already inside an event loop (e.g. notebook). Fallback to blocking path.
        try:
            with open("news_events.json", "r") as f:
                events = json.load(f)
        except Exception:
            events = run_news_fetcher()
        analysis = analyze_news_with_llm(events)
        return str(analysis.get("reason", ""))
    except Exception:
        logger.debug("Synchronous news summary failed", exc_info=True)
        return ""


def _load_cached_events(path: str = "news_events.json") -> list:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        raise


def prepare_trade_decision(
    *,
    symbol: str,
    score: float,
    direction: Optional[str],
    indicators: Mapping[str, float],
    session: str,
    pattern_name: str,
    orderflow: str,
    sentiment: Mapping[str, Any],
    macro_news: Mapping[str, Any],
    volatility: Optional[float],
    fear_greed: Optional[int],
    auction_state: Optional[str],
    setup_type: Optional[str],
    news_summary: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[PreparedTradeDecision]]:
    """Prepare deterministic trade context and LLM prompt.

    Returns a tuple ``(result, prepared)`` where ``result`` is a final decision
    dictionary if the trade is rejected before involving the LLM.  When the LLM
    needs to be queried, ``result`` is ``None`` and ``prepared`` contains the
    information required to finalise the decision once a response is
    available.
    """

    initial_direction = direction or "long"
    technical_score = summarise_technical_score(indicators, initial_direction)

    try:
        sentiment_bias = str(sentiment.get("bias", "neutral"))
        sentiment_confidence = float(sentiment.get("score", 5.0))
    except Exception:
        sentiment_bias = "neutral"
        sentiment_confidence = 5.0

    try:
        macro_safe = bool(macro_news.get("safe", True))
    except Exception:
        macro_safe = True
    if not macro_safe:
        reason = str(macro_news.get("reason", "unknown"))
        return (
            {
                "decision": False,
                "confidence": 0.0,
                "reason": f"Macro news unsafe: {reason}",
                "llm_decision": None,
                "llm_approval": None,
                "llm_confidence": None,
                "llm_error": False,
                "technical_indicator_score": technical_score,
            },
            None,
        )

    news_summary_value = news_summary if news_summary is not None else summarize_recent_news()

    try:
        pattern_lower = pattern_name.lower() if isinstance(pattern_name, str) else ""
    except Exception:
        pattern_lower = ""

    setup_lower = (setup_type or "").lower() if isinstance(setup_type, str) else ""
    breakout_patterns = {
        "triangle_wedge",
        "flag",
        "cup_handle",
        "double_bottom",
    }
    breakout_match = (
        setup_lower in {"trend", "breakout"}
        or pattern_lower in breakout_patterns
        or ("breakout" in pattern_lower if pattern_lower else False)
    )
    if auction_state and auction_state.lower() == "balanced" and breakout_match:
        return (
            {
                "decision": False,
                "confidence": float(score),
                "reason": "Balanced market regime – breakout setups disabled",
                "llm_decision": None,
                "llm_approval": None,
                "llm_confidence": None,
                "llm_error": False,
                "technical_indicator_score": technical_score,
            },
            None,
        )

    pattern_stats = get_pattern_posterior_stats(symbol, pattern_name)
    posterior_mean = float(pattern_stats.get("mean", 0.5))
    posterior_variance = float(pattern_stats.get("variance", 1.0 / 12.0))
    trade_observations = float(pattern_stats.get("trades", 0.0))
    pattern_memory_context = {
        "posterior_mean": round(posterior_mean, 4),
        "posterior_variance": round(posterior_variance, 6),
        "trades": int(trade_observations),
        "alpha": float(pattern_stats.get("alpha", 1.0)),
        "beta": float(pattern_stats.get("beta", 1.0)),
    }

    base_threshold = get_adaptive_conf_threshold() or 4.5
    if sentiment_bias == "bullish":
        score_threshold = base_threshold - 0.3
    elif sentiment_bias == "bearish":
        score_threshold = base_threshold + 0.3
    else:
        score_threshold = base_threshold
    score_threshold = round(score_threshold, 2)

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
        score_threshold -= 0.5

    try:
        base_thr = get_adaptive_conf_threshold() or 4.5
    except Exception:
        base_thr = 4.5
    if sentiment_bias in {"bullish", "neutral"} and score >= (base_thr - 0.5):
        score_threshold -= 0.2

    if trade_observations >= 5:
        threshold_delta = 0.0
        if posterior_mean >= 0.6:
            threshold_delta -= min(0.4, (posterior_mean - 0.6) * 1.5)
        elif posterior_mean <= 0.4:
            threshold_delta += min(0.4, (0.4 - posterior_mean) * 1.5)

        if posterior_variance > 0.03:
            threshold_delta += min(0.3, (posterior_variance - 0.03) * 10.0)
        elif posterior_variance < 0.01:
            threshold_delta -= min(0.2, (0.01 - posterior_variance) * 15.0)

        score_threshold = round(score_threshold + threshold_delta, 2)

    if volatility is not None and not math.isnan(volatility):
        if volatility < 0.2:
            score_threshold += 0.4
        elif volatility > 0.8:
            score_threshold -= 0.2

    if fear_greed is not None:
        try:
            fg_val = float(fear_greed)
        except Exception:
            fg_val = None
        if fg_val is not None:
            if fg_val < 20:
                score_threshold += 0.8
            elif fg_val > 80:
                score_threshold -= 0.2

    score_threshold = round(max(score_threshold, 4.0), 2)

    if direction is None and score >= score_threshold and sentiment_bias != "bearish":
        direction = "long"
        logger.info("Fallback direction applied: long (Sentiment: %s)", sentiment_bias)

    if direction != initial_direction:
        technical_score = summarise_technical_score(indicators, direction or initial_direction)

    confidence = float(score)

    if sentiment_bias == "bullish":
        confidence += 1.0
    elif sentiment_bias == "bearish":
        confidence -= 1.0
    try:
        confidence += (float(sentiment_confidence) - 5.0) * 0.3
    except Exception:
        pass

    if (direction or "long") == "long":
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

    memory_boost = recall_pattern_confidence(symbol, pattern_name)
    confidence += memory_boost

    if trade_observations >= 3:
        confidence += (posterior_mean - 0.5) * 2.0
        if posterior_variance < 0.02:
            confidence += min(0.8, (0.02 - posterior_variance) * 20.0)
        elif posterior_variance > 0.06:
            confidence -= min(0.8, (posterior_variance - 0.06) * 12.0)

    hist_result = calculate_historical_confidence(symbol, score, direction or "long", session, pattern_name)
    try:
        confidence += (hist_result.get("confidence", 50) - 50) / 10.0  # type: ignore[arg-type]
    except Exception:
        pass

    if isinstance(orderflow, str):
        flow = orderflow.lower()
        if "buy" in flow:
            confidence += 0.5
        elif "sell" in flow:
            confidence -= 0.5

    symbol_context_cache[symbol] = {
        "bias": sentiment_bias,
        "direction": direction or initial_direction,
        "confidence": score,
    }
    if symbol != "BTCUSDT" and (direction or "long") == "long":
        btc_ctx = symbol_context_cache.get("BTCUSDT", {})
        if (
            btc_ctx.get("bias") == "bullish"
            and btc_ctx.get("direction") == "long"
            and btc_ctx.get("confidence", 0) >= 6
        ):
            confidence += 0.8

    exp_ret = indicators.get("next_return")
    if exp_ret is not None:
        try:
            confidence += max(min(float(exp_ret) * 50.0, 2.0), -2.0)
        except Exception:
            pass

    final_confidence = round(max(0.0, min(confidence, 10.0)), 2)

    trade_direction = direction or "long"
    if trade_direction != "long":
        return (
            {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Trade direction is not long (spot-only mode)",
                "llm_decision": None,
                "llm_approval": None,
                "llm_confidence": None,
                "llm_error": False,
                "technical_indicator_score": technical_score,
                "pattern_memory": pattern_memory_context,
            },
            None,
        )
    if score < score_threshold:
        return (
            {
                "decision": False,
                "confidence": final_confidence,
                "reason": f"Score {score:.2f} below threshold {score_threshold:.2f}",
                "llm_decision": None,
                "llm_approval": None,
                "llm_confidence": None,
                "llm_error": False,
                "technical_indicator_score": technical_score,
                "pattern_memory": pattern_memory_context,
            },
            None,
        )
    if final_confidence < 4.0:
        return (
            {
                "decision": False,
                "confidence": final_confidence,
                "reason": "Low confidence",
                "llm_decision": None,
                "llm_approval": None,
                "llm_confidence": None,
                "llm_error": False,
                "technical_indicator_score": technical_score,
                "pattern_memory": pattern_memory_context,
            },
            None,
        )

    recent_summary = get_recent_trade_summary(symbol=symbol, pattern=pattern_name, max_entries=3)
    example_approval_json = json.dumps(
        {
            "decision": "Yes",
            "confidence": 7.8,
            "reason": "Order flow aligns with bullish macro backdrop.",
            "thesis": "Buyers dominate while macro data turns positive; expect continuation higher.",
        },
        indent=2,
    )
    example_rejection_json = json.dumps(
        {
            "decision": "No",
            "confidence": 3.2,
            "reason": "Momentum and order flow diverge from bullish case.",
            "thesis": "Technical momentum is fading and sellers control order flow, increasing downside risk.",
        },
        indent=2,
    )

    advisor_prompt = (
        "You are an experienced crypto-trading advisor. Review the following context and respond only with valid JSON.\n\n"
        f"### Trade Metadata\n"
        f"Symbol: {symbol}\n"
        f"Direction: {trade_direction}\n"
        f"Session: {session}\n"
        f"Setup Type: {setup_type if setup_type is not None else 'N/A'}\n"
        f"Pre-LLM Confidence: {final_confidence:.2f}/10\n"
        f"Technical Score: {score:.2f}\n\n"
        "### Technical Indicators\n"
        f"RSI: {indicators.get('rsi', 0):.1f}\n"
        f"MACD: {indicators.get('macd', 0):.4f}\n"
        f"ADX: {indicators.get('adx', 0):.1f}\n"
        f"Volatility (ATR percentile): {volatility if volatility is not None and not math.isnan(volatility) else 'N/A'}\n\n"
        "### Macro Sentiment\n"
        f"Bias: {sentiment_bias}\n"
        f"Confidence: {sentiment_confidence}\n"
        f"Fear & Greed Index: {fear_greed if fear_greed is not None else 'N/A'}\n"
        f"Macro News Safe: {macro_news.get('safe', True)}\n"
        f"Macro News Notes: {macro_news.get('reason', 'N/A')}\n"
        f"Recent News Summary: {news_summary_value}\n\n"
        "### Order Flow\n"
        f"Order Flow State: {orderflow}\n"
        f"Auction State: {auction_state if auction_state is not None else 'N/A'}\n\n"
        "### Historical Memory\n"
        f"Pattern: {pattern_name}\n"
        f"Posterior Mean Win Rate: {posterior_mean:.2%}\n"
        f"Posterior Variance: {posterior_variance:.5f}\n"
        f"Recorded Pattern Trades: {int(trade_observations)}\n"
        f"Historical Context Summary: {recent_summary if recent_summary else 'No recent trades available'}\n\n"
        "Evaluate support and conflict between sections. Provide a concise rationale highlighting decisive factors.\n"
        "Return a JSON object with exactly these keys: decision (\"Yes\" to approve the long trade, \"No\" to reject), confidence (0-10 float), reason (<=200 characters), thesis (2-3 sentences summarizing the trade idea).\n"
        "Do not include any additional commentary outside the JSON.\n\n"
        "Example Approval:\n"
        f"{example_approval_json}\n"
        "Example Rejection:\n"
        f"{example_rejection_json}\n"
    )

    prepared = PreparedTradeDecision(
        symbol=symbol,
        direction=trade_direction,
        session=session,
        setup_type=setup_type,
        score=score,
        indicators=dict(indicators),
        sentiment_bias=sentiment_bias,
        sentiment_confidence=sentiment_confidence,
        fear_greed=fear_greed,
        macro_news=dict(macro_news),
        news_summary=news_summary_value,
        orderflow=orderflow,
        auction_state=auction_state,
        pattern_name=pattern_name,
        pattern_memory_context=pattern_memory_context,
        technical_score=technical_score,
        final_confidence=final_confidence,
        advisor_prompt=advisor_prompt,
    )

    return None, prepared


def finalize_trade_decision(
    prepared: PreparedTradeDecision,
    llm_response: Any,
) -> Dict[str, Any]:
    """Combine LLM response with prepared context to produce final decision."""

    final_confidence = prepared.final_confidence
    technical_score = prepared.technical_score
    pattern_memory_context = prepared.pattern_memory_context
    news_summary = prepared.news_summary

    try:
        resp_lc = llm_response.lower() if isinstance(llm_response, str) else ""
    except Exception:
        resp_lc = ""

    if "error" in resp_lc:
        narrative = generate_trade_narrative(
            symbol=prepared.symbol,
            direction=prepared.direction,
            score=prepared.score,
            confidence=final_confidence,
            indicators=prepared.indicators,
            sentiment_bias=prepared.sentiment_bias,
            sentiment_confidence=prepared.sentiment_confidence,
            orderflow=prepared.orderflow,
            pattern=prepared.pattern_name,
            macro_reason=prepared.macro_news.get("reason", ""),
            news_summary=news_summary,
        ) or f"No major pattern, but macro/sentiment context favors {prepared.direction} setup."
        return {
            "decision": True,
            "confidence": final_confidence,
            "reason": "LLM unavailable or returned error; auto-approval",
            "narrative": narrative,
            "news_summary": news_summary,
            "llm_decision": "LLM unavailable",
            "llm_approval": True,
            "llm_confidence": None,
            "llm_error": True,
            "technical_indicator_score": technical_score,
            "pattern_memory": pattern_memory_context,
        }

    parsed_decision, advisor_rating, advisor_reason, advisor_thesis = _parse_llm_response(str(llm_response))
    if parsed_decision is None:
        match = re.search(r"(\d+(?:\.\d+)?)", str(llm_response))
        if match:
            try:
                advisor_rating = float(match.group(1))
            except Exception:
                advisor_rating = None
        parsed_decision = str(llm_response).strip().lower().startswith("yes")
        advisor_reason = str(llm_response).strip()
        advisor_thesis = ""

    if advisor_rating is not None:
        advisor_rating = max(0.0, min(float(advisor_rating), 10.0))
        final_confidence = round((final_confidence + advisor_rating) / 2.0, 2)

    if not parsed_decision:
        return {
            "decision": False,
            "confidence": final_confidence,
            "reason": f"LLM advisor vetoed trade: {advisor_reason}",
            "narrative": advisor_thesis,
            "news_summary": news_summary,
            "llm_decision": advisor_reason,
            "llm_approval": parsed_decision,
            "llm_confidence": advisor_rating,
            "llm_error": False,
            "technical_indicator_score": technical_score,
            "pattern_memory": pattern_memory_context,
        }

    narrative = advisor_thesis or generate_trade_narrative(
        symbol=prepared.symbol,
        direction=prepared.direction,
        score=prepared.score,
        confidence=final_confidence,
        indicators=prepared.indicators,
        sentiment_bias=prepared.sentiment_bias,
        sentiment_confidence=prepared.sentiment_confidence,
        orderflow=prepared.orderflow,
        pattern=prepared.pattern_name,
        macro_reason=prepared.macro_news.get("reason", ""),
        news_summary=news_summary,
    ) or f"No major pattern, but macro/sentiment context favors {prepared.direction} setup."

    return {
        "decision": True,
        "confidence": final_confidence,
        "reason": "All filters passed",
        "narrative": narrative,
        "news_summary": news_summary,
        "llm_decision": advisor_reason or "Approved",
        "llm_approval": True,
        "llm_confidence": advisor_rating,
        "llm_error": False,
        "technical_indicator_score": technical_score,
        "pattern_memory": pattern_memory_context,
    }


def _parse_llm_response(resp: str) -> Tuple[bool | None, float | None, str, str]:
    """Attempt to parse a JSON response from the LLM.

    Returns a tuple ``(decision_bool, advisor_rating, reason, thesis)``.  If
    parsing fails, returns ``(None, None, raw_response, "")``.
    """
    try:
        data = json.loads(resp)
        decision = data.get("decision", "No")
        rating = data.get("confidence", None)
        reason = data.get("reason", "")
        thesis = data.get("thesis", "")
        decision_bool = str(decision).strip().lower().startswith("y")
        rating_val: float | None = None
        if isinstance(rating, (int, float, str)):
            try:
                rating_val = float(rating)  # type: ignore[arg-type]
            except Exception:
                rating_val = None
        return decision_bool, rating_val, reason, thesis
    except Exception:
        return None, None, resp, ""



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
    volatility: float | None = None,
    fear_greed: int | None = None,
    auction_state: str | None = None,
    setup_type: str | None = None,
    news_summary: str | None = None,
) -> Dict[str, Any]:
    """Determine whether to take a trade based on quantitative metrics and LLM guidance."""

    initial_direction = direction or "long"
    technical_score = summarise_technical_score(indicators, initial_direction)

    try:
        pre_result, prepared = prepare_trade_decision(
            symbol=symbol,
            score=score,
            direction=direction,
            indicators=indicators,
            session=session,
            pattern_name=pattern_name,
            orderflow=orderflow,
            sentiment=sentiment,
            macro_news=macro_news,
            volatility=volatility,
            fear_greed=fear_greed,
            auction_state=auction_state,
            setup_type=setup_type,
            news_summary=news_summary,
        )
        if prepared is None:
            assert pre_result is not None
            return pre_result

        llm_response: Any = get_llm_judgment(prepared.advisor_prompt)
        return finalize_trade_decision(prepared, llm_response)

    except Exception as e:
        return {
            "decision": False,
            "confidence": 0.0,
            "reason": f"Error in should_trade(): {e}",
            "llm_decision": None,
            "llm_approval": None,
            "llm_confidence": None,
            "llm_error": True,
            "technical_indicator_score": technical_score,
        }
