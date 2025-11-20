"""
Decision logic for Spot AI Super Agent with structured LLM integration.

This module implements the core confidence aggregation and filtering pipeline
for deciding whether to open a trade.  It supports parsing JSON responses
from a large language model (LLM) advisor and gracefully falls back to
reasonable defaults when the advisor is unavailable or returns an error.

When the LLM is unreachable we now require quantitative signals to clear
stricter thresholds instead of blindly auto-approving trades.  This keeps the
system resilient during outages while still prioritising capital preservation
when qualitative oversight is missing.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import re
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from log_utils import setup_logger
import logging
import math
import time

from json_utils import parse_llm_json_response

from llm_approval import LLMTradeDecision, get_llm_trade_decision

from trade_utils import summarise_technical_score


LLM_ERROR_CONFIDENCE_FLOOR = 6.5
LLM_ERROR_SCORE_BUFFER = 0.5
# Require exceptionally strong quantitative conviction to auto-approve when the
# LLM advisor is unavailable.  This helps prevent borderline trades from being
# executed without qualitative oversight.
LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE = 8.0
QUANT_CONF_MIN_FOR_LLM_APPROVAL = 4.0

FALLBACK_LLM_SEQUENCE: List[Tuple[str, str]] = [
    (
        "groq",
        os.getenv("GROQ_MODEL_TRADE")
        or os.getenv("TRADE_LLM_MODEL")
        or "qwen/qwen3-32b",
    ),
    (
        "groq",
        os.getenv("GROQ_MODEL_FALLBACK")
        or os.getenv("GROQ_OVERFLOW_MODEL")
        or "llama-3.3-70b-versatile",
    ),
]

# Timeout for each LLM provider attempt when iterating through
# ``FALLBACK_LLM_SEQUENCE``.
LLM_FALLBACK_TIMEOUT_SECONDS = 5.0

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
        fetch_news as fetch_symbol_news_async,
        fetch_news_sync as fetch_symbol_news_sync,
    )  # type: ignore
except Exception:
    def run_news_fetcher() -> dict:  # type: ignore
        return {"ok": False, "items": [], "source": "neutral", "error": "unavailable"}

    def analyze_news_with_llm(events: list) -> Dict[str, Any]:  # type: ignore
        return {"safe": True, "reason": ""}

    async def run_news_fetcher_async(path: str = "news_events.json") -> dict:  # type: ignore
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, run_news_fetcher)

    async def analyze_news_with_llm_async(events: list) -> Dict[str, Any]:  # type: ignore
        return analyze_news_with_llm(events)

    async def fetch_symbol_news_async(symbol: str) -> list:  # type: ignore
        _ = symbol
        return []

    def fetch_symbol_news_sync(symbol: str) -> list:  # type: ignore
        _ = symbol
        return []


def _invoke_llm_provider(
    provider: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Invoke a single LLM provider/model combination."""

    if provider == "groq":
        return get_llm_judgment(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def call_llm_with_fallbacks(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 500,
    timeout: float = LLM_FALLBACK_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Attempt LLM approval across a sequence of providers/models.

    Each provider is given ``timeout`` seconds to respond.  If the call fails or
    times out we proceed to the next entry in ``FALLBACK_LLM_SEQUENCE``.
    """

    errors: List[str] = []

    for provider, model in FALLBACK_LLM_SEQUENCE:
        if not provider or not model:
            continue

        executor = ThreadPoolExecutor(max_workers=1)
        shutdown_kwargs = {"wait": True, "cancel_futures": True}

        try:
            future = executor.submit(
                _invoke_llm_provider,
                provider,
                model,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            shutdown_kwargs = {"wait": False, "cancel_futures": True}
            err = f"{provider}:{model} timed out after {timeout:.1f}s"
            errors.append(err)
            logger.warning("LLM provider %s:%s timed out after %.1fs", provider, model, timeout)
            continue
        except Exception as exc:  # noqa: BLE001 - want the full error
            err = f"{provider}:{model} -> {exc}"
            errors.append(err)
            logger.warning("LLM provider %s:%s failed: %s", provider, model, exc)
            continue
        finally:
            executor.shutdown(**shutdown_kwargs)

        trimmed = str(result).strip() if result is not None else ""
        if not trimmed or "llm error" in trimmed.lower():
            err = f"{provider}:{model} returned error payload"
            errors.append(err)
            logger.warning(
                "LLM provider %s:%s returned error payload, trying fallback", provider, model
            )
            continue

        if errors:
            logger.warning("Primary LLM failed, used fallback %s:%s", provider, model)

        return {
            "ok": True,
            "provider": provider,
            "model": model,
            "result": result,
            "errors": errors,
        }

    if errors:
        logger.error(
            "All LLM providers failed for trade approval: %s",
            "; ".join(errors),
        )
    else:
        logger.error("All LLM providers failed for trade approval: no providers configured")

    return {"ok": False, "errors": errors}

# Cache for BTC context awareness
symbol_context_cache: Dict[str, Dict[str, Any]] = {}


@dataclass
class PreparedTradeDecision:
    symbol: str
    timeframe: str
    side: str
    direction: str
    indicators: Dict[str, float]
    sentiment_bias: str
    fear_greed: float
    macro_news: Dict[str, Any]
    news_summary: str
    symbol_news_summary: str
    orderflow: str
    action_state: Optional[str]
    pattern_name: str
    pattern_memory_context: Dict[str, Any]
    technical_score: float
    final_confidence: float
    score_threshold: float
    # fields with defaults MUST come last
    macro_context: Dict[str, Any] = field(default_factory=dict)
    advisor_prompt: str = ""


def _quantitative_fallback_decision(
    prepared: PreparedTradeDecision,
    *,
    llm_errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Apply quantitative-only approval rules when all LLMs fail."""

    final_confidence = prepared.final_confidence
    score_requirement = prepared.score_threshold + LLM_ERROR_SCORE_BUFFER
    score_ok = prepared.score >= score_requirement
    high_confidence = final_confidence >= LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE
    strong_quant = high_confidence and score_ok
    score_buffer = prepared.score - score_requirement

    if llm_errors:
        logger.warning("LLM providers unavailable: %s", "; ".join(llm_errors))

    base_payload = {
        "confidence": final_confidence,
        "news_summary": prepared.news_summary,
        "symbol_news_summary": prepared.symbol_news_summary,
        "llm_decision": "LLM unavailable",
        "llm_approval": None,
        "llm_confidence": None,
        "llm_error": True,
        "llm_provider": None,
        "llm_model": None,
        "technical_indicator_score": prepared.technical_score,
        "pattern_memory": prepared.pattern_memory_context,
        "macro_context": prepared.macro_context,
        "fallback_auto_approval_threshold": LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE,
        "fallback_score_requirement": score_requirement,
    }

    if strong_quant:
        logger.warning(
            (
                "LLM unavailable across all fallbacks; proceeding with quant-only "
                "auto-approval (confidence=%.2f ≥ %.2f, score=%.2f ≥ %.2f)"
            ),
            final_confidence,
            LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE,
            prepared.score,
            score_requirement,
        )
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
            news_summary=prepared.news_summary,
        ) or f"No major pattern, but macro/sentiment context favors {prepared.direction} setup."

        return {
            **base_payload,
            "decision": True,
            "reason": (
                "LLM unavailable across all fallbacks; proceeding with quant-only "
                f"auto-approval (confidence {final_confidence:.2f} ≥ "
                f"{LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE:.2f}, score buffer {score_buffer:.2f} "
                f"≥ {LLM_ERROR_SCORE_BUFFER:.2f})"
            ),
            "narrative": narrative,
        }

    logger.warning(
        (
            "Quantitative conviction insufficient under LLM-unavailable fallback "
            "(confidence=%.2f < %.2f or score=%.2f < %.2f)"
        ),
        final_confidence,
        LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE,
        prepared.score,
        score_requirement,
    )

    return {
        **base_payload,
        "decision": False,
        "reason": (
            "Quantitative conviction insufficient under LLM-unavailable fallback "
            f"(confidence {final_confidence:.2f} < {LLM_FALLBACK_AUTO_APPROVAL_CONFIDENCE:.2f} "
            f"or score buffer {score_buffer:.2f} < {LLM_ERROR_SCORE_BUFFER:.2f})"
        ),
        "narrative": "",
    }


_SYMBOL_NEWS_CACHE_TTL_SECONDS = 15 * 60
_SIGNIFICANT_NEWS_KEYWORDS: Dict[str, float] = {
    "etf": 2.5,
    "approval": 2.0,
    "regulat": 1.8,
    "sec": 1.7,
    "lawsuit": 1.6,
    "ban": 2.0,
    "halving": 2.2,
    "upgrade": 1.4,
    "fork": 1.2,
    "whale": 1.2,
    "institution": 1.1,
    "fund": 1.0,
    "inflow": 1.3,
    "outflow": 1.3,
    "liquidation": 1.3,
    "hack": 2.2,
    "exploit": 2.2,
    "adoption": 1.1,
    "listing": 1.6,
    "delist": 2.0,
    "partnership": 1.2,
    "acquisition": 1.4,
    "merger": 1.4,
    "treasury": 1.1,
    "fidelity": 1.1,
    "blackrock": 1.3,
    "grayscale": 1.3,
    "buyback": 1.4,
    "seiz": 1.5,
    "crackdown": 1.5,
    "penalty": 1.2,
    "fine": 1.1,
}
_SYMBOL_NEWS_NOISE_PATTERNS = {
    "price analysis",
    "price prediction",
    "technical analysis",
    "how to",
    "opinion:",
    "newsletter",
    "recap",
    "week in review",
    "daily hodl",
}
_SYMBOL_NEWS_CREDIBLE_SOURCES = {
    "Bloomberg",
    "Reuters",
    "The Wall Street Journal",
    "The Wall St. Journal",
    "WSJ",
    "CoinDesk",
    "Cointelegraph",
    "The Block",
    "CNBC",
    "Forbes",
    "Fortune",
    "Bloomberg Crypto",
    "Barron's",
    "Yahoo Finance",
}
_symbol_news_cache: Dict[str, Tuple[float, str]] = {}


def _normalise_article_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_article_datetime(article: Mapping[str, Any]) -> Optional[datetime]:
    published_raw = article.get("publishedAt") or article.get("published_at")
    if not isinstance(published_raw, str):
        return None
    try:
        cleaned = published_raw.strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _score_symbol_news_article(article: Mapping[str, Any]) -> float:
    title = _normalise_article_text(article.get("title"))
    description = _normalise_article_text(article.get("description"))
    if not title:
        return 0.0

    combined = f"{title} {description}".lower()
    if any(noise in combined for noise in _SYMBOL_NEWS_NOISE_PATTERNS):
        return 0.0

    score = 0.0
    for keyword, weight in _SIGNIFICANT_NEWS_KEYWORDS.items():
        if keyword in combined:
            score += weight

    if score <= 0.0:
        return 0.0

    published_dt = _extract_article_datetime(article)
    if published_dt is not None:
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600.0
        if age_hours > 72:
            return 0.0
        if age_hours > 36:
            score *= 0.5
        elif age_hours > 18:
            score *= 0.75

    source_obj = article.get("source")
    source_name = ""
    if isinstance(source_obj, Mapping):
        source_name = _normalise_article_text(source_obj.get("name"))
    elif hasattr(source_obj, "get"):
        try:
            source_name = _normalise_article_text(source_obj.get("name"))  # type: ignore[arg-type]
        except Exception:
            source_name = ""
    if source_name in _SYMBOL_NEWS_CREDIBLE_SOURCES:
        score += 0.4

    if score < 1.5:
        return 0.0

    return score


def _select_symbol_news_headlines(
    articles: Sequence[Mapping[str, Any]], limit: int = 2
) -> Sequence[Mapping[str, Any]]:
    if limit <= 0:
        return []

    scored: list[Tuple[float, Mapping[str, Any]]] = []
    for article in articles:
        try:
            score = _score_symbol_news_article(article)
        except Exception:
            continue
        if score <= 0.0:
            continue
        scored.append((score, article))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [article for _, article in scored[:limit]]


def _format_symbol_news_line(article: Mapping[str, Any]) -> str:
    title = _normalise_article_text(article.get("title")) or "Unnamed headline"
    description = _normalise_article_text(article.get("description"))
    if description:
        sentence_end = description.find(".")
        if sentence_end != -1:
            description = description[: sentence_end + 1]
        if len(description) > 160:
            description = description[:157].rstrip() + "..."

    source_name = ""
    source = article.get("source")
    if isinstance(source, Mapping):
        source_name = _normalise_article_text(source.get("name"))

    published_dt = _extract_article_datetime(article)
    timestamp = ""
    if published_dt is not None:
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        timestamp = published_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    detail_parts = [part for part in (source_name, timestamp) if part]
    detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""

    if description:
        return f"- {title}{detail_suffix}: {description}"
    return f"- {title}{detail_suffix}"


def summarize_symbol_news(symbol: str, max_items: int = 2) -> str:
    symbol_key = symbol.upper()
    now_ts = time.time()
    cached = _symbol_news_cache.get(symbol_key)
    if cached and now_ts - cached[0] < _SYMBOL_NEWS_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        articles = fetch_symbol_news_sync(symbol)
    except Exception:
        logger.debug("Symbol news fetch failed", exc_info=True)
        _symbol_news_cache[symbol_key] = (now_ts, "")
        return ""

    selected = _select_symbol_news_headlines(articles, limit=max_items)
    if not selected:
        summary = "No material symbol-specific headlines detected in the last 72 hours."
    else:
        summary = "\n".join(_format_symbol_news_line(article) for article in selected)

    _symbol_news_cache[symbol_key] = (now_ts, summary)
    return summary


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
            payload = await run_news_fetcher_async()
            events = list(payload.get("items", []))
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
            payload = run_news_fetcher()
            events = list(payload.get("items", []))
        analysis = analyze_news_with_llm(events)
        return str(analysis.get("reason", ""))
    except Exception:
        logger.debug("Synchronous news summary failed", exc_info=True)
        return ""


def _load_cached_events(path: str = "news_events.json") -> list:
    """Return cached news events if the on-disk file is available.

    The previous implementation re-raised any exception which in turn forced
    :func:`summarize_recent_news_async` to fall back to ``run_news_fetcher_async``.
    In environments without outbound network access (such as unit tests or
    sandboxed deployments) that fallback would hang indefinitely while the
    HTTP client waited for a connection.  Instead we now treat missing or
    malformed cache files as an empty dataset so the caller can continue with a
    neutral summary immediately.
    """

    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug("No cached news events found at %s", path)
        return []
    except json.JSONDecodeError:
        logger.warning("Cached news events file %s is corrupt; ignoring", path, exc_info=True)
        return []
    except Exception:
        # Re-raise unexpected I/O errors so callers can fall back to fetching fresh data.
        logger.debug("Unexpected error loading cached news events from %s", path, exc_info=True)
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
    macro_context: Optional[Mapping[str, Any]] = None,
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
    symbol_news_summary = summarize_symbol_news(symbol)
    if not symbol_news_summary:
        symbol_news_summary = "No material symbol-specific headlines detected in the last 72 hours."

    def _format_macro_age(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not math.isfinite(number) or number < 0:
            return "N/A"
        return f"{int(number)}s"

    def _format_macro_value(value: Any, suffix: str = "") -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not math.isfinite(number):
            return "N/A"
        return f"{number:.2f}{suffix}"

    macro_snapshot = dict(macro_context) if isinstance(macro_context, Mapping) else {}
    if macro_snapshot.get("macro_bias") is None:
        macro_snapshot["macro_bias"] = sentiment_bias
    if macro_snapshot.get("fear_greed") is None:
        macro_snapshot["fear_greed"] = fear_greed
    macro_bias_label = macro_snapshot.get("macro_bias") or sentiment_bias
    macro_fg_value = macro_snapshot.get("fear_greed", fear_greed)
    macro_fg_text = f"{macro_fg_value}" if macro_fg_value is not None else "N/A"
    macro_fg_age_text = _format_macro_age(macro_snapshot.get("fear_greed_age_sec"))
    macro_btc_dom_text = _format_macro_value(macro_snapshot.get("btc_dom"), "%")
    macro_btc_age_text = _format_macro_age(macro_snapshot.get("btc_dom_age_sec"))
    macro_flag_text = ", ".join(macro_snapshot.get("macro_flags", [])) or "None"

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
        f"Macro Bias: {macro_bias_label}\n"
        f"Fear & Greed Index: {macro_fg_text}\n"
        f"Fear & Greed Age: {macro_fg_age_text}\n"
        f"BTC Dominance: {macro_btc_dom_text}\n"
        f"BTC Dominance Age: {macro_btc_age_text}\n"
        f"Macro Flags: {macro_flag_text}\n"
        f"Macro News Safe: {macro_news.get('safe', True)}\n"
        f"Macro News Notes: {macro_news.get('reason', 'N/A')}\n"
        f"Recent News Summary: {news_summary_value}\n\n"
        "### Symbol-Specific News\n"
        f"{symbol_news_summary}\n\n"
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
        macro_context=dict(macro_snapshot),
        news_summary=news_summary_value,
        symbol_news_summary=symbol_news_summary,
        orderflow=orderflow,
        auction_state=auction_state,
        pattern_name=pattern_name,
        pattern_memory_context=pattern_memory_context,
        technical_score=technical_score,
        final_confidence=final_confidence,
        score_threshold=score_threshold,
        advisor_prompt=advisor_prompt,
    )

    return None, prepared


def finalize_trade_decision(
    prepared: PreparedTradeDecision,
    llm_decision: LLMTradeDecision,
) -> Dict[str, Any]:
    """Combine LLM response with prepared context to produce final decision."""

    final_confidence = prepared.final_confidence
    technical_score = prepared.technical_score
    pattern_memory_context = prepared.pattern_memory_context
    news_summary = prepared.news_summary
    symbol_news_summary = prepared.symbol_news_summary

    base_payload = {
        "news_summary": news_summary,
        "symbol_news_summary": symbol_news_summary,
        "llm_decision": llm_decision.decision,
        "llm_approval": llm_decision.approved,
        "llm_confidence": llm_decision.confidence,
        "llm_error": llm_decision.approved is None and llm_decision.decision != "approved",
        "llm_provider": "groq",
        "llm_model": llm_decision.model,
        "technical_indicator_score": technical_score,
        "pattern_memory": pattern_memory_context,
        "macro_context": prepared.macro_context,
    }

    if llm_decision.approved is False:
        reason = llm_decision.rationale or "LLM veto"
        logger.warning(
            "LLM vetoed trade: model=%s decision=%s conf=%s reason=%s symbol=%s",
            llm_decision.model,
            llm_decision.decision,
            llm_decision.confidence,
            reason,
            prepared.symbol,
        )
        return {
            **base_payload,
            "decision": False,
            "confidence": final_confidence,
            "reason": f"LLM veto: {reason}",
            "narrative": llm_decision.rationale,
        }

    if llm_decision.approved is None:
        logger.warning(
            "LLM approval unavailable; proceeding without gate for %s",
            prepared.symbol,
        )
        return {
            **base_payload,
            "decision": True,
            "confidence": final_confidence,
            "reason": "LLM unavailable; proceeding with quant signals",
            "narrative": generate_trade_narrative(
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
                macro_sentiment=prepared.macro_context.get("macro_bias"),
                btc_dominance=prepared.macro_context.get("btc_dom"),
                fear_greed=prepared.macro_context.get("fear_greed"),
            ),
        }

    if final_confidence < QUANT_CONF_MIN_FOR_LLM_APPROVAL:
        logger.warning(
            "LLM approval but quantitative conviction %.2f < %.2f",
            final_confidence,
            QUANT_CONF_MIN_FOR_LLM_APPROVAL,
        )
        return {
            **base_payload,
            "decision": False,
            "confidence": final_confidence,
            "reason": (
                "LLM approval but quantitative conviction "
                f"{final_confidence:.2f} < {QUANT_CONF_MIN_FOR_LLM_APPROVAL:.2f}"
            ),
            "narrative": llm_decision.rationale,
        }

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
        macro_sentiment=prepared.macro_context.get("macro_bias"),
        btc_dominance=prepared.macro_context.get("btc_dom"),
        fear_greed=prepared.macro_context.get("fear_greed"),
    ) or f"No major pattern, but macro/sentiment context favors {prepared.direction} setup."

    return {
        **base_payload,
        "decision": True,
        "confidence": final_confidence,
        "reason": llm_decision.rationale or "All filters passed",
        "narrative": narrative,
    }


def _parse_llm_response(resp: str) -> Tuple[bool | None, float | None, str, str]:
    """Attempt to parse a JSON response from the LLM.

    Returns a tuple ``(decision_bool, advisor_rating, reason, thesis)``.  If
    parsing fails, returns ``(None, None, raw_response, "")``.
    """

    data, success = parse_llm_json_response(resp, logger=logger)
    if not success or not data:
        return None, None, resp, ""

    keys_lower = {str(key).lower() for key in data.keys()}
    if "error" in keys_lower and "decision" not in keys_lower:
        return None, None, resp, ""

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
    return decision_bool, rating_val, str(reason), str(thesis)



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

        trade_context = {
            "symbol": symbol,
            "direction": initial_direction,
            "score": score,
            "confidence": final_confidence,
            "technical_score": technical_score,
            "session": session,
            "pattern": pattern_name,
            "orderflow": orderflow,
            "sentiment_bias": sentiment.get("bias"),
            "sentiment_confidence": sentiment.get("confidence"),
            "volatility": volatility,
            "fear_greed": fear_greed,
            "macro_news_safe": macro_news.get("safe"),
            "macro_news_reason": macro_news.get("reason"),
            "news_summary": news_summary,
        }
        llm_decision = get_llm_trade_decision(trade_context)
        return finalize_trade_decision(prepared, llm_decision)

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
