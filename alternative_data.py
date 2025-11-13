"""Alternative data ingestion for the Spot AI agent.

This module aggregates on-chain activity and social sentiment so the
trading agent can factor non-price data into its decisions.  The design
favours graceful degradation: when APIs or the Groq LLM are unavailable
the functions fall back to neutral defaults so the wider system keeps
running.

Usage
-----
Call :func:`get_alternative_data` with a trading symbol (e.g.
``"BTCUSDT"``) to receive an :class:`AlternativeDataBundle`.  The bundle
contains on-chain metrics summarised into a composite score plus social
sentiment generated through the Groq LLM.  A ``score_adjustment`` helper
translates the alternative signals into a small numeric tweak that can be
added to the technical score.

Environment
-----------
``GLASSNODE_API_KEY``
    Optional API key for the Glassnode REST endpoints used to retrieve
    exchange flows and whale balances.
``TWITTER_BEARER_TOKEN``
    Optional bearer token for the Twitter v2 recent search endpoint.
``ENABLE_REDDIT_SCRAPE``
    When set to ``"1"`` the module will query Reddit's public search
    API.  Disabled by default to avoid unauthenticated scraping during
    automated tests.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

try:  # Optional Groq-powered alt data fusion
    from groq_alt_data import analyze_alt_data as groq_fetch_alt_data
except Exception:  # pragma: no cover - best effort fallback
    groq_fetch_alt_data = None

GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
ENABLE_REDDIT_SCRAPE = os.getenv("ENABLE_REDDIT_SCRAPE", "0") == "1"
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "spot-ai-agent/1.0")

_GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"
_TWITTER_ENDPOINT = "https://api.twitter.com/2/tweets/search/recent"
_REDDIT_SEARCH_ENDPOINT = "https://www.reddit.com/search.json"


@dataclass
class OnChainMetrics:
    """Container for on-chain flow statistics."""

    exchange_inflow: Optional[float] = None
    exchange_outflow: Optional[float] = None
    whale_inflow: Optional[float] = None
    whale_outflow: Optional[float] = None
    whale_ratio: Optional[float] = None
    net_exchange_flow: Optional[float] = None
    large_holder_netflow: Optional[float] = None
    composite_score: float = 0.0
    sources: Tuple[str, ...] = field(default_factory=tuple)

    def availability(self) -> float:
        """Return the fraction of populated metrics (0-1)."""

        values = [
            self.exchange_inflow,
            self.exchange_outflow,
            self.whale_inflow,
            self.whale_outflow,
            self.whale_ratio,
            self.net_exchange_flow,
            self.large_holder_netflow,
        ]
        available = sum(1 for value in values if value is not None)
        return available / len(values)


@dataclass
class SocialSentiment:
    """Aggregated social media sentiment."""

    bias: str = "neutral"
    score: float = 0.0
    confidence: float = 0.0
    posts_analyzed: int = 0
    source_models: Tuple[str, ...] = field(default_factory=tuple)
    raw_output: Optional[Mapping[str, object]] = None


@dataclass
class AlternativeDataBundle:
    """Combined alternative data snapshot."""

    onchain: OnChainMetrics
    social: SocialSentiment
    fetched_at: float
    sources: Tuple[str, ...] = field(default_factory=tuple)

    def score_adjustment(self, direction: Optional[str] = None) -> float:
        """Translate alternative signals into a numeric adjustment."""

        polarity = 1.0
        if direction and str(direction).lower() == "short":
            polarity = -1.0
        social_component = self.social.score * max(
            0.0, min(self.social.confidence, 1.0)
        )
        onchain_component = self.onchain.composite_score
        adjustment = 0.6 * social_component + 0.4 * onchain_component
        return float(max(-0.75, min(0.75, adjustment * polarity)))

    def to_features(self, direction: Optional[str] = None) -> Dict[str, object]:
        """Return a JSON-serialisable feature dictionary."""

        features: Dict[str, object] = {
            "fetched_at": self.fetched_at,
            "sources": list(self.sources),
            "onchain_score": self.onchain.composite_score,
            "onchain_net_flow": self.onchain.net_exchange_flow,
            "onchain_whale_ratio": self.onchain.whale_ratio,
            "onchain_large_holder_netflow": self.onchain.large_holder_netflow,
            "social_bias": self.social.bias,
            "social_score": self.social.score,
            "social_confidence": self.social.confidence,
            "social_posts": self.social.posts_analyzed,
            "social_models": list(self.social.source_models),
        }
        if direction is not None:
            features["score_adjustment"] = self.score_adjustment(direction)
        return features


_alt_cache: Dict[str, Tuple[AlternativeDataBundle, float]] = {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_models(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        models = [str(item) for item in value if isinstance(item, (str, bytes))]
        return tuple(models)
    if isinstance(value, str):
        return (value,)
    return tuple()


def _onchain_snapshot(metrics: Optional["OnChainMetrics"]) -> Mapping[str, Any]:
    if metrics is None:
        return {}
    return {
        "exchange_inflow": metrics.exchange_inflow,
        "exchange_outflow": metrics.exchange_outflow,
        "whale_inflow": metrics.whale_inflow,
        "whale_outflow": metrics.whale_outflow,
        "net_exchange_flow": metrics.net_exchange_flow,
        "large_holder_netflow": metrics.large_holder_netflow,
        "whale_ratio": metrics.whale_ratio,
        "composite_score": metrics.composite_score,
    }


def _groq_social(
    payload: Mapping[str, Any], *, posts_hint: Optional[int] = None
) -> SocialSentiment:
    bias = str(
        payload.get("bias")
        or payload.get("sentiment")
        or payload.get("label")
        or "neutral"
    )
    score = _safe_float(payload.get("score"))
    if score is None:
        score = _safe_float(payload.get("score_normalized"))
    if score is None and payload.get("score_percent") is not None:
        score = _safe_float(payload.get("score_percent"))
        if score is not None:
            score = score / 100.0
    confidence = _safe_float(payload.get("confidence"))
    if confidence is None and payload.get("confidence_percent") is not None:
        confidence = _safe_float(payload.get("confidence_percent"))
        if confidence is not None:
            confidence = confidence / 100.0
    posts = _safe_int(
        payload.get("posts")
        or payload.get("count")
        or payload.get("samples")
        or payload.get("sample_size")
    )
    if posts in (None, 0) and posts_hint is not None:
        posts = posts_hint
    models = _normalise_models(
        payload.get("models") or payload.get("sources") or payload.get("model")
    )
    if models:
        models = tuple(dict.fromkeys(models + ("groq",)))
    else:
        models = ("groq",)
    return SocialSentiment(
        bias=bias,
        score=float(max(-1.0, min(1.0, score or 0.0))),
        confidence=float(max(0.0, min(1.0, confidence or 0.0))),
        posts_analyzed=posts or 0,
        source_models=models,
        raw_output=dict(payload),
    )


def _groq_onchain(
    payload: Mapping[str, Any],
    *,
    fallback: Optional[OnChainMetrics] = None,
) -> OnChainMetrics:
    def clamp(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def with_fallback(primary: Optional[float], attr: str) -> Optional[float]:
        if primary is not None:
            return primary
        if fallback is None:
            return None
        return getattr(fallback, attr)

    inflow = _safe_float(payload.get("exchange_inflow"))
    outflow = _safe_float(payload.get("exchange_outflow"))
    whale_in = _safe_float(payload.get("whale_inflow"))
    whale_out = _safe_float(payload.get("whale_outflow"))
    whale_ratio = _safe_float(
        payload.get("whale_ratio") or payload.get("whale_exchange_ratio")
    )
    net_flow = _safe_float(
        payload.get("net_exchange_flow") or payload.get("exchange_netflow")
    )
    large_holder = _safe_float(
        payload.get("large_holder_netflow")
        or payload.get("whale_netflow")
        or payload.get("large_holder_ratio")
    )
    composite = _safe_float(
        payload.get("composite_score")
        or payload.get("score")
        or payload.get("composite")
    )
    sources = _normalise_models(payload.get("sources") or payload.get("models"))
    if sources:
        sources = tuple(dict.fromkeys(sources + ("groq",)))
    else:
        sources = ("groq",)
    metrics = OnChainMetrics(
        exchange_inflow=with_fallback(inflow, "exchange_inflow"),
        exchange_outflow=with_fallback(outflow, "exchange_outflow"),
        whale_inflow=with_fallback(whale_in, "whale_inflow"),
        whale_outflow=with_fallback(whale_out, "whale_outflow"),
        whale_ratio=clamp(whale_ratio)
        if whale_ratio is not None
        else with_fallback(None, "whale_ratio"),
        net_exchange_flow=clamp(net_flow)
        if net_flow is not None
        else with_fallback(None, "net_exchange_flow"),
        large_holder_netflow=clamp(large_holder)
        if large_holder is not None
        else with_fallback(None, "large_holder_netflow"),
        composite_score=float(
            max(
                -1.0,
                min(
                    1.0,
                    (
                        composite
                        if composite is not None
                        else (fallback.composite_score if fallback else 0.0)
                    ),
                ),
            )
        ),
        sources=sources,
    )
    return metrics


def _symbol_to_asset(symbol: str) -> str:
    sym = str(symbol).upper()
    # Strip common quote assets
    for suffix in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"):
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return sym[: -len(suffix)]
    return sym


def _glassnode_latest(endpoint: str, asset: str) -> Optional[float]:
    if not GLASSNODE_API_KEY:
        return None
    url = f"{_GLASSNODE_BASE}/{endpoint}"
    params = {"a": asset, "api_key": GLASSNODE_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            logger.debug(
                "Glassnode endpoint %s returned %s", endpoint, response.status_code
            )
            return None
        payload = response.json()
        if isinstance(payload, list) and payload:
            latest = payload[-1]
            value = latest.get("v") if isinstance(latest, Mapping) else None
            if value is None and isinstance(latest, Mapping):
                value = latest.get("value")
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    except Exception as exc:  # pragma: no cover - network best effort
        logger.debug("Glassnode request failed: %s", exc)
    return None


def fetch_onchain_metrics(symbol: str) -> OnChainMetrics:
    """Fetch on-chain metrics for ``symbol`` using Glassnode when available."""

    asset = _symbol_to_asset(symbol)
    sources: List[str] = []
    inflow = _glassnode_latest("transactions/transfers_volume_to_exchanges", asset)
    outflow = _glassnode_latest("transactions/transfers_volume_from_exchanges", asset)
    whale_in = _glassnode_latest(
        "transactions/transfers_volume_whales_to_exchanges", asset
    )
    whale_out = _glassnode_latest(
        "transactions/transfers_volume_whales_from_exchanges", asset
    )
    whale_balance = _glassnode_latest("supply/supply_balance_whales", asset)
    exchange_balance = _glassnode_latest("supply/supply_on_exchanges", asset)
    if GLASSNODE_API_KEY:
        sources.append("glassnode")
    net_exchange_flow = None
    if inflow is not None and outflow is not None and inflow + outflow > 0:
        net_exchange_flow = (outflow - inflow) / (inflow + outflow)
    large_holder_net = None
    if whale_in is not None and whale_out is not None and whale_in + whale_out > 0:
        large_holder_net = (whale_out - whale_in) / (whale_in + whale_out)
    whale_ratio = None
    if whale_balance is not None and exchange_balance not in (None, 0):
        try:
            whale_ratio = float(whale_balance) / float(exchange_balance)
        except Exception:
            whale_ratio = None
    components: List[float] = []
    for value in (net_exchange_flow, large_holder_net, whale_ratio):
        if value is None:
            continue
        components.append(max(-1.0, min(1.0, float(value))))
    composite = sum(components) / len(components) if components else 0.0
    return OnChainMetrics(
        exchange_inflow=inflow,
        exchange_outflow=outflow,
        whale_inflow=whale_in,
        whale_outflow=whale_out,
        whale_ratio=whale_ratio,
        net_exchange_flow=net_exchange_flow,
        large_holder_netflow=large_holder_net,
        composite_score=float(composite),
        sources=tuple(sources) if sources else tuple(),
    )


def _fetch_twitter_posts(query: str, limit: int) -> List[str]:
    if not TWITTER_BEARER_TOKEN or limit <= 0:
        return []
    params = {
        "query": f"({query}) (crypto OR bitcoin OR btc) -is:retweet lang:en",
        "max_results": max(10, min(limit, 100)),
        "tweet.fields": "lang,text",
    }
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    try:
        response = requests.get(_TWITTER_ENDPOINT, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.debug(
                "Twitter API returned %s for query %s", response.status_code, query
            )
            return []
        data = response.json()
        tweets = data.get("data", []) if isinstance(data, Mapping) else []
        texts: List[str] = []
        for tweet in tweets[:limit]:
            text = tweet.get("text") if isinstance(tweet, Mapping) else None
            if isinstance(text, str):
                texts.append(text.strip())
        return texts
    except Exception as exc:  # pragma: no cover - network best effort
        logger.debug("Twitter fetch failed: %s", exc)
        return []


def _fetch_reddit_posts(query: str, limit: int) -> List[str]:
    if not ENABLE_REDDIT_SCRAPE or limit <= 0:
        return []
    params = {"q": query, "limit": max(10, min(limit, 100)), "sort": "new", "t": "day"}
    headers = {"User-Agent": REDDIT_USER_AGENT}
    try:
        response = requests.get(
            _REDDIT_SEARCH_ENDPOINT, params=params, headers=headers, timeout=10
        )
        if response.status_code != 200:
            logger.debug(
                "Reddit search returned %s for query %s", response.status_code, query
            )
            return []
        payload = response.json()
        posts: List[str] = []
        children = (
            payload.get("data", {}).get("children", [])
            if isinstance(payload, Mapping)
            else []
        )
        for child in children:
            data = child.get("data") if isinstance(child, Mapping) else None
            if not isinstance(data, Mapping):
                continue
            title = data.get("title")
            selftext = data.get("selftext")
            snippets = [part for part in (title, selftext) if isinstance(part, str)]
            if snippets:
                posts.append(" ".join(snippets).strip())
            if len(posts) >= limit:
                break
        return posts
    except Exception as exc:  # pragma: no cover - network best effort
        logger.debug("Reddit fetch failed: %s", exc)
        return []


def fetch_social_posts(symbol: str, limit: int = 60) -> List[str]:
    """Collect recent social media posts mentioning ``symbol``."""

    asset = _symbol_to_asset(symbol)
    posts: List[str] = []
    twitter_quota = min(limit, 30)
    posts.extend(_fetch_twitter_posts(asset, twitter_quota))
    remaining = limit - len(posts)
    if remaining > 0:
        posts.extend(_fetch_reddit_posts(asset, remaining))
    return [post for post in posts if isinstance(post, str) and post.strip()]


_POSITIVE_HINTS = {
    "bullish",
    "buying",
    "accumulating",
    "moon",
    "pump",
    "surge",
    "rally",
    "strong",
    "uptrend",
}

_NEGATIVE_HINTS = {
    "bearish",
    "selling",
    "dump",
    "crash",
    "collapse",
    "selloff",
    "weak",
    "downtrend",
    "fear",
}


def analyze_social_sentiment(posts: Sequence[str]) -> SocialSentiment:
    """Analyse social media posts with a lightweight lexical heuristic."""

    cleaned = [post.strip() for post in posts if isinstance(post, str) and post.strip()]
    if not cleaned:
        return SocialSentiment(source_models=("heuristic",))

    positive_hits = 0
    negative_hits = 0
    for text in cleaned:
        lowered = text.lower()
        if any(token in lowered for token in _POSITIVE_HINTS):
            positive_hits += 1
        if any(token in lowered for token in _NEGATIVE_HINTS):
            negative_hits += 1

    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        score = 0.0
    else:
        score = (positive_hits - negative_hits) / float(total_hits)
    score = max(-1.0, min(1.0, score))

    confidence = min(1.0, len(cleaned) / 20.0)

    if total_hits == 0:
        bias = "neutral"
    elif score > 0.05:
        bias = "bullish"
    elif score < -0.05:
        bias = "bearish"
    else:
        bias = "neutral"

    return SocialSentiment(
        bias=bias,
        score=score,
        confidence=confidence,
        posts_analyzed=len(cleaned),
        source_models=("heuristic",),
    )


def get_alternative_data(
    symbol: str,
    *,
    ttl: float = 300.0,
    force_refresh: bool = False,
) -> AlternativeDataBundle:
    """Return cached alternative data for ``symbol`` with TTL."""

    key = _symbol_to_asset(symbol)
    now = time.time()
    cached = _alt_cache.get(key)
    if not force_refresh and cached is not None:
        bundle, timestamp = cached
        if now - timestamp <= max(0.0, ttl):
            return bundle
    fallback_onchain = fetch_onchain_metrics(symbol)
    social_posts = fetch_social_posts(symbol)

    groq_social_section: Optional[Mapping[str, Any]] = None
    groq_onchain_section: Optional[Mapping[str, Any]] = None
    groq_sources: List[str] = []

    if groq_fetch_alt_data is not None:
        try:
            groq_payload = groq_fetch_alt_data(
                symbol=symbol,
                onchain_snapshot=_onchain_snapshot(fallback_onchain),
                social_posts=social_posts,
            )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.debug("Groq alt-data analysis failed: %s", exc)
            groq_payload = None
        if isinstance(groq_payload, Mapping):
            social_candidate = groq_payload.get("social")
            if isinstance(social_candidate, Mapping):
                groq_social_section = social_candidate
            onchain_candidate = groq_payload.get("onchain")
            if isinstance(onchain_candidate, Mapping):
                groq_onchain_section = onchain_candidate
            raw_sources = groq_payload.get("sources")
            if isinstance(raw_sources, (list, tuple, set)):
                groq_sources = [
                    str(src)
                    for src in raw_sources
                    if isinstance(src, (str, bytes)) and str(src).strip()
                ]
            elif isinstance(raw_sources, (str, bytes)):
                groq_sources = [str(raw_sources)]

    if groq_onchain_section is not None:
        onchain = _groq_onchain(groq_onchain_section, fallback=fallback_onchain)
    else:
        onchain = fallback_onchain

    if groq_social_section is not None:
        social = _groq_social(
            groq_social_section,
            posts_hint=len(social_posts or []),
        )
    else:
        social = analyze_social_sentiment(social_posts)

    sources: List[str] = []
    if onchain.sources:
        sources.extend(str(src) for src in onchain.sources)
    if social.source_models:
        sources.extend(str(model) for model in social.source_models)
    if groq_sources:
        sources.extend(str(src) for src in groq_sources)
    if TWITTER_BEARER_TOKEN and social.posts_analyzed:
        sources.append("twitter")
    if ENABLE_REDDIT_SCRAPE and social.posts_analyzed:
        sources.append("reddit")
    if groq_social_section or groq_onchain_section:
        sources.append("groq")
    if not sources:
        sources.append("fallback")

    bundle = AlternativeDataBundle(
        onchain=onchain,
        social=social,
        fetched_at=now,
        sources=tuple(dict.fromkeys(sources)),
    )
    _alt_cache[key] = (bundle, now)
    return bundle


__all__ = [
    "AlternativeDataBundle",
    "OnChainMetrics",
    "SocialSentiment",
    "analyze_social_sentiment",
    "fetch_onchain_metrics",
    "fetch_social_posts",
    "get_alternative_data",
]
