"""Alternative data ingestion for the Spot AI agent.

This module aggregates on-chain activity and social sentiment so the
trading agent can factor non-price data into its decisions.  The design
favours graceful degradation: when APIs or heavy NLP models are
unavailable the functions fall back to neutral defaults so the wider
system keeps running.

Usage
-----
Call :func:`get_alternative_data` with a trading symbol (e.g.
``"BTCUSDT"``) to receive an :class:`AlternativeDataBundle`.  The bundle
contains on-chain metrics summarised into a composite score plus social
sentiment derived from FinLlama/FinGPT via :mod:`fused_sentiment`.  A
``score_adjustment`` helper translates the alternative signals into a
small numeric tweak that can be added to the technical score.

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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

try:  # Optional dependency – FinLlama/FinGPT sentiment fusion
    from fused_sentiment import analyze_headlines
except Exception:  # pragma: no cover - best-effort fallback
    analyze_headlines = None
    logger.warning(
        "fused_sentiment unavailable; social sentiment will default to neutral"
    )

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


def analyze_social_sentiment(posts: Sequence[str]) -> SocialSentiment:
    """Analyse social media posts using FinGPT/FinLlama fusion."""

    cleaned = [post.strip() for post in posts if isinstance(post, str) and post.strip()]
    if not cleaned:
        return SocialSentiment()
    if analyze_headlines is None:
        return SocialSentiment()
    sample = cleaned[:100]
    try:
        analysis = analyze_headlines(sample)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("Sentiment fusion failed: %s", exc)
        return SocialSentiment()
    fused = analysis.get("fused", {}) if isinstance(analysis, Mapping) else {}
    score = fused.get("score", 0.0)
    bias = fused.get("bias", "neutral")
    confidence = fused.get("confidence", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(-1.0, min(1.0, score))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    models = [
        name
        for name in ("finbert", "finllama", "fingpt")
        if isinstance(analysis, Mapping) and name in analysis
    ]
    return SocialSentiment(
        bias=str(bias),
        score=score,
        confidence=confidence,
        posts_analyzed=len(sample),
        source_models=tuple(models),
        raw_output=analysis if isinstance(analysis, Mapping) else None,
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
    onchain = fetch_onchain_metrics(symbol)
    social_posts = fetch_social_posts(symbol)
    social = analyze_social_sentiment(social_posts)
    sources: List[str] = []
    if onchain.sources:
        sources.extend(onchain.sources)
    if TWITTER_BEARER_TOKEN and any(
        src for src in social.source_models if src.lower() == "fingpt"
    ):
        sources.append("twitter")
    if ENABLE_REDDIT_SCRAPE:
        sources.append("reddit")
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
