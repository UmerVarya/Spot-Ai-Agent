"""Event relevance scoring for macro and crypto news items.

The :mod:`event_relevance` module provides a deterministic classifier that
assigns weights to incoming news events based on their historical impact on
crypto markets.  The scorer performs three main tasks:

* Keyword and metadata based categorisation of each event into macro,
  regulatory, crypto-native or geopolitical themes.
* Heuristic filters that down-weight or ignore events that historically have
  negligible effect on crypto prices (for example FX technical forecasts or
  sentiment surveys).
* Optional back-testing logic that inspects realised volatility after past
  events and flags only the categories that consistently precede significant
  crypto moves.  The resulting set of categories is then used to decide whether
  a piece of news is eligible to request a trading halt.

The classifier is intentionally lightweight and dependency free aside from
``pandas`` which is already a core dependency of the project.  It can be used
both by the LLM facing guard-rails and by deterministic fallbacks such as the
``news_monitor`` to reason about the importance of upcoming events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "EventRelevanceScorer",
    "EventRelevance",
    "DEFAULT_EVENT_SCORER",
    "BASELINE_HALT_CATEGORIES",
]


_DEFAULT_IMPACT_MULTIPLIERS = {
    "high": 1.0,
    "medium": 0.65,
    "low": 0.35,
}


_EXCLUDE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bforecast\b",
        r"\bprice target\b",
        r"\bprice outlook\b",
        r"\btechnical (analysis|outlook)\b",
        r"\bsentiment\b",
        r"\bconfidence index\b",
        r"\bconsumer confidence\b",
        r"\bpositioning report\b",
        r"\b(optional )?survey\b",
    )
)


_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "macro:cpi": (
        re.compile(r"\bconsumer price index\b", re.IGNORECASE),
        re.compile(r"\bCPI\b"),
        re.compile(r"\binflation\b", re.IGNORECASE),
        re.compile(r"\bPCE\b"),
    ),
    "macro:fomc": (
        re.compile(r"\bfomc\b", re.IGNORECASE),
        re.compile(r"federal reserve", re.IGNORECASE),
        re.compile(r"interest rate decision", re.IGNORECASE),
    ),
    "macro:employment": (
        re.compile(r"non[- ]farm payrolls", re.IGNORECASE),
        re.compile(r"jobs report", re.IGNORECASE),
        re.compile(r"unemployment", re.IGNORECASE),
        re.compile(r"initial jobless", re.IGNORECASE),
    ),
    "macro:inflation_expectations": (
        re.compile(r"ppi", re.IGNORECASE),
        re.compile(r"producer price index", re.IGNORECASE),
        re.compile(r"inflation expectations", re.IGNORECASE),
    ),
    "macro:gdp": (
        re.compile(r"gdp", re.IGNORECASE),
        re.compile(r"growth data", re.IGNORECASE),
    ),
    "macro:employment_costs": (
        re.compile(r"employment cost", re.IGNORECASE),
        re.compile(r"average hourly earnings", re.IGNORECASE),
    ),
    "regulation:enforcement": (
        re.compile(r"sec", re.IGNORECASE),
        re.compile(r"cftc", re.IGNORECASE),
        re.compile(r"department of justice", re.IGNORECASE),
        re.compile(r"lawsuit", re.IGNORECASE),
        re.compile(r"regulator", re.IGNORECASE),
        re.compile(r"ban", re.IGNORECASE),
    ),
    "crypto:exchange": (
        re.compile(r"exchange hack", re.IGNORECASE),
        re.compile(r"exchange outage", re.IGNORECASE),
        re.compile(r"withdrawal suspended", re.IGNORECASE),
        re.compile(r"liquidation cascade", re.IGNORECASE),
        re.compile(r"exchange\s+(halting|halts)", re.IGNORECASE),
    ),
    "crypto:infrastructure": (
        re.compile(r"network upgrade", re.IGNORECASE),
        re.compile(r"hard fork", re.IGNORECASE),
        re.compile(r"staking upgrade", re.IGNORECASE),
        re.compile(r"fee (cut|increase)", re.IGNORECASE),
    ),
    "crypto:stablecoin": (
        re.compile(r"stablecoin\s+(?:de-?peg(?:ged|ging)?|crisis|collapse)", re.IGNORECASE),
        re.compile(r"\bde-?peg(?:ged|ging)?\b", re.IGNORECASE),
        re.compile(r"loses\s+peg", re.IGNORECASE),
        re.compile(r"peg\s+(?:break|loss|failure)", re.IGNORECASE),
    ),
    "crypto:etf": (
        re.compile(r"bitcoin etf", re.IGNORECASE),
        re.compile(r"spot etf", re.IGNORECASE),
        re.compile(r"ethereum etf", re.IGNORECASE),
    ),
    "geopolitics:conflict": (
        re.compile(r"sanction", re.IGNORECASE),
        re.compile(r"geopolitical", re.IGNORECASE),
        re.compile(r"conflict", re.IGNORECASE),
        re.compile(r"war", re.IGNORECASE),
        re.compile(r"missile", re.IGNORECASE),
    ),
}


_METADATA_CATEGORY_MAP = {
    "crypto": "crypto:exchange",
    "defi": "crypto:infrastructure",
    "staking": "crypto:infrastructure",
    "stablecoin": "crypto:stablecoin",
    "macro": "macro:fomc",
    "rates": "macro:fomc",
    "inflation": "macro:cpi",
    "employment": "macro:employment",
    "regulation": "regulation:enforcement",
    "geopolitics": "geopolitics:conflict",
}


_DEFAULT_CATEGORY_WEIGHTS = {
    "macro:fomc": 3.8,
    "macro:cpi": 3.6,
    "macro:employment": 3.4,
    "macro:inflation_expectations": 2.8,
    "macro:gdp": 2.6,
    "macro:employment_costs": 2.5,
    "regulation:enforcement": 3.5,
    "crypto:exchange": 3.6,
    "crypto:infrastructure": 3.1,
    "crypto:stablecoin": 3.6,
    "crypto:etf": 3.3,
    "geopolitics:conflict": 3.0,
}


BASELINE_HALT_CATEGORIES = frozenset(
    {
        "macro:fomc",
        "macro:cpi",
        "macro:employment",
        "regulation:enforcement",
        "geopolitics:conflict",
        "crypto:exchange",
        "crypto:infrastructure",
        "crypto:stablecoin",
    }
)


@dataclass(frozen=True)
class EventRelevance:
    """Structured relevance data returned by :class:`EventRelevanceScorer`."""

    score: float
    category: str | None
    halt_relevant: bool
    base_weight: float
    impact_multiplier: float


class EventRelevanceScorer:
    """Score events based on their crypto market relevance."""

    def __init__(
        self,
        *,
        category_weights: Mapping[str, float] | None = None,
        halt_categories: Iterable[str] | None = None,
        volatility_horizon: str = "6h",
        baseline_window: str = "3D",
        quantile: float = 0.9,
        min_observations: int = 3,
    ) -> None:
        self.category_weights = dict(_DEFAULT_CATEGORY_WEIGHTS)
        if category_weights:
            for key, value in category_weights.items():
                if value <= 0:
                    continue
                self.category_weights[key] = float(value)

        if halt_categories is None:
            self.halt_categories = set(BASELINE_HALT_CATEGORIES)
        else:
            self.halt_categories = {str(cat) for cat in halt_categories if cat}

        if isinstance(volatility_horizon, str):
            volatility_horizon = volatility_horizon.strip()
        self.volatility_horizon = pd.Timedelta(volatility_horizon)
        if self.volatility_horizon <= pd.Timedelta(0):
            raise ValueError("volatility_horizon must be positive")

        if isinstance(baseline_window, str):
            baseline_window = baseline_window.strip()
        self.baseline_window = pd.Timedelta(baseline_window)
        if self.baseline_window <= pd.Timedelta(0):
            raise ValueError("baseline_window must be positive")

        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = float(quantile)
        self.min_observations = max(1, int(min_observations))
        self.category_stats: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score_event(self, event: Mapping[str, Any]) -> EventRelevance:
        """Return the relevance score for ``event``."""

        text_fragments = self._extract_text(event)
        if not text_fragments:
            return EventRelevance(0.0, None, False, 0.0, 0.0)

        if self._should_exclude(text_fragments):
            return EventRelevance(0.0, None, False, 0.0, 0.0)

        category = self._classify_category(text_fragments, event)
        if category is None:
            return EventRelevance(0.0, None, False, 0.0, 0.0)

        base_weight = float(self.category_weights.get(category, 1.0))
        impact_multiplier = self._impact_multiplier(event.get("impact"))
        score = base_weight * impact_multiplier
        halt_relevant = category in self.halt_categories

        return EventRelevance(score, category, halt_relevant, base_weight, impact_multiplier)

    def score_events(
        self, events: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Score ``events`` and return aggregate metrics."""

        enriched_events: list[Mapping[str, Any]] = []
        aggregate_score = 0.0
        halt_score = 0.0
        relevant_events = 0
        halt_relevant_events = 0

        for event in events:
            relevance = self.score_event(event)
            if relevance.score <= 0:
                continue

            relevant_events += 1
            aggregate_score += relevance.score
            if relevance.halt_relevant:
                halt_relevant_events += 1
                halt_score += relevance.score

            enriched = dict(event)
            enriched["relevance"] = {
                "score": round(relevance.score, 3),
                "category": relevance.category,
                "base_weight": round(relevance.base_weight, 3),
                "impact_multiplier": round(relevance.impact_multiplier, 3),
                "halt_relevant": relevance.halt_relevant,
            }
            enriched_events.append(enriched)

        return {
            "events": enriched_events,
            "aggregate_score": aggregate_score,
            "halt_score": halt_score,
            "relevant_events": relevant_events,
            "halt_relevant_events": halt_relevant_events,
        }

    def fit(
        self,
        price_history: pd.DataFrame | pd.Series,
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        """Back-test event categories against historical volatility.

        ``price_history`` must contain a ``close`` column (if it is a
        :class:`pandas.DataFrame`) or represent closing prices directly when it
        is a :class:`pandas.Series`.  The index must be timezone aware and
        sorted chronologically.
        """

        if isinstance(price_history, pd.DataFrame):
            if "close" not in price_history.columns:
                raise ValueError("price_history DataFrame must contain a 'close' column")
            series = price_history["close"].astype(float)
        else:
            series = price_history.astype(float)

        if series.empty:
            return

        if series.index.tzinfo is None:
            series = series.tz_localize("UTC")
        else:
            series = series.tz_convert("UTC")

        returns = series.pct_change().abs().dropna()
        if returns.empty:
            return

        baseline = returns.rolling(self.baseline_window).sum().dropna()
        if baseline.empty:
            baseline = returns

        threshold = float(baseline.quantile(self.quantile))
        if not math.isfinite(threshold) or threshold <= 0:
            positive = returns[returns > 0]
            if positive.empty:
                return
            threshold = float(positive.median())

        category_hits: dict[str, int] = {}
        category_counts: dict[str, int] = {}

        for event in events:
            relevance = self.score_event(event)
            if relevance.score <= 0 or relevance.category is None:
                continue

            event_time = self._event_datetime(event)
            if event_time is None:
                continue

            window_end = event_time + self.volatility_horizon
            mask = (returns.index >= event_time) & (returns.index <= window_end)
            if not mask.any():
                continue

            realised = float(returns.loc[mask].sum())
            if not math.isfinite(realised):
                continue

            category_counts[relevance.category] = category_counts.get(relevance.category, 0) + 1
            if realised >= threshold:
                category_hits[relevance.category] = category_hits.get(relevance.category, 0) + 1

        updated_halt_categories = set()
        stats: dict[str, dict[str, float]] = {}
        for category, count in category_counts.items():
            hits = category_hits.get(category, 0)
            hit_ratio = hits / count if count else 0.0
            stats[category] = {
                "count": float(count),
                "hits": float(hits),
                "hit_ratio": round(hit_ratio, 4),
            }
            if count >= self.min_observations and hit_ratio >= 0.5:
                updated_halt_categories.add(category)

        if updated_halt_categories:
            self.halt_categories = updated_halt_categories

        self.category_stats = stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _impact_multiplier(value: Any) -> float:
        if isinstance(value, str):
            key = value.strip().lower()
            return _DEFAULT_IMPACT_MULTIPLIERS.get(key, 0.5)
        return 0.5

    @staticmethod
    def _extract_text(event: Mapping[str, Any]) -> list[str]:
        fragments: list[str] = []

        def _append(value: Any) -> None:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    fragments.append(text)
            elif isinstance(value, Mapping):
                for nested in value.values():
                    _append(nested)
            elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
                for item in value:
                    _append(item)

        _append(event.get("event"))
        _append(event.get("title"))
        _append(event.get("description"))
        _append(event.get("summary"))
        _append(event.get("body"))
        _append(event.get("metadata"))

        return fragments

    @staticmethod
    def _should_exclude(text_fragments: Sequence[str]) -> bool:
        for fragment in text_fragments:
            for pattern in _EXCLUDE_PATTERNS:
                if pattern.search(fragment):
                    return True
        return False

    @staticmethod
    def _event_datetime(event: Mapping[str, Any]) -> datetime | None:
        raw = event.get("datetime")
        if not isinstance(raw, str):
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _classify_category(
        self, text_fragments: Sequence[str], event: Mapping[str, Any]
    ) -> str | None:
        best_category: str | None = None
        best_weight = -math.inf

        metadata = event.get("metadata")
        metadata_categories: set[str] = set()
        if isinstance(metadata, Mapping):
            raw_categories = metadata.get("categories")
            if isinstance(raw_categories, str):
                metadata_categories.add(raw_categories.lower())
            elif isinstance(raw_categories, Iterable):
                for item in raw_categories:
                    if isinstance(item, str):
                        metadata_categories.add(item.lower())

        for meta_tag in metadata_categories:
            mapped = _METADATA_CATEGORY_MAP.get(meta_tag)
            if mapped:
                weight = self.category_weights.get(mapped, 0.0)
                if weight > best_weight:
                    best_category = mapped
                    best_weight = weight

        for category, patterns in _CATEGORY_PATTERNS.items():
            weight = self.category_weights.get(category, 0.0)
            if weight <= best_weight:
                continue
            for fragment in text_fragments:
                if any(pattern.search(fragment) for pattern in patterns):
                    best_category = category
                    best_weight = weight
                    break

        return best_category


# Module level default scorer used by guard-rails.
DEFAULT_EVENT_SCORER = EventRelevanceScorer()

