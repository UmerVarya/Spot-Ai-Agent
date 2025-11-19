"""Deterministic macro/news gating helpers for the trading engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "NewsHaltState",
    "classify_news",
    "get_halt_duration_for_category",
    "make_event_id",
    "should_apply_halt_for_event",
    "process_news_item",
    "get_news_gate_state",
    "reset_news_halt_state",
]


@dataclass
class NewsHaltState:
    """Runtime container tracking the active news halt."""

    halt_until: float = 0.0
    reason: str = ""
    category: str = ""


halt_state = NewsHaltState()
seen_events: Dict[str, float] = {}


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def classify_news(headline: str, body: str) -> str:
    """Classify ``headline``/``body`` into a deterministic news bucket."""

    text = f"{headline or ''} {body or ''}".strip().lower()
    compact = " ".join(text.split())
    if not compact:
        return "IRRELEVANT"

    stables = {"tether", "usdt", "usdc", "busd", "dai"}
    if "etf" in compact and any(
        token in compact for token in ("approval", "approve", "denial", "reject", "decision", "application")
    ):
        return "CRYPTO_SYSTEMIC"
    if "sec" in compact and any(exchange in compact for exchange in ("binance", "coinbase")):
        return "CRYPTO_SYSTEMIC"
    if any(stable in compact for stable in stables) and any(
        term in compact for term in ("depeg", "de-peg", "loses peg", "lost peg", "off-peg")
    ):
        return "CRYPTO_SYSTEMIC"
    if "halt" in compact and "withdraw" in compact:
        return "CRYPTO_SYSTEMIC"
    if "hack" in compact and any(word in compact for word in ("exchange", "binance", "coinbase", "bridge")):
        return "CRYPTO_SYSTEMIC"
    if "ban" in compact and "crypto" in compact:
        return "CRYPTO_SYSTEMIC"

    macro_t1_terms = (
        "fomc",
        "federal reserve",
        "fed meeting",
        "fed rate",
        "rate decision",
        "cpi",
        "core cpi",
        "pce",
        "core pce",
        "nfp",
        "nonfarm payroll",
        "jobs report",
        "powell",
        "inflation",
    )
    if any(term in compact for term in macro_t1_terms):
        return "MACRO_USD_T1"

    macro_t2_terms = (
        "ism",
        "pmi",
        "retail sales",
        "consumer confidence",
        "housing starts",
        "jobless claims",
        "initial claims",
        "continuing claims",
    )
    if any(term in compact for term in macro_t2_terms):
        return "MACRO_USD_T2"

    crypto_medium_terms = (
        "liquidation",
        "funding rate",
        "open interest",
        "listing",
        "delisting",
        "crypto",
        "binance",
        "coinbase",
    )
    if any(term in compact for term in crypto_medium_terms):
        return "CRYPTO_MEDIUM"

    if any(token in compact for token in ("bitcoin", "btc", "eth", "ethereum")):
        return "CRYPTO_MEDIUM"

    return "IRRELEVANT"


def get_halt_duration_for_category(category: str) -> int:
    """Return halt duration (minutes) for ``category`` based on env configuration."""

    normalized = (category or "").strip().upper()
    if normalized == "CRYPTO_SYSTEMIC":
        return _env_int("NEWS_CRYPTO_SYS_HALT_MINS", 120)
    if normalized == "MACRO_USD_T1":
        return _env_int("NEWS_USD_T1_HALT_MINS", 30)
    return 0


def make_event_id(headline: str, source: str) -> str:
    """Return a stable hash identifier for a news event."""

    normalized = f"{_clean_text(headline)}|{_clean_text(source)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def should_apply_halt_for_event(event_id: str, now: float) -> bool:
    """Return True if ``event_id`` is eligible for a new halt at ``now``."""

    suppress_window = max(0, _env_int("NEWS_REHALT_SUPPRESS_WINDOW_MINS", 240)) * 60
    previous = seen_events.get(event_id)
    if previous is not None and now - previous < suppress_window:
        return False
    seen_events[event_id] = now
    return True


def _format_ts(timestamp: float) -> str:
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def process_news_item(
    headline: str,
    body: str,
    source: str,
    *,
    now: Optional[float] = None,
) -> dict:
    """Process a single news payload and update the halt state if required."""

    timestamp = float(now if now is not None else time.time())
    category = classify_news(headline, body)
    halt_minutes = get_halt_duration_for_category(category)

    if halt_minutes <= 0:
        logger.info("[NEWS] %s: %s", category, headline)
        return {"category": category, "halt_applied": False, "halt_minutes": 0}

    event_id = make_event_id(headline, source)
    if not should_apply_halt_for_event(event_id, timestamp):
        logger.info("[NEWS] Duplicate HARD HALT event suppressed: %s", headline)
        return {"category": category, "halt_applied": False, "halt_minutes": halt_minutes}

    halt_until_candidate = timestamp + halt_minutes * 60
    if halt_until_candidate > halt_state.halt_until:
        halt_state.halt_until = halt_until_candidate
        halt_state.reason = f"{category}: {headline}".strip()
        halt_state.category = category
        logger.warning(
            "[NEWS] HARD HALT %s for %sm (until %s)",
            category,
            halt_minutes,
            _format_ts(halt_state.halt_until),
        )
        applied = True
    else:
        logger.info(
            "[NEWS] Existing halt already in place until %s (category %s)",
            _format_ts(halt_state.halt_until),
            halt_state.category or category,
        )
        applied = False

    return {"category": category, "halt_applied": applied, "halt_minutes": halt_minutes}


def get_news_gate_state(*, now: Optional[float] = None) -> dict:
    """Return the current gate status for the trading pipeline."""

    timestamp = float(now if now is not None else time.time())
    if timestamp >= halt_state.halt_until:
        if halt_state.halt_until:
            halt_state.halt_until = 0.0
            halt_state.reason = ""
            halt_state.category = ""
        return {"mode": "NONE", "reason": "", "ttl_secs": 0}

    ttl = int(max(0.0, halt_state.halt_until - timestamp))
    return {
        "mode": "HARD_HALT",
        "reason": halt_state.reason or "News hard halt in effect",
        "ttl_secs": ttl,
    }


def reset_news_halt_state() -> None:
    """Reset the halt state and clear duplicate tracking (primarily for tests)."""

    halt_state.halt_until = 0.0
    halt_state.reason = ""
    halt_state.category = ""
    seen_events.clear()
