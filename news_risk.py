"""Deterministic macro/news gating helpers for the trading engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Optional

from news_llm import (
    NEWS_LLM_ALLOW_DOWNGRADE,
    NEWS_LLM_ALLOW_UPGRADE,
    NEWS_LLM_ENABLED,
    NewsLLMDecision,
    NewsLLMInput,
    queue_for_llm,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NewsHaltState",
    "classify_news",
    "get_halt_duration_for_category",
    "make_event_id",
    "should_apply_halt_for_event",
    "process_news_item",
    "get_news_gate_state",
    "get_news_status",
    "format_news_status_line",
    "write_news_status",
    "load_news_status",
    "reset_news_halt_state",
    "apply_llm_decision",
]

CRISIS_KEYWORDS = [
    "etf approval",
    "etf approved",
    "etf denied",
    "etf rejected",
    "sec sues",
    "sec charges",
    "sec files lawsuit",
    "enforcement action",
    "tether depeg",
    "usdt depeg",
    "usdc depeg",
    "stablecoin depeg",
    "halts withdrawals",
    "suspends withdrawals",
    "withdrawals halted",
    "exchange hacked",
    "exchange hack",
    "security breach",
    "insolvency",
    "insolvent",
    "bankrupt",
    "bankruptcy",
    "freeze accounts",
    "freezes accounts",
    "asset freeze",
    "ban crypto",
    "crypto ban",
    "bans bitcoin",
    "bitcoin ban",
]

# Crisis phrases that are generic enough to require an explicit crypto context.
GENERIC_CRISIS_KEYWORDS = {
    "enforcement action",
    "security breach",
    "insolvency",
    "insolvent",
    "bankrupt",
    "bankruptcy",
    "freeze accounts",
    "freezes accounts",
    "asset freeze",
}

CRYPTO_CONTEXT_TOKENS = {
    "crypto",
    "cryptocurrency",
    "digital asset",
    "digital assets",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "stablecoin",
    "stablecoins",
    "tether",
    "usdt",
    "usdc",
    "binance",
    "coinbase",
    "kraken",
    "exchange",
    "defi",
    "token",
    "tokens",
    "blockchain",
}

REG_POLICY_KEYWORDS = [
    "fdic",
    "federal deposit insurance",
    "senate banking committee",
    "senate banking panel",
    "senate banking",
    "banking panel",
    "banking committee",
    "nominee",
    "nomination",
    "pick",
    "confirmed by the senate",
    "confirmation vote",
    "confirmation hearing",
    "hearing on crypto",
    "hearing on cryptocurrency",
    "oversight of crypto",
    "crypto oversight",
    "regulatory approach",
    "approach to crypto",
    "crypto policy",
]

EXPANSION_KEYWORDS = [
    "secures license",
    "secures key license",
    "obtains license",
    "granted license",
    "approved license",
    "launch institutional",
    "launches",
    "launching",
    "introducing",
    "introduces",
    "expands",
    "expanding",
    "rolls out",
    "crypto trading services",
    "institutional crypto",
    "hong kong license",
    "hk license",
    "virtual asset license",
    "crypto platform",
    "crypto division",
    "crypto trading desk",
    "bank secures",
    "crypto unit",
]


@dataclass
class NewsHaltState:
    """Runtime container tracking the active news halt."""

    halt_until: float = 0.0
    reason: str = ""
    category: str = ""
    last_event_headline: str = ""
    last_event_ts: float = 0.0


halt_state = NewsHaltState()
seen_events: Dict[str, float] = {}
_LLM_RULE_CATEGORY: Dict[str, str] = {}
_LLM_SUPPRESS_REHALT: set[str] = set()
_LLM_EVENT_CONTEXT: Dict[str, Dict[str, str]] = {}

NEWS_STATUS_FILE = os.getenv(
    "NEWS_STATUS_FILE",
    "/home/ubuntu/spot_data/status/news_status.json",
)


def _base_status_template() -> dict:
    return {
        "mode": "NONE",
        "category": "NONE",
        "ttl_secs": 0,
        "reason": "",
        "last_event_headline": halt_state.last_event_headline,
        "last_event_ts": halt_state.last_event_ts,
    }


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _is_reg_policy_only(text: str) -> bool:
    text = (text or "").lower()
    has_reg = any(keyword in text for keyword in REG_POLICY_KEYWORDS)
    has_crisis = any(keyword in text for keyword in CRISIS_KEYWORDS)
    return has_reg and not has_crisis


def _has_crypto_context(text: str) -> bool:
    text = (text or "").lower()
    return any(token in text for token in CRYPTO_CONTEXT_TOKENS)


def _is_crypto_systemic(text: str) -> bool:
    text = (text or "").lower()
    if _is_reg_policy_only(text):
        return False
    for keyword in CRISIS_KEYWORDS:
        if keyword not in text:
            continue
        if keyword in GENERIC_CRISIS_KEYWORDS and not _has_crypto_context(text):
            continue
        return True
    return False


def _is_expansion_news(text: str) -> bool:
    return any(k in text for k in EXPANSION_KEYWORDS)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def classify_news(headline: str, body: str) -> str:
    """Classify ``headline``/``body`` into a deterministic news bucket."""

    text = f"{headline or ''} {body or ''}"
    text = text.lower()
    compact = " ".join(text.split())
    if not compact:
        return "IRRELEVANT"

    if _is_crypto_systemic(compact):
        return "CRYPTO_SYSTEMIC"

    if _is_reg_policy_only(compact):
        return "CRYPTO_MEDIUM"

    if _is_expansion_news(compact):
        return "CRYPTO_MEDIUM"

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
    if event_id in _LLM_SUPPRESS_REHALT:
        return False
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
    event_id = make_event_id(headline, source)
    _LLM_RULE_CATEGORY[event_id] = category
    _LLM_EVENT_CONTEXT[event_id] = {
        "headline": headline,
        "body": body,
        "source": source,
        "ts": timestamp,
    }

    if NEWS_LLM_ENABLED and _is_ambiguous_for_llm(category, f"{headline} {body}"):
        try:
            queue_for_llm(
                NewsLLMInput(
                    event_id=event_id,
                    headline=headline,
                    body=body,
                    source=source,
                    rule_category=category,
                    ts=timestamp,
                )
            )
        except Exception:
            logger.debug("Failed to enqueue news item for LLM", exc_info=True)

    halt_minutes = get_halt_duration_for_category(category)

    def _result(applied: bool, minutes: float) -> dict:
        status = get_news_status(now=timestamp)["mode"]
        return {
            "category": category,
            "halt_applied": applied,
            "halt_minutes": minutes,
            "status": status,
        }

    if halt_minutes <= 0:
        logger.info("[NEWS] %s: %s", category, headline)
        return _result(False, 0)

    if not should_apply_halt_for_event(event_id, timestamp):
        logger.info("[NEWS] Duplicate HARD HALT event suppressed: %s", headline)
        return _result(False, halt_minutes)

    halt_until_candidate = timestamp + halt_minutes * 60
    if halt_until_candidate > halt_state.halt_until:
        halt_state.halt_until = halt_until_candidate
        halt_state.reason = f"{category}: {headline}".strip()
        halt_state.category = category
        halt_state.last_event_headline = headline
        halt_state.last_event_ts = timestamp
        logger.warning(
            "[NEWS] HARD HALT %s for %sm (until %s)",
            category,
            halt_minutes,
            _format_ts(halt_state.halt_until),
        )
        try:
            write_news_status(now=timestamp)
        except Exception:
            logger.debug("Failed to write news status", exc_info=True)
        applied = True
    else:
        logger.info(
            "[NEWS] Existing halt already in place until %s (category %s)",
            _format_ts(halt_state.halt_until),
            halt_state.category or category,
        )
        applied = False

    return _result(applied, halt_minutes)


def _is_ambiguous_for_llm(rule_category: str, text: str) -> bool:
    compact = (text or "").lower()
    if rule_category == "CRYPTO_MEDIUM":
        return True
    return False


def apply_llm_decision(decision: NewsLLMDecision) -> None:
    """Adjust classification or halt behavior based on LLM output."""

    try:
        rule_category = _LLM_RULE_CATEGORY.get(decision.event_id, decision.suggested_category)
        context = _LLM_EVENT_CONTEXT.get(decision.event_id, {})

        if (
            NEWS_LLM_ALLOW_UPGRADE
            and rule_category == "CRYPTO_MEDIUM"
            and decision.suggested_category == "CRYPTO_SYSTEMIC"
            and decision.systemic_risk >= 2
        ):
            headline = context.get("headline", "")
            timestamp = float(context.get("ts", time.time()))
            halt_minutes = get_halt_duration_for_category("CRYPTO_SYSTEMIC")
            if halt_minutes > 0 and should_apply_halt_for_event(decision.event_id, timestamp):
                halt_until_candidate = timestamp + halt_minutes * 60
                if halt_until_candidate > halt_state.halt_until:
                    halt_state.halt_until = halt_until_candidate
                    halt_state.reason = f"CRYPTO_SYSTEMIC: {headline}".strip()
                    halt_state.category = "CRYPTO_SYSTEMIC"
                    halt_state.last_event_headline = headline
                    halt_state.last_event_ts = timestamp
                    logger.warning(
                        "[NEWS] LLM UPGRADE HARD HALT CRYPTO_SYSTEMIC for %sm (until %s)",
                        halt_minutes,
                        _format_ts(halt_state.halt_until),
                    )
                    try:
                        write_news_status(now=timestamp)
                    except Exception:
                        logger.debug("Failed to write news status", exc_info=True)

        if (
            NEWS_LLM_ALLOW_DOWNGRADE
            and rule_category == "CRYPTO_SYSTEMIC"
            and decision.systemic_risk <= 1
            and decision.suggested_category == "CRYPTO_MEDIUM"
        ):
            _LLM_SUPPRESS_REHALT.add(decision.event_id)
    except Exception:
        logger.debug("Error while applying LLM decision", exc_info=True)


def get_news_gate_state(*, now: Optional[float] = None) -> dict:
    """Return the current gate status for the trading pipeline."""

    status = get_news_status(now)
    if status["mode"] == "NONE":
        return {"mode": "NONE", "reason": "", "ttl_secs": 0}

    return {
        "mode": "HARD_HALT",
        "reason": status.get("reason") or "News hard halt in effect",
        "ttl_secs": status.get("ttl_secs", 0),
    }


def get_news_status(now: float | None = None) -> dict:
    """Return a normalized view of the current news risk state."""

    timestamp = float(now if now is not None else time.time())
    if timestamp >= halt_state.halt_until:
        if halt_state.halt_until:
            halt_state.halt_until = 0.0
            halt_state.reason = ""
            halt_state.category = ""
        base = _base_status_template()
        return base

    ttl = int(max(0.0, halt_state.halt_until - timestamp))
    return {
        "mode": "HARD_HALT",
        "category": halt_state.category or "NONE",
        "ttl_secs": ttl,
        "reason": halt_state.reason or "News hard halt in effect",
        "last_event_headline": halt_state.last_event_headline,
        "last_event_ts": halt_state.last_event_ts,
    }


def format_news_status_line(
    now: float | None = None,
    *,
    status: dict | None = None,
) -> str:
    """Return the standardized NEWS status log line."""

    if status is None:
        status = get_news_status(now)

    if status.get("mode") != "HARD_HALT":
        return "NEWS: OK (no active hard halt)"

    category = status.get("category") or "NONE"
    ttl = int(status.get("ttl_secs", 0))
    mins = ttl // 60
    secs = ttl % 60
    headline = (
        status.get("last_event_headline")
        or status.get("reason")
        or "Unknown event"
    )
    if mins > 0:
        ttl_str = f"{mins}m left"
    else:
        ttl_str = f"{secs}s left"
    return f"NEWS: HARD_HALT ({category}, {ttl_str}) – {headline}"


def write_news_status(now: float | None = None) -> None:
    """Persist the current news status for external consumers."""

    status = get_news_status(now)
    status["updated_at"] = time.time()
    tmp_path = NEWS_STATUS_FILE + ".tmp"
    os.makedirs(os.path.dirname(NEWS_STATUS_FILE) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle)
    os.replace(tmp_path, NEWS_STATUS_FILE)


def load_news_status() -> dict:
    """Load the persisted news status for the dashboard."""

    path = os.getenv(
        "NEWS_STATUS_FILE",
        "/home/ubuntu/spot_data/status/news_status.json",
    )
    status_path = Path(path)
    if not status_path.exists():
        default = _base_status_template()
        default["updated_at"] = 0.0
        return default

    try:
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except Exception:
        default = _base_status_template()
        default["reason"] = "Error reading news status"
        default["updated_at"] = 0.0
        return default

    defaults = _base_status_template()
    defaults["updated_at"] = 0.0
    defaults.update(status)
    return defaults


def reset_news_halt_state() -> None:
    """Reset the halt state and clear duplicate tracking (primarily for tests)."""

    halt_state.halt_until = 0.0
    halt_state.reason = ""
    halt_state.category = ""
    halt_state.last_event_headline = ""
    halt_state.last_event_ts = 0.0
    seen_events.clear()
    _LLM_RULE_CATEGORY.clear()
    _LLM_SUPPRESS_REHALT.clear()
    _LLM_EVENT_CONTEXT.clear()
