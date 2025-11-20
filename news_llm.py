"""Background LLM-assisted classification for ambiguous news items."""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List

import config
from groq_safe import safe_chat_completion
from log_utils import setup_logger


logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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


NEWS_LLM_ENABLED = _env_bool("NEWS_LLM_ENABLED", False)
NEWS_LLM_ALLOW_UPGRADE = _env_bool("NEWS_LLM_ALLOW_UPGRADE", False)
NEWS_LLM_ALLOW_DOWNGRADE = _env_bool("NEWS_LLM_ALLOW_DOWNGRADE", False)
NEWS_LLM_MAX_ITEMS_PER_BATCH = max(1, int(os.getenv("NEWS_LLM_MAX_ITEMS_PER_BATCH", "5")))
NEWS_LLM_SLEEP_SECONDS = max(0.1, float(os.getenv("NEWS_LLM_POLL_INTERVAL", "1.0")))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class NewsLLMInput:
    event_id: str
    headline: str
    body: str
    source: str
    rule_category: str
    ts: float


@dataclass
class NewsLLMDecision:
    event_id: str
    systemic_risk: int
    direction: str
    suggested_category: str
    reason: str


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """
You are a risk analyst for a crypto trading system.
You will be given a list of news items about crypto, finance, and macro.
For each item, classify how systemically dangerous it is for crypto markets.

Systemic risk scale:
- 0 = No meaningful systemic risk (routine news, mild policy, expansion, or unrelated).
- 1 = Low systemic risk (minor regulatory news, small incidents).
- 2 = Medium systemic risk (big regulatory actions, major lawsuits, serious policy shifts).
- 3 = High systemic risk (depegs, large exchange hacks, withdrawals halted, outright bans, ETF approval/denial shocks).

Important:
- Headlines about licences, new products, banks adding crypto services, or routine policy hearings are usually risk 0 or 1.
- Headlines about "halts withdrawals", "depeg", "hack", "insolvent", "ban crypto", "ETF denied/approved" are often risk 2 or 3.
- When unsure, choose the lower risk.
- Default to CRYPTO_MEDIUM and lower systemic risk when the story is ambiguous.
- Real systemic disasters are: stablecoin depegs; major exchange hacks or insolvencies; Binance/Coinbase/major CEX halting withdrawals; hard bans or ETF approvals/denials with large market impact.

You must respond with pure JSON only.
""".strip()


def build_llm_user_prompt(inputs: Iterable[NewsLLMInput]) -> str:
    items = []
    for item in inputs:
        items.append(
            {
                "id": item.event_id,
                "headline": item.headline,
                "body": item.body,
            }
        )
    payload = {"items": items}
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Queueing helpers
# ---------------------------------------------------------------------------


_LLM_QUEUE: Deque[NewsLLMInput] = deque()
_LLM_QUEUE_IDS: set[str] = set()
_LLM_QUEUE_LOCK = asyncio.Lock()


async def _dequeue_batch(max_items: int) -> List[NewsLLMInput]:
    async with _LLM_QUEUE_LOCK:
        items: List[NewsLLMInput] = []
        for _ in range(min(max_items, len(_LLM_QUEUE))):
            candidate = _LLM_QUEUE.popleft()
            _LLM_QUEUE_IDS.discard(candidate.event_id)
            items.append(candidate)
        return items


def queue_for_llm(item: NewsLLMInput) -> bool:
    """Enqueue ``item`` for LLM analysis, ignoring duplicates."""

    if not NEWS_LLM_ENABLED:
        return False

    if item.event_id in _LLM_QUEUE_IDS:
        return False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(_queue_async(item))
    else:
        asyncio.run(_queue_async(item))
    return True


async def _queue_async(item: NewsLLMInput) -> None:
    async with _LLM_QUEUE_LOCK:
        if item.event_id in _LLM_QUEUE_IDS:
            return
        _LLM_QUEUE.append(item)
        _LLM_QUEUE_IDS.add(item.event_id)


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------


async def _call_llm_batch(items: List[NewsLLMInput]) -> List[NewsLLMDecision]:
    if not items:
        return []

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_llm_user_prompt(items),
        },
    ]

    try:
        response = await safe_chat_completion(
            client=None,
            model=config.get_news_model(),
            messages=messages,
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
            timeout=10,
        )
    except Exception:
        logger.exception("LLM batch call failed")
        return []

    content = (getattr(response, "choices", None) or [{}])[0].get("message", {}).get("content", "")
    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON response: %s", content)
        return []

    decisions: List[NewsLLMDecision] = []
    for raw in parsed.get("items", []) if isinstance(parsed, dict) else []:
        if not isinstance(raw, dict):
            continue
        event_id = str(raw.get("event_id") or raw.get("id") or "").strip()
        if not event_id:
            continue
        systemic_risk = raw.get("systemic_risk")
        try:
            systemic_risk = int(systemic_risk)
        except (TypeError, ValueError):
            systemic_risk = 0
        direction = (raw.get("direction") or "unclear").strip() or "unclear"
        suggested_category = (raw.get("suggested_category") or "").strip() or ""
        reason = (raw.get("reason") or "").strip()
        # Default category to original rule category when missing
        original = next((i for i in items if i.event_id == event_id), None)
        if not suggested_category and original:
            suggested_category = original.rule_category
        decisions.append(
            NewsLLMDecision(
                event_id=event_id,
                systemic_risk=max(0, min(3, systemic_risk)),
                direction=direction,
                suggested_category=suggested_category or (original.rule_category if original else ""),
                reason=reason,
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------


async def run_llm_batch_loop(*, stop_event: asyncio.Event) -> None:
    """Continuously drain the queue and apply LLM decisions."""

    if not NEWS_LLM_ENABLED:
        logger.info("News LLM disabled; background loop not started")
        return

    while not stop_event.is_set():
        try:
            batch = await _dequeue_batch(NEWS_LLM_MAX_ITEMS_PER_BATCH)
            if not batch:
                await asyncio.sleep(NEWS_LLM_SLEEP_SECONDS)
                continue

            decisions = await _call_llm_batch(batch)
            if not decisions:
                continue

            try:
                from news_risk import apply_llm_decision
            except Exception:
                logger.exception("Unable to import news_risk.apply_llm_decision")
                continue

            for decision in decisions:
                try:
                    apply_llm_decision(decision)
                except Exception:
                    logger.exception("Failed to apply LLM decision for %s", decision.event_id)
        except Exception:
            logger.exception("LLM batch loop encountered an error")
            await asyncio.sleep(NEWS_LLM_SLEEP_SECONDS)


__all__ = [
    "NEWS_LLM_ENABLED",
    "NEWS_LLM_ALLOW_UPGRADE",
    "NEWS_LLM_ALLOW_DOWNGRADE",
    "NEWS_LLM_MAX_ITEMS_PER_BATCH",
    "NewsLLMInput",
    "NewsLLMDecision",
    "queue_for_llm",
    "run_llm_batch_loop",
]
