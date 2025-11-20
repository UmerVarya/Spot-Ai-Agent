"""Background LLM-assisted classification for ambiguous news items."""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List

from llm_tasks import LLMTask, call_llm_for_task
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
NEWS_LLM_CONFIRM_FOR_HALT = _env_bool("NEWS_LLM_CONFIRM_FOR_HALT", True)
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


SYSTEM_PROMPT = """
You are the risk officer for a CRYPTO spot trading system.

Your job: given news headlines, decide how SYSTEMIC and IMMEDIATELY DANGEROUS they are for crypto markets.

The system trades only crypto spot. It does NOT care about most FX, gold, or stock noise unless it clearly signals a major USD or risk-asset shock.

You will be given a list of items. For each item, you must output:
- a systemic risk score from 0 to 3
- a directional bias for crypto
- a suggested HIGH-LEVEL category

Return ONLY valid JSON, no explanations outside JSON.

====================
RISK SCALE (0–3)
====================

0 = No meaningful systemic risk
    - Routine news, opinions, previews
    - Licenses, new services, banks adding crypto offerings
    - Policy discussions, committee hearings, personnel changes
    - Minor price commentary (“gold edges higher ahead of NFP”)

1 = Low systemic risk
    - Mild regulatory comments
    - Small incidents with limited impact
    - Non-US macro data that might matter a bit but is not a shock
    - Early discussions of potential rules without concrete action

2 = Medium systemic risk
    - Major regulatory actions or lawsuits that clearly target key crypto entities
      (e.g. SEC sues a big exchange, new law that directly restricts stablecoins)
    - Concrete, impactful US macro data surprises (CPI/NFP/FOMC decisions) that could move BTC/ETH meaningfully
    - Large negative news about key infrastructure providers, but not outright collapse

3 = High systemic risk
    - Stablecoin depegs (USDT, USDC or other major stablecoins losing their peg)
    - Large exchanges hacked, insolvent, or halting withdrawals
    - Governments or regulators effectively banning or severely restricting crypto
    - ETF approvals/denials or regulatory decisions that are known to move BTC/ETH strongly
    - Other events that could plausibly cause 10–20% moves or severe liquidity stress

IMPORTANT:
- Headlines about LICENCES, NEW PRODUCTS, BANKS LAUNCHING CRYPTO SERVICES, OR EXPANSION are usually risk 0 or 1, NOT 2 or 3.
- Headlines about COMMITTEES, HEARINGS, or NOMINATIONS are usually risk 0 or 1 unless they describe a concrete, harsh measure already decided.
- Headlines that say “ahead of NFP”, “before NFP”, “preview”, “what to expect” are PREVIEWS (usually risk 0 or 1), NOT actual data releases.
- Headlines about gold, FX, or stocks are usually low or no risk for crypto unless they explicitly describe a big macro shock.

When you are unsure, choose the LOWER risk score.

====================
CATEGORIES
====================

Use these high-level categories:

- "CRYPTO_SYSTEMIC"
    For truly systemic crypto events (risk 2–3) like depegs, major exchange crises,
    hard bans, or ETF decisions that clearly hit the whole crypto market.

- "CRYPTO_MEDIUM"
    For crypto-related news that is important but not catastrophic:
    regulation discussions, hearings, licences, new products,
    exchange listings/delistings, liquidation cascades, etc.

- "MACRO_USD_T1"
    For ACTUAL US macro DATA/DECISIONS with big potential impact:
    CPI/PCE/NFP results, FOMC decisions, major Fed statements.
    Do NOT use this for previews or “ahead of NFP” style headlines.

- "MACRO_USD_T2"
    For softer or preview macro items related to the US:
    “ahead of NFP”, “eyes on jobs data”, ISM/PMI, retail sales, sentiment surveys, etc.

- "IRRELEVANT"
    For news that has basically no direct importance for crypto:
    individual stock earnings, local politics, unrelated FX pairs, etc.

====================
DIRECTION
====================

direction should be:
- "bullish"  = likely positive for crypto overall
- "bearish"  = likely negative for crypto overall
- "mixed"    = contains both good and bad elements
- "unclear"  = cannot infer a clear sign

Examples:
- A bank gaining a license to offer crypto trading: usually "bullish".
- A depeg or ban: "bearish".
- Neutral policy hearings: often "unclear".

====================
OUTPUT FORMAT
====================

You will receive JSON input of the form:

{
  "items": [
    {
      "event_id": "...",
      "headline": "...",
      "body": "...",
      "rule_category": "CRYPTO_SYSTEMIC | CRYPTO_MEDIUM | MACRO_USD_T1 | MACRO_USD_T2 | IRRELEVANT"
    },
    ...
  ]
}

You MUST respond with JSON only:

{
  "items": [
    {
      "event_id": "...",
      "systemic_risk": 0,
      "direction": "bullish" | "bearish" | "mixed" | "unclear",
      "suggested_category": "CRYPTO_SYSTEMIC" | "CRYPTO_MEDIUM" | "MACRO_USD_T1" | "MACRO_USD_T2" | "IRRELEVANT",
      "reason": "short one-sentence explanation"
    },
    ...
  ]
}

Rules:
- Keep "reason" short but specific.
- If information is insufficient, choose lower risk and "unclear" direction.
- If the input rule_category suggests something strongly (e.g. "CRYPTO_SYSTEMIC") but the text clearly looks routine (licence, expansion, preview), you SHOULD downshift it to CRYPTO_MEDIUM with systemic_risk 0 or 1.
- Never mark expansion/licence/“ahead of NFP” headlines as high systemic risk (2 or 3) unless there is an explicit, serious threat described.
"""


def build_llm_user_prompt(inputs: Iterable[NewsLLMInput]) -> str:
    payload = {
        "items": [
            {
                "event_id": item.event_id,
                "headline": item.headline,
                "body": item.body or "",
                "rule_category": item.rule_category,
            }
            for item in inputs
        ]
    }
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_llm_user_prompt(items),
        },
    ]

    try:
        response, model_used = await asyncio.to_thread(
            call_llm_for_task,
            LLMTask.NEWS,
            messages,
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

        logger.info(
            "News LLM decision: task=news model=%s systemic=%s reason=%s",
            model_used,
            systemic_risk,
            reason,
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
    "NEWS_LLM_CONFIRM_FOR_HALT",
    "NEWS_LLM_ALLOW_UPGRADE",
    "NEWS_LLM_ALLOW_DOWNGRADE",
    "NEWS_LLM_MAX_ITEMS_PER_BATCH",
    "NewsLLMInput",
    "NewsLLMDecision",
    "queue_for_llm",
    "run_llm_batch_loop",
]
