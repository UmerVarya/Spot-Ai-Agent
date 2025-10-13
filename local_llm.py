"""Glue layer between the trading agent and a local Ollama deployment.

The functions in this module act as glue between the agent's domain-specific
context objects and the general purpose prompts sent to the fallback LLM. Each
entry point is defensive: if the local server is unavailable the caller
receives ``None`` or a safe default so the trading loop can continue using
remote providers.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Mapping

from log_utils import setup_logger
from risk_veto import evaluate_risk_veto

try:
    from llm_client import chat, generate, warm_up as _warm_up
except Exception:  # pragma: no cover - allow import without Ollama installed
    def chat(*args, **kwargs):  # type: ignore
        raise RuntimeError("llm_client unavailable")

    def generate(*args, **kwargs):  # type: ignore
        raise RuntimeError("llm_client unavailable")

    def _warm_up(*args, **kwargs):  # type: ignore
        return False


logger = setup_logger(__name__)

_STRUCTURED_SYSTEM_MESSAGE = (
    "You are a quantitative crypto trading assistant. Respond only with JSON containing decision (Yes/No), "
    "confidence (0-10 float), reason, and thesis (2-3 sentences).\n"
    "Think through the problem step by step in a private scratchpad. Before replying, silently run this private "
    "verification checklist (DO NOT OUTPUT IT):\n"
    "- Confirm the decision aligns with the provided context.\n"
    "- Ensure confidence is a numeric value between 0 and 10.\n"
    "- Provide a concise reason highlighting the main drivers.\n"
    "- Write a thesis of 2-3 sentences consistent with the decision.\n"
    "- Double-check the final reply is valid JSON with the required keys and no extra text."
)

_LOCAL_ENABLED = os.getenv("ENABLE_LOCAL_LLM", "1").lower() not in {"0", "false", "no"}
_DEFAULT_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))


def is_enabled() -> bool:
    """Return ``True`` when the local fallback is enabled via environment."""

    return _LOCAL_ENABLED


def warm_up_local_llm() -> bool:
    """Preload the configured Ollama model so first use is faster."""

    if not is_enabled():
        return False
    success = _warm_up()
    if success:
        logger.info("Local LLM warm-up succeeded")
    else:
        logger.debug("Local LLM warm-up skipped or failed")
    return success


def structured_trade_judgment(prompt: str, *, temperature: float = 0.2, num_ctx: int | None = None) -> str | None:
    """Run the trade-approval prompt against the local LLM returning JSON text."""

    if not is_enabled():
        return None
    messages = [
        {
            "role": "system",
            "content": _STRUCTURED_SYSTEM_MESSAGE,
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return chat(messages, temperature=temperature, num_ctx=num_ctx or _DEFAULT_CTX)
    except Exception as exc:
        logger.warning("Local structured judgment failed: %s", exc)
        return None


def build_narrative_prompt(trade_data: Mapping[str, Any]) -> str:
    """Render a concise prompt for explaining a trade idea."""

    return (
        "You are a professional crypto trading assistant. Write a concise narrative explaining the trade rationale.\n"
        f"Symbol: {trade_data.get('symbol')}\n"
        f"Direction: {trade_data.get('direction')}\n"
        f"Confidence Score: {trade_data.get('confidence')}\n"
        f"Indicator Score: {trade_data.get('score')}\n"
        f"Pattern: {trade_data.get('pattern')}\n"
        f"Macro Bias: {trade_data.get('macro_sentiment')}\n"
        f"Order Flow: {trade_data.get('orderflow')}\n"
        f"News Summary: {trade_data.get('news_summary')}\n"
        "Highlight the top two supporting factors in 3 sentences or fewer and finish with a verdict phrase."
    )


def generate_local_narrative(trade_data: Mapping[str, Any]) -> str | None:
    """Return a fallback narrative using the local model when Groq is unavailable."""

    if not is_enabled():
        return None
    prompt = build_narrative_prompt(trade_data)
    try:
        return generate(prompt, temperature=0.6, num_ctx=_DEFAULT_CTX)
    except Exception as exc:
        logger.warning("Local narrative generation failed: %s", exc)
        return None


def generate_daily_recap(prompt: str) -> str | None:
    """Return a local-LLM recap for the provided daily summary prompt."""

    if not is_enabled():
        return None
    try:
        return generate(prompt, temperature=0.4, num_ctx=_DEFAULT_CTX)
    except Exception as exc:
        logger.debug("Local daily recap generation failed: %s", exc)
        return None


def build_risk_check_prompt(payload: Mapping[str, Any]) -> str:
    """Format the guardrail prompt instructing the model to return JSON."""

    return (
        "You are a crypto risk manager. Respond ONLY with a JSON object containing the keys"
        " `enter` (boolean), `reasons` (array of short strings), `conflicts` (array of short strings),"
        " and `max_rr` (float).\n"
        f"Symbol: {payload.get('symbol')}\n"
        f"Direction: {payload.get('direction')}\n"
        f"Model Confidence: {payload.get('confidence')}\n"
        f"ML Win Probability: {payload.get('ml_probability')}\n"
        f"Volatility Percentile: {payload.get('volatility')}\n"
        f"HTF Trend %: {payload.get('htf_trend_pct')}\n"
        f"BTC Trend: {payload.get('btc_trend')}\n"
        f"Minutes To News: {payload.get('time_to_news_minutes')}\n"
        f"Order Flow Note: {payload.get('orderflow')}\n"
        f"Macro Bias: {payload.get('macro_bias')}\n"
        f"Session: {payload.get('session')}\n"
        f"Setup Type: {payload.get('setup_type')}\n"
        f"Risk Limits: max_trades={payload.get('max_trades')} open_positions={payload.get('open_positions')}\n"
        "Respect the deterministic guardrails; if any conflicts exist, set `enter` to false and explain succinctly."
    )


def _parse_risk_json(raw: str) -> Mapping[str, Any] | None:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    enter = data.get("enter")
    reasons = data.get("reasons")
    conflicts = data.get("conflicts")
    max_rr = data.get("max_rr")
    if not isinstance(enter, bool):
        return None
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        return None
    if not isinstance(conflicts, list) or not all(isinstance(item, str) for item in conflicts):
        return None
    try:
        max_rr_val = float(max_rr)
    except (TypeError, ValueError):
        return None
    cleaned = {
        "enter": enter,
        "reasons": [item.strip() for item in reasons if item],
        "conflicts": [item.strip() for item in conflicts if item],
        "max_rr": max_rr_val,
    }
    return cleaned


def _merge_veto_results(base: Mapping[str, Any], llm: Mapping[str, Any]) -> dict[str, Any]:
    merged = {
        "enter": bool(base.get("enter", True)) and bool(llm.get("enter", True)),
        "reasons": [],
        "conflicts": [],
        "max_rr": float(base.get("max_rr", llm.get("max_rr", 0.0))),
    }
    reasons = list(base.get("reasons", [])) + list(llm.get("reasons", []))
    conflicts = list(base.get("conflicts", [])) + list(llm.get("conflicts", []))
    seen: set[str] = set()
    for item in reasons:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            merged["reasons"].append(stripped)
    seen.clear()
    for item in conflicts:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            merged["conflicts"].append(stripped)
    try:
        merged["max_rr"] = float(min(base.get("max_rr", float("inf")), llm.get("max_rr", float("inf"))))
    except Exception:
        merged["max_rr"] = float(base.get("max_rr", llm.get("max_rr", 0.0)))
    if merged["conflicts"]:
        merged["enter"] = False
    return merged


def run_pretrade_risk_check(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Ask the local model to vet a trade before it is submitted."""

    base = evaluate_risk_veto(payload)
    if not is_enabled() or not base.get("enter", True):
        return base
    prompt = build_risk_check_prompt(payload)
    try:
        raw = generate(prompt, temperature=0.1, num_ctx=_DEFAULT_CTX)
    except Exception as exc:
        logger.debug("Local risk check failed: %s", exc)
        return base
    parsed = _parse_risk_json(raw)
    if parsed is None:
        logger.debug("Local risk check returned unparseable payload: %s", raw)
        return base
    return _merge_veto_results(base, parsed)


def build_signal_explainer_prompt(payload: Mapping[str, Any]) -> str:
    """Prompt template describing the chosen signal for logging."""

    return (
        "You are preparing a concise log entry explaining why a trade signal was accepted."
        " Respond in 2 sentences covering the strongest indicator cluster and risk factors.\n"
        f"Symbol: {payload.get('symbol')}\n"
        f"Pattern: {payload.get('pattern')}\n"
        f"Score: {payload.get('score')}\n"
        f"Confidence: {payload.get('confidence')}\n"
        f"Macro Bias: {payload.get('macro_bias')}\n"
        f"Order Flow: {payload.get('orderflow')}\n"
        f"Volume Profile: {payload.get('volume_profile')}\n"
        f"Additional Context: {payload.get('context')}\n"
        "End with a short next-step note (e.g. 'monitor funding' or 'tighten stop')."
    )


def generate_signal_explainer(payload: Mapping[str, Any]) -> str | None:
    if not is_enabled():
        return None
    prompt = build_signal_explainer_prompt(payload)
    try:
        return generate(prompt, temperature=0.3, num_ctx=_DEFAULT_CTX)
    except Exception as exc:
        logger.debug("Signal explainer generation failed: %s", exc)
        return None


def build_post_trade_summary_prompt(payload: Mapping[str, Any]) -> str:
    return (
        "Summarise the closed trade for a performance log in 2 sentences."
        " Mention entry, exit, outcome and realised PnL relative to risk.\n"
        f"Symbol: {payload.get('symbol')}\n"
        f"Direction: {payload.get('direction')}\n"
        f"Entry: {payload.get('entry_price')}\n"
        f"Exit: {payload.get('exit_price')}\n"
        f"Outcome: {payload.get('outcome')}\n"
        f"Reason: {payload.get('reason')}\n"
        f"PnL: {payload.get('pnl')}\n"
        f"Holding Time (mins): {payload.get('holding_minutes')}\n"
        f"Notes: {payload.get('notes')}\n"
        "Finish with a suggested improvement for future trades."
    )


def generate_post_trade_summary(payload: Mapping[str, Any]) -> str | None:
    base_summary = (
        f"{payload.get('symbol', 'Unknown')} {payload.get('direction', '').upper()} trade "
        f"closed at {payload.get('exit_price')} (entry {payload.get('entry_price')}). "
        f"Outcome: {payload.get('outcome', 'n/a')} | PnL: {payload.get('pnl')} "
        f"| Hold: {payload.get('holding_minutes')}m."
    )
    if not is_enabled():
        return base_summary
    prompt = build_post_trade_summary_prompt(payload)
    try:
        response = generate(prompt, temperature=0.2, num_ctx=_DEFAULT_CTX)
        return response or base_summary
    except Exception as exc:
        logger.debug("Post-trade summary generation failed: %s", exc)
        return base_summary


async def async_structured_judgment(prompt: str, *, temperature: float = 0.2, num_ctx: int | None = None) -> str | None:
    """Async wrapper for structured judgment using ``asyncio.to_thread``."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: structured_trade_judgment(prompt, temperature=temperature, num_ctx=num_ctx),
    )
