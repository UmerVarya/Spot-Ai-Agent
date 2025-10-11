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
from typing import Any, Mapping, MutableMapping

from log_utils import setup_logger

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
            "content": (
                "You are a quantitative crypto trading assistant. Respond only with JSON containing"
                " decision (Yes/No), confidence (0-10 float), reason, and thesis (2-3 sentences)."
            ),
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


def build_risk_check_prompt(payload: Mapping[str, Any]) -> str:
    """Format the guardrail prompt instructing the model to return JSON."""

    return (
        "You are a risk manager ensuring trades obey guardrails. Review the metrics and return JSON with keys"
        " approve (true/false), risk_flags (array of short strings), and comment (short guidance).\n"
        f"Symbol: {payload.get('symbol')}\n"
        f"Direction: {payload.get('direction')}\n"
        f"Model Confidence: {payload.get('confidence')}\n"
        f"ML Win Probability: {payload.get('ml_probability')}\n"
        f"Volatility Percentile: {payload.get('volatility')}\n"
        f"HTF Trend %: {payload.get('htf_trend_pct')}\n"
        f"Order Flow Note: {payload.get('orderflow')}\n"
        f"Macro Bias: {payload.get('macro_bias')}\n"
        f"Session: {payload.get('session')}\n"
        f"Setup Type: {payload.get('setup_type')}\n"
        f"Risk Limits: max_trades={payload.get('max_trades')} open_positions={payload.get('open_positions')}\n"
        "Be conservative if volatility is extreme, conviction is low, or risk flags conflict."
    )


def _parse_risk_json(raw: str) -> Mapping[str, Any] | None:
    try:
        data = json.loads(raw)
        if not isinstance(data, MutableMapping):
            return None
        return data
    except Exception:
        return None


def run_pretrade_risk_check(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Ask the local model to vet a trade before it is submitted."""

    if not is_enabled():
        return None
    prompt = build_risk_check_prompt(payload)
    try:
        raw = generate(prompt, temperature=0.1, num_ctx=_DEFAULT_CTX)
    except Exception as exc:
        logger.debug("Local risk check failed: %s", exc)
        return None
    parsed = _parse_risk_json(raw)
    if parsed is None:
        logger.debug("Local risk check returned unparseable payload: %s", raw)
        return {
            "approve": True,
            "risk_flags": ["unparsed_response"],
            "comment": str(raw)[:200],
        }
    return parsed


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
    if not is_enabled():
        return None
    prompt = build_post_trade_summary_prompt(payload)
    try:
        return generate(prompt, temperature=0.2, num_ctx=_DEFAULT_CTX)
    except Exception as exc:
        logger.debug("Post-trade summary generation failed: %s", exc)
        return None


async def async_structured_judgment(prompt: str, *, temperature: float = 0.2, num_ctx: int | None = None) -> str | None:
    """Async wrapper for structured judgment using ``asyncio.to_thread``."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: structured_trade_judgment(prompt, temperature=temperature, num_ctx=num_ctx),
    )
