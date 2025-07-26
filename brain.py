"""
Decision logic for the Spot AI Super Agent.

This module coordinates signals from technical indicators, machine
learning, and a large language model (LLM) to decide whether to open a
trade.  It uses a structured prompt for the LLM, ensures deterministic
JSON output, and blends various confidence scores.  Historical trade
statistics are incorporated via the ``symbol_scores.json`` and
``trade_learning_log.csv`` files to adjust baseline confidence.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

from groq_llm import get_llm_judgment
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence


def _load_symbol_scores() -> Dict[str, Any]:
    path = "symbol_scores.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def should_trade(
    symbol: str,
    score: float,
    direction: str,
    indicators: Dict[str, float],
    session: str,
    pattern_name: str,
    orderflow: str,
    sentiment: Dict[str, Any],
    macro_news: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether to enter a trade using LLM and historical context.

    Returns a dict with keys ``decision`` (bool), ``confidence`` (float) and
    ``narrative`` (str) describing the rationale.
    """
    # Baseline confidence from pattern memory and historical performance
    pattern_conf = recall_pattern_confidence(pattern_name)
    hist_conf = calculate_historical_confidence(symbol)
    baseline_conf = min(10.0, score * 0.6 + pattern_conf * 0.2 + hist_conf * 0.2)
    # Construct prompt payload for LLM
    prompt = {
        "symbol": symbol,
        "score": score,
        "direction": direction,
        "indicators": indicators,
        "session": session,
        "pattern": pattern_name,
        "order_flow": orderflow,
        "sentiment": sentiment,
        "macro_news": macro_news,
        "baseline_confidence": baseline_conf,
    }
    llm_out = get_llm_judgment(prompt)
    # Parse and blend confidence
    llm_decision = llm_out.get("decision", False)
    llm_confidence = float(llm_out.get("confidence", 0.0))
    reason = llm_out.get("reason", "")
    # Adjust final confidence
    final_confidence = (baseline_conf + llm_confidence) / 2.0
    return {
        "decision": bool(llm_decision),
        "confidence": float(final_confidence),
        "narrative": reason,
        "reason": reason,
    }
