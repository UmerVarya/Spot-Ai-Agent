"""Utilities for combining FinBERT and FinLlama sentiment models.

This module fuses fast FinBERT classification with a light FinLlama
model to obtain finance‑specific sentiment with a simple numeric score.
FinBERT provides class probabilities for individual headlines while
FinLlama aggregates the headlines into a discrete sentiment with a short
rationale.  The two models are then combined into a single weighted score
that can be used upstream by the Groq LLM for broader reasoning.

Mapping:
    * FinBERT expectation ``s_fb`` lies in ``[-1, 1]`` and is derived from
      positive minus negative probabilities.  ``c_fb`` is the mean maximum
      class probability across headlines.
    * FinLlama sentiment ``s_fl`` is in ``{-1, 0, 1}`` (bearish, neutral,
      bullish) with confidence ``c_fl`` between ``0`` and ``1``.
    * Fused score ``fused = 0.55*s_fb + 0.45*s_fl``.  Bias is determined
      by thresholds ``>+0.15`` bullish, ``<-0.15`` bearish, otherwise
      neutral.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:  # optional heavy dependency
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )
except Exception:  # pragma: no cover - handles absence gracefully
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None
    logger.warning("transformers not available; returning neutral sentiment")

FINBERT_MODEL = "yiyanghkust/finbert-tone"
FINLLAMA_MODEL = "kaiokendev/FinLlama-7B"  # placeholder model name

_finbert_pipe = None
_finllama_model = None
_finllama_tokenizer = None


def _load_finbert() -> None:
    """Load FinBERT pipeline if possible."""
    global _finbert_pipe
    if _finbert_pipe is None and pipeline is not None:
        try:
            _finbert_pipe = pipeline(
                "text-classification", model=FINBERT_MODEL, return_all_scores=True
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to load FinBERT: %s", exc)


def _finbert_expectation(headlines: List[str]) -> Tuple[float, float, List[Dict[str, float]]]:
    """Compute expectation ``s_fb`` and confidence ``c_fb`` from FinBERT."""
    _load_finbert()
    if _finbert_pipe is None:
        return 0.0, 0.0, []
    expectations: List[float] = []
    confidences: List[float] = []
    details: List[Dict[str, float]] = []
    for hl in headlines:
        result = _finbert_pipe(hl)[0]
        probs = {item["label"].lower(): item["score"] for item in result}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        expectation = pos - neg  # [-1,1]
        confidence = max(probs.values()) if probs else 0.0
        expectations.append(expectation)
        confidences.append(confidence)
        details.append(probs)
    s_fb = sum(expectations) / len(expectations) if expectations else 0.0
    c_fb = sum(confidences) / len(confidences) if confidences else 0.0
    return s_fb, c_fb, details


def _load_finllama() -> None:
    """Load FinLlama model and tokenizer."""
    global _finllama_model, _finllama_tokenizer
    if _finllama_model is None and AutoTokenizer is not None:
        try:
            _finllama_tokenizer = AutoTokenizer.from_pretrained(FINLLAMA_MODEL)
            _finllama_model = AutoModelForCausalLM.from_pretrained(FINLLAMA_MODEL)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load FinLlama: %s", exc)


def _finllama_sentiment(headlines: List[str]) -> Tuple[int, float, str]:
    """Get discrete sentiment ``s_fl`` with confidence and rationale."""
    _load_finllama()
    if _finllama_model is None:
        return 0, 0.0, "FinLlama unavailable"
    prompt = (
        "You are FinLlama, a finance news analyst.\n"
        "Classify the overall sentiment of these headlines as bullish, bearish or neutral, "
        "provide confidence 0-1 and a 1-2 line rationale in JSON with keys 'sentiment', 'confidence', 'rationale'.\n"
        + "\n".join(f"- {h}" for h in headlines)
    )
    inputs = _finllama_tokenizer(prompt, return_tensors="pt")
    output = _finllama_model.generate(**inputs, max_new_tokens=80)
    text = _finllama_tokenizer.decode(output[0], skip_special_tokens=True)
    sentiment = 0
    confidence = 0.0
    rationale = text.strip()
    try:  # best effort JSON parse
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            bias = str(data.get("sentiment", "neutral")).lower()
            sentiment = {"bullish": 1, "bearish": -1, "neutral": 0}.get(bias, 0)
            confidence = float(data.get("confidence", 0.0))
            rationale = data.get("rationale", rationale)
    except Exception:  # pragma: no cover
        pass
    return sentiment, confidence, rationale


def analyze_headlines(headlines: List[str]) -> Dict[str, Dict[str, float]]:
    """Return fused sentiment analysis for a list of headlines."""
    s_fb, c_fb, fb_details = _finbert_expectation(headlines)
    s_fl, c_fl, rationale = _finllama_sentiment(headlines)
    fused = 0.55 * s_fb + 0.45 * s_fl
    bias = "bullish" if fused > 0.15 else "bearish" if fused < -0.15 else "neutral"
    confidence = 0.55 * c_fb + 0.45 * c_fl
    return {
        "finbert": {"score": s_fb, "confidence": c_fb, "details": fb_details},
        "finllama": {"score": s_fl, "confidence": c_fl, "rationale": rationale},
        "fused": {"score": fused, "bias": bias, "confidence": confidence},
    }


def arbitrate_with_groq(fused_result: Dict[str, Dict[str, float]], context: str) -> str:
    """Ask the Groq LLM to approve or veto a setup based on fused sentiment.

    The function simply forwards a compact description of the fused
    sentiment along with additional context (e.g. macro events or
    technical signals) to the existing :func:`groq_llm.get_llm_judgment`.
    If the Groq API key is missing, a fallback string is returned.
    """
    try:
        from groq_llm import get_llm_judgment  # late import to avoid cycle
    except Exception:  # pragma: no cover
        return "Groq module unavailable"
    score = fused_result.get("fused", {}).get("score", 0.0)
    bias = fused_result.get("fused", {}).get("bias", "neutral")
    conf = fused_result.get("fused", {}).get("confidence", 0.0)
    prompt = (
        f"Fused sentiment: {bias} (score {score:.3f}, confidence {conf:.2f}).\n" +
        f"Context: {context}\n" +
        "Provide approval or veto with justification in JSON."
    )
    return get_llm_judgment(prompt)
