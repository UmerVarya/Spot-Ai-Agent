"""Utilities for combining specialist financial language models.

The original pipeline relied predominantly on FinBERT with a secondary
FinLlama opinion.  Recent research shows that FinLlama and newer open
models such as FinGPT are markedly more numerically aware and resilient
to long-form macro headlines than FinBERT.  This module therefore
provides:

* Lightweight wrappers for FinBERT (classification), FinLlama and
  FinGPT (instruction-tuned causal LLMs) with graceful degradation when
  the heavy dependencies are missing.
* Dynamic fusion weights that default to emphasising FinLlama and FinGPT
  but can be recalibrated using a historical validation set.
* Helper routines that evaluate each model and search a simplex grid of
  candidate weights so the fused score can be tuned as new financial LLMs
  appear in 2024–2025.

Mapping:
    * FinBERT expectation ``s_fb`` lies in ``[-1, 1]`` and is derived from
      positive minus negative probabilities.  ``c_fb`` is the mean maximum
      class probability across headlines.
    * FinLlama sentiment ``s_fl`` is in ``{-1, 0, 1}`` (bearish, neutral,
      bullish) with confidence ``c_fl`` between ``0`` and ``1``.
    * FinGPT sentiment ``s_fg`` mirrors FinLlama's output schema.
    * Fused score defaults to ``0.2*s_fb + 0.45*s_fl + 0.35*s_fg`` with
      configurable weights that always renormalise to one.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
FINGPT_MODEL = "AI4Finance/FinGPT-4B-2024"  # released 2024, instruction tuned

DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
    "finbert": 0.20,
    "finllama": 0.45,
    "fingpt": 0.35,
}

_finbert_pipe = None
_finllama_model = None
_finllama_tokenizer = None
_fingpt_model = None
_fingpt_tokenizer = None


@dataclass
class SentimentResult:
    """Container for a model's score, confidence and optional metadata."""

    score: float
    confidence: float
    rationale: Optional[str] = None


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


def _load_fingpt() -> None:
    """Load FinGPT model and tokenizer when available."""

    global _fingpt_model, _fingpt_tokenizer
    if _fingpt_model is None and AutoTokenizer is not None:
        try:
            _fingpt_tokenizer = AutoTokenizer.from_pretrained(FINGPT_MODEL)
            _fingpt_model = AutoModelForCausalLM.from_pretrained(FINGPT_MODEL)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load FinGPT: %s", exc)


def _fingpt_sentiment(headlines: List[str]) -> Tuple[int, float, str]:
    """Return FinGPT sentiment, mirroring the FinLlama schema."""

    _load_fingpt()
    if _fingpt_model is None:
        return 0, 0.0, "FinGPT unavailable"
    prompt = (
        "You are FinGPT, a financial markets specialist.\n"
        "Determine the aggregate sentiment (bullish, bearish, neutral) for the following headlines. "
        "Respond as compact JSON with keys 'sentiment', 'confidence', 'rationale'.\n"
        + "\n".join(f"- {h}" for h in headlines)
    )
    inputs = _fingpt_tokenizer(prompt, return_tensors="pt")
    output = _fingpt_model.generate(**inputs, max_new_tokens=80)
    text = _fingpt_tokenizer.decode(output[0], skip_special_tokens=True)
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


def _normalise_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Return a non-negative weight mapping that sums to one."""

    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if not math.isfinite(total) or total <= 0:
        return {key: 1.0 / len(clipped) for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def analyze_headlines(
    headlines: List[str],
    *,
    fusion_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return fused sentiment analysis for a list of headlines."""

    weights = _normalise_weights(fusion_weights or DEFAULT_FUSION_WEIGHTS)
    s_fb, c_fb, fb_details = _finbert_expectation(headlines)
    s_fl, c_fl, rationale_fl = _finllama_sentiment(headlines)
    s_fg, c_fg, rationale_fg = _fingpt_sentiment(headlines)

    model_outputs: Dict[str, SentimentResult] = {
        "finbert": SentimentResult(
            score=s_fb, confidence=c_fb, rationale=json.dumps(fb_details)
        ),
        "finllama": SentimentResult(score=float(s_fl), confidence=c_fl, rationale=rationale_fl),
        "fingpt": SentimentResult(score=float(s_fg), confidence=c_fg, rationale=rationale_fg),
    }

    active_weights = {
        name: weights.get(name, 0.0)
        for name, result in model_outputs.items()
        if result.confidence > 0.0
    }
    if not active_weights:
        active_weights = dict(weights)
    fused_weights = _normalise_weights(active_weights)

    fused = sum(
        model_outputs[name].score * fused_weights.get(name, 0.0)
        for name in fused_weights
    )
    bias = "bullish" if fused > 0.15 else "bearish" if fused < -0.15 else "neutral"
    fused_conf = sum(
        model_outputs[name].confidence * fused_weights.get(name, 0.0)
        for name in fused_weights
    )
    return {
        "finbert": {"score": s_fb, "confidence": c_fb, "details": fb_details},
        "finllama": {"score": s_fl, "confidence": c_fl, "rationale": rationale_fl},
        "fingpt": {"score": s_fg, "confidence": c_fg, "rationale": rationale_fg},
        "fused": {"score": fused, "bias": bias, "confidence": fused_conf},
        "weights": dict(fused_weights),
    }


def _simplex_grid(step: float, labels: Sequence[str]) -> Iterable[Dict[str, float]]:
    """Yield weight dictionaries on the simplex with ``sum == 1``."""

    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    resolution = int(round(1 / step))
    indices = range(resolution + 1)
    for combo in itertools.product(indices, repeat=len(labels)):
        if sum(combo) != resolution:
            continue
        yield {label: count * step for label, count in zip(labels, combo)}


def evaluate_models(
    validation_set: Sequence[Tuple[List[str], float]],
    *,
    analyzer=analyze_headlines,
    fusion_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return per-model error metrics on a validation set.

    ``validation_set`` is an iterable of ``(headlines, target_score)``
    pairs.  ``target_score`` should lie within ``[-1, 1]`` and represent
    the ground-truth sentiment label for the aggregated headlines.
    """

    metrics: Dict[str, Dict[str, float]] = {}
    per_model_errors: Dict[str, List[float]] = {}
    fused_errors: List[float] = []

    for headlines, target in validation_set:
        result = analyzer(headlines, fusion_weights=fusion_weights)
        fused_errors.append(result["fused"]["score"] - target)
        for key in ("finbert", "finllama", "fingpt"):
            if key not in result:
                continue
            per_model_errors.setdefault(key, []).append(result[key]["score"] - target)

    def summarise(errors: List[float]) -> Dict[str, float]:
        if not errors:
            return {"mae": float("nan"), "rmse": float("nan")}
        abs_errors = [abs(err) for err in errors]
        mse = sum(err ** 2 for err in errors) / len(errors)
        return {"mae": sum(abs_errors) / len(errors), "rmse": math.sqrt(mse)}

    for key, errors in per_model_errors.items():
        metrics[key] = summarise(errors)
    metrics["fused"] = summarise(fused_errors)
    return metrics


def calibrate_fusion_weights(
    validation_set: Sequence[Tuple[List[str], float]],
    *,
    step: float = 0.05,
    analyzer=analyze_headlines,
    base_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Search a coarse grid for fusion weights with the lowest MAE."""

    labels = ["finbert", "finllama", "fingpt"]
    base = _normalise_weights(base_weights or DEFAULT_FUSION_WEIGHTS)
    best_weights = dict(base)
    best_mae = float("inf")

    cache: List[Tuple[Dict[str, Dict[str, float]], float]] = []
    for headlines, target in validation_set:
        cache.append((analyzer(headlines, fusion_weights=base), target))

    for candidate in _simplex_grid(step, labels):
        errors = []
        for result, target in cache:
            fused_score = sum(
                result.get(label, {}).get("score", 0.0) * candidate.get(label, 0.0)
                for label in labels
            )
            errors.append(abs(fused_score - target))
        mae = sum(errors) / len(errors) if errors else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_weights = _normalise_weights(candidate)
    return best_weights


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
