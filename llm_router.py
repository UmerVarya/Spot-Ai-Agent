"""Centralized helpers for selecting LLM providers and models."""
from __future__ import annotations

import logging
import os
from typing import Optional

from some_module import groq_client, ollama_client

import config


logger = logging.getLogger(__name__)


class LLMChoice:
    """Container describing the selected provider, model, and instantiated client."""

    def __init__(self, provider: str, model: str, client):
        self.provider = provider
        self.model = model
        self.client = client


def _groq(model: str) -> LLMChoice:
    cli = groq_client(model=model)
    return LLMChoice("groq", model, cli)


def _ollama(model: str) -> LLMChoice:
    cli = ollama_client(model=model, base_url=config.get_ollama_url())
    return LLMChoice("ollama", model, cli)


def _try_groq(model: str) -> Optional[LLMChoice]:
    """Return a Groq client when enabled; otherwise ``None``."""

    if not config.use_groq():
        return None

    try:
        return _groq(model)
    except Exception as exc:  # noqa: BLE001 - surface provider errors
        if not config.use_ollama():
            raise RuntimeError("Groq client initialization failed") from exc
        logger.warning("Groq unavailable for model %s; falling back to Ollama", model, exc_info=exc)
        return None


def _require_ollama(model: str) -> LLMChoice:
    if not config.use_ollama():
        raise RuntimeError("No LLM provider enabled. Set USE_GROQ=true or USE_OLLAMA=true")
    return _ollama(model)


def get_trade_llm() -> LLMChoice:
    """Return the primary LLM for trade workflows."""

    groq_choice = _try_groq(config.get_default_groq_model())
    if groq_choice is not None:
        return groq_choice
    return _require_ollama(os.getenv("MODEL_ID", "llama3.2:8b"))


def get_macro_llm() -> LLMChoice:
    """Return the macro analysis LLM."""

    groq_choice = _try_groq(config.get_macro_groq_model())
    if groq_choice is not None:
        return groq_choice
    return _require_ollama(os.getenv("MACRO_MODEL_ID", "llama3.2:8b"))


def get_narrative_llm() -> LLMChoice:
    """Return the narrative generation LLM."""

    groq_choice = _try_groq(config.get_narrative_groq_model())
    if groq_choice is not None:
        return groq_choice
    return _require_ollama(os.getenv("NARRATIVE_MODEL_ID", "llama3.2:8b"))


__all__ = [
    "LLMChoice",
    "get_trade_llm",
    "get_macro_llm",
    "get_narrative_llm",
]
