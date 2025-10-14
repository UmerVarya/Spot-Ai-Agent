"""Central configuration loader for environment variables."""
from __future__ import annotations

from dotenv import load_dotenv

# Load environment variables once when this module is imported.
load_dotenv()

import os


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def get(key: str, default: str | None = None) -> str | None:
    """Retrieve an environment variable with an optional default."""
    return os.getenv(key, default)


# Model presets for different workloads -------------------------------------------------
#
# We route requests to specialised Groq (or Groq-compatible) models depending on the
# task to spread rate-limit pressure:
#
# * ``DEFAULT_GROQ_MODEL`` – Qwen 32B is the primary reasoning model used for trade
#   decisions and approvals.
# * ``DEFAULT_MACRO_MODEL`` – Meta Llama Scout handles bursty macro/regime updates.
# * ``DEFAULT_NEWS_MODEL`` – Llama 3.1 Instant is the light-weight summariser for
#   news filtering and narrative copy.
# * ``DEFAULT_OVERFLOW_MODEL`` – Llama 3.3 70B remains as the high-quality overflow
#   when other endpoints are rate limited or unavailable.
DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"
DEFAULT_MACRO_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_NEWS_MODEL = "llama-3.1-8b-instant"
DEFAULT_OVERFLOW_MODEL = "llama-3.3-70b-versatile"

# Groq periodically retires older Llama releases (for example the
# ``llama-3.1-70b-versatile`` variant).  To keep the application working without manual
# intervention we map known, deprecated identifiers to the latest compatible overflow
# model.  Update the mapping whenever Groq announces a replacement model.
_DEPRECATED_GROQ_MODELS = {
    "llama3-70b-8192": DEFAULT_OVERFLOW_MODEL,
    "llama-3.1-70b": DEFAULT_OVERFLOW_MODEL,
    "llama-3.1-70b-versatile": DEFAULT_OVERFLOW_MODEL,
}
_DEPRECATED_LOOKUP = {key.lower(): value for key, value in _DEPRECATED_GROQ_MODELS.items()}


def _resolve_model(env_var: str, default: str) -> str:
    """Resolve the configured model name for ``env_var`` falling back to ``default``."""

    raw_model = os.getenv(env_var)
    normalized = raw_model.strip() if raw_model else ""
    if not normalized:
        normalized = default

    replacement = _DEPRECATED_LOOKUP.get(normalized.lower())
    if replacement:
        return replacement

    return normalized


def get_groq_model() -> str:
    """Return the primary reasoning model for trade approvals."""

    return _resolve_model("TRADE_LLM_MODEL", DEFAULT_GROQ_MODEL)


def get_macro_model() -> str:
    """Return the high-burst model for macro sentiment/regime analysis."""

    return _resolve_model("MACRO_LLM_MODEL", DEFAULT_MACRO_MODEL)


def get_news_model() -> str:
    """Return the lightweight model for news filters and summaries."""

    return _resolve_model("NEWS_LLM_MODEL", DEFAULT_NEWS_MODEL)


def get_overflow_model() -> str:
    """Return the overflow model used when primaries are rate limited or down."""

    return _resolve_model("GROQ_OVERFLOW_MODEL", DEFAULT_OVERFLOW_MODEL)
