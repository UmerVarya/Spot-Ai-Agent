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


# Default Groq model and mapping for deprecated names.
# Groq periodically retires older Llama releases (for example the
# ``llama-3.1-70b-versatile`` variant).  To keep the application working
# without manual intervention we map known, deprecated identifiers to the
# latest compatible default.  Update the mapping whenever Groq announces a
# replacement model.  The current production-ready choice is ``llama-3.3-70b-
# versatile`` which Groq lists as the successor for the retired 3.1 models.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_DEPRECATED_GROQ_MODELS = {
    "llama3-70b-8192": DEFAULT_GROQ_MODEL,
    "llama-3.1-70b": DEFAULT_GROQ_MODEL,
    "llama-3.1-70b-versatile": DEFAULT_GROQ_MODEL,
}
_DEPRECATED_LOOKUP = {key.lower(): value for key, value in _DEPRECATED_GROQ_MODELS.items()}


def get_groq_model() -> str:
    """Return a supported Groq model name.

    If ``GROQ_MODEL`` is set to a deprecated model identifier, it is mapped to
    the current default model so API calls do not fail with a 400 error.
    """

    raw_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if raw_model is None:
        return DEFAULT_GROQ_MODEL

    normalized = raw_model.strip()
    if not normalized:
        return DEFAULT_GROQ_MODEL

    replacement = _DEPRECATED_LOOKUP.get(normalized.lower())
    if replacement:
        return replacement

    return normalized
