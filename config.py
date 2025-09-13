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


# Default Groq model and mapping for deprecated names
# ``llama-3.1-70b-versatile`` has been decommissioned; use a currently
# available model instead.
DEFAULT_GROQ_MODEL = "llama-3.1-70b"
_DEPRECATED_GROQ_MODELS = {
    "llama3-70b-8192": DEFAULT_GROQ_MODEL,
    "llama-3.1-70b-versatile": DEFAULT_GROQ_MODEL,
}


def get_groq_model() -> str:
    """Return a supported Groq model name.

    If ``GROQ_MODEL`` is set to a deprecated model identifier, it is mapped to
    the current default model so API calls do not fail with a 400 error.
    """

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return _DEPRECATED_GROQ_MODELS.get(model, model)
