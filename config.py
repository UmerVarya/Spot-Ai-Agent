"""Central configuration loader for environment variables."""
from __future__ import annotations

from dotenv import load_dotenv

# Load environment variables once when this module is imported.
load_dotenv()

import os

def get(key: str, default: str | None = None) -> str | None:
    """Retrieve an environment variable with an optional default."""
    return os.getenv(key, default)
