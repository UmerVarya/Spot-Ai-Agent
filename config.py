"""Central configuration loader for environment variables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

from dotenv import load_dotenv

# Load environment variables once when this module is imported.
load_dotenv()

import os


def _clean_path(value: str | None) -> str:
    """Return ``value`` without inline comments or surrounding whitespace."""

    if not value:
        return ""
    return value.split("#", 1)[0].strip()


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for ``path`` when possible."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM routing feature flags
# ---------------------------------------------------------------------------


def _truthy(x: str | None) -> bool:
    return str(x or "").strip().lower() in {"1", "true", "yes", "on"}


def use_groq() -> bool:
    return _truthy(os.getenv("USE_GROQ", "true"))


def get_default_groq_model() -> str:
    return os.getenv("DEFAULT_GROQ_MODEL", "qwen/qwen3-32b")


def get_macro_groq_model() -> str:
    return os.getenv("MACRO_GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def get_news_groq_model() -> str:
    return os.getenv("NEWS_GROQ_MODEL", "llama-3.1-8b-instant")


def get_narrative_groq_model() -> str:
    return os.getenv("NARRATIVE_GROQ_MODEL", get_news_groq_model())


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
DEFAULT_GROQ_MODEL = get_default_groq_model()
DEFAULT_MACRO_MODEL = get_macro_groq_model()
DEFAULT_NEWS_MODEL = get_news_groq_model()
DEFAULT_NARRATIVE_MODEL = get_narrative_groq_model()
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


def _resolve_model(env_var: str | tuple[str, ...], default: str) -> str:
    """Resolve the configured model name for ``env_var`` falling back to ``default``."""

    env_sources: tuple[str, ...]
    if isinstance(env_var, str):
        env_sources = (env_var,)
    else:
        env_sources = env_var

    normalized = ""
    for candidate in env_sources:
        raw_model = os.getenv(candidate)
        normalized = raw_model.strip() if raw_model else ""
        if normalized:
            break

    if not normalized:
        normalized = default

    replacement = _DEPRECATED_LOOKUP.get(normalized.lower())
    if replacement:
        return replacement

    return normalized


def get_groq_model() -> str:
    """Return the primary reasoning model for trade approvals."""

    return _resolve_model(("TRADE_LLM_MODEL", "GROQ_MODEL"), DEFAULT_GROQ_MODEL)


def get_macro_model() -> str:
    """Return the high-burst model for macro sentiment/regime analysis."""

    return _resolve_model("MACRO_LLM_MODEL", DEFAULT_MACRO_MODEL)


def get_news_model() -> str:
    """Return the lightweight model for news filters and summaries."""

    return _resolve_model(("NEWS_LLM_MODEL", "NEWS_GROQ_MODEL"), DEFAULT_NEWS_MODEL)


def get_narrative_model() -> str:
    """Return the model dedicated to narrative and recap generation."""

    return _resolve_model(
        ("NARRATIVE_LLM_MODEL", "NARRATIVE_GROQ_MODEL", "NEWS_LLM_MODEL"),
        DEFAULT_NARRATIVE_MODEL,
    )


def get_overflow_model() -> str:
    """Return the overflow model used when primaries are rate limited or down."""

    return _resolve_model("GROQ_OVERFLOW_MODEL", DEFAULT_OVERFLOW_MODEL)


# ---------------------------------------------------------------------------
# Runtime configuration for the real-time agent
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


# ---------------------------------------------------------------------------
# Trade storage locations
# ---------------------------------------------------------------------------

DEFAULT_TRADE_DATA_DIR = "/home/ubuntu/spot_data/trades"


def _resolve_trade_data_dir() -> str:
    candidate = _clean_path(os.getenv("DATA_DIR")) or DEFAULT_TRADE_DATA_DIR
    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except OSError:
        os.makedirs(DEFAULT_TRADE_DATA_DIR, exist_ok=True)
        return DEFAULT_TRADE_DATA_DIR


def _resolve_history_override() -> tuple[str, bool]:
    override = _clean_path(os.getenv("TRADE_HISTORY_FILE"))
    if not override:
        override = _clean_path(os.getenv("COMPLETED_TRADES_FILE"))
    return override, bool(override)


def _resolve_backtest_override() -> str:
    return _clean_path(os.getenv("BACKTEST_TRADE_HISTORY_FILE"))


def _resolve_history_path(default_path: str, override: str) -> str:
    path = override or default_path
    if path:
        _ensure_parent_dir(path)
    return path


TRADE_DATA_DIR = _resolve_trade_data_dir()
_TRADE_HISTORY_OVERRIDE, TRADE_HISTORY_ENV_OVERRIDE = _resolve_history_override()
TRADE_HISTORY_FILE = _resolve_history_path(
    os.path.join(TRADE_DATA_DIR, "historical_trades.csv"), _TRADE_HISTORY_OVERRIDE
)
BACKTEST_TRADE_HISTORY_FILE = _resolve_history_path(
    os.path.join(TRADE_DATA_DIR, "backtest_trades.csv"), _resolve_backtest_override()
)

# Keys used to collapse duplicate rows in trade history consolidation.
TRADE_DEDUP_KEYS = ("trade_id", "entry_time", "symbol", "strategy")


@dataclass(frozen=True)
class SymbolOverride:
    """Per-symbol runtime overrides loaded from configuration."""

    debounce_ms: int | None = None
    refresh_interval: float | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime configuration knobs for the live trading agent."""

    use_ws_prices: bool = True
    use_ws_book_ticker: bool = True
    use_user_stream: bool = True
    rest_backfill_enabled: bool = True
    debounce_ms: int = 1000
    refresh_interval: float = 2.0
    max_symbols: int = 30
    max_queue: int = 100
    max_ws_gap_before_rest: float = 5.0
    server_time_sync_interval: float = 120.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_window: float = 30.0
    symbol_overrides: Dict[str, SymbolOverride] = field(default_factory=dict)


def _parse_symbol_overrides() -> Dict[str, SymbolOverride]:
    raw = os.getenv("SYMBOL_RUNTIME_OVERRIDES")
    overrides: Dict[str, SymbolOverride] = {}
    if raw:
        import json

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, Mapping):
            for key, value in data.items():
                if not isinstance(value, Mapping):
                    continue
                debounce = value.get("debounce_ms")
                refresh = value.get("refresh_interval")
                try:
                    debounce_val = int(debounce) if debounce is not None else None
                except (TypeError, ValueError):
                    debounce_val = None
                try:
                    refresh_val = float(refresh) if refresh is not None else None
                except (TypeError, ValueError):
                    refresh_val = None
                overrides[str(key).upper()] = SymbolOverride(
                    debounce_ms=debounce_val, refresh_interval=refresh_val
                )
    if "ETHUSDT" not in overrides:
        overrides["ETHUSDT"] = SymbolOverride(debounce_ms=1000)
    return overrides


def load_runtime_settings() -> RuntimeSettings:
    """Load runtime settings for the live agent from environment variables."""

    return RuntimeSettings(
        use_ws_prices=_env_bool("USE_WS_PRICES", True),
        use_ws_book_ticker=_env_bool("USE_WS_BOOK_TICKER", True),
        use_user_stream=_env_bool("USE_USER_STREAM", True),
        rest_backfill_enabled=_env_bool("REST_BACKFILL_ENABLED", True),
        debounce_ms=max(100, _env_int("DEBOUNCE_MS", 1000)),
        refresh_interval=max(0.5, _env_float("REFRESH_INTERVAL", 2.0)),
        max_symbols=max(1, _env_int("MAX_SYMBOLS", 30)),
        max_queue=max(10, _env_int("MAX_EVENT_QUEUE", 100)),
        max_ws_gap_before_rest=max(1.0, _env_float("MAX_WS_GAP_BEFORE_REST", 5.0)),
        server_time_sync_interval=max(30.0, _env_float("SERVER_TIME_SYNC_INTERVAL", 120.0)),
        circuit_breaker_threshold=max(1, _env_int("EVALUATOR_CIRCUIT_BREAKER_THRESHOLD", 5)),
        circuit_breaker_window=max(5.0, _env_float("EVALUATOR_CIRCUIT_BREAKER_WINDOW", 30.0)),
        symbol_overrides=_parse_symbol_overrides(),
    )


def use_groq(default: bool = True) -> bool:
    """Return ``True`` when Groq should be used as the primary LLM provider."""

    return _env_bool("USE_GROQ", default)


def get_default_groq_model() -> str:
    """Return the default Groq model for trade workflows."""

    return get_groq_model()


def get_macro_groq_model() -> str:
    """Return the Groq model for macro analysis workloads."""

    return get_macro_model()


def get_narrative_groq_model() -> str:
    """Return the Groq model for narrative/news generation workloads."""

    return get_narrative_model()


__all__ = [
    "get",
    "get_groq_model",
    "get_macro_model",
    "get_news_model",
    "get_narrative_model",
    "get_overflow_model",
    "get_default_groq_model",
    "get_macro_groq_model",
    "get_narrative_groq_model",
    "use_groq",
    "load_runtime_settings",
    "RuntimeSettings",
    "SymbolOverride",
]
