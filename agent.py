"""
Main trading agent loop for the Spot AI Super Agent.

This script orchestrates periodic market scans, evaluates signals, asks the
brain whether to take trades, applies an ML-based veto and finally opens
trades in paper trading mode.  It also handles diversification to avoid
overexposure to correlated symbols, enforces a maximum number of open trades
and logs decisions and outcomes.

The call to ``select_diversified_signals`` was updated to pass the
``max_trades`` parameter explicitly.  Without this fix, the previously
computed ``allowed_new`` value was being treated as a correlation threshold,
resulting in at most two signals being selected regardless of available slots.
"""

# ---- kill legacy python-binance websockets globally when disabled ----
import os

if os.getenv("DISABLE_LEGACY_BINANCE_WS", "0") == "1":
    try:
        import binance.streams as _bn_streams

        class _NoopTWM:
            def __init__(self, *a, **k):
                pass

            def start(self, *a, **k):
                pass

            def stop(self, *a, **k):
                pass

            def join(self, *a, **k):
                pass

            # return dummy "conn_key" so callers don't crash
            def start_kline_socket(self, *a, **k):
                return None

            def start_miniticker_socket(self, *a, **k):
                return None

            def start_symbol_ticker_socket(self, *a, **k):
                return None

            def start_user_socket(self, *a, **k):
                return None

        _bn_streams.ThreadedWebsocketManager = _NoopTWM
    except Exception:
        pass
# ----------------------------------------------------------------------

import quiet_logging  # silences Binance/RTSC spam globally

from log_utils import setup_logger, LOG_FILE
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from signal_audit import log_signal_audit

logger = setup_logger(__name__)


def log_simple_decision_metric(
    symbol: str,
    *,
    action: str,
    direction: Optional[str],
    size_value: Optional[float],
    score_value: Optional[float],
    reason_text: Optional[str],
) -> None:
    """Emit the SIMPLE_DECISION metric without affecting control flow."""

    try:
        size = float(size_value) if size_value is not None else 0.0
    except (TypeError, ValueError):
        size = 0.0
    score = 0.0
    try:
        if score_value is not None:
            score = float(score_value)
    except (TypeError, ValueError):
        score = 0.0
    logger.info(
        "[METRIC] SIMPLE_DECISION: symbol=%s, action=%s, direction=%s, size=%.1f, score=%.2f, reason=%s",
        symbol,
        action,
        direction or "None",
        size,
        score,
        reason_text or "unspecified",
    )

import math
import time
import sys
import asyncio
import random
import logging
import threading
from datetime import datetime, timezone

from ws_price_bridge import WSPriceBridge
from decision_metrics import (
    ensure_breakdown_fields,
    log_decision_breakdown,
    maybe_log_summary,
    record_signal_evaluated,
    record_skip_reason,
    record_trade_opened,
    update_breakdown_reason,
)

# Centralized configuration loader
import config

runtime_settings = config.load_runtime_settings()
use_ws_prices = runtime_settings.use_ws_prices

SCAN_MIN_INTERVAL = 0.20  # seconds; keeps scans responsive without spamming
_scan_event: Optional[threading.Event] = None
_last_scan_fire = 0.0
_scan_lock = threading.Lock()
_scan_tick_thread: Optional[threading.Thread] = None
_scan_heartbeat_lock = threading.Lock()
_last_scan_heartbeat = 0.0


def attach_scan_event(ev: threading.Event) -> None:
    """Register the threading event used to wake the scan loop."""

    global _scan_event
    _scan_event = ev


def _trigger_scan(reason: str = "market") -> None:
    """Set the scan event with light global debouncing."""

    global _last_scan_fire
    event = _scan_event
    if event is None:
        return
    now = time.monotonic()
    with _scan_lock:
        if now - _last_scan_fire < SCAN_MIN_INTERVAL:
            return
        _last_scan_fire = now
    event.set()


async def notify_scan(reason: str = "market") -> None:
    """Coroutine-compatible helper to wake the scan loop."""

    _trigger_scan(reason)


def _mark_scan_heartbeat(ts: Optional[float] = None) -> None:
    """Record the wall-clock time of the most recent scan."""

    global _last_scan_heartbeat
    if ts is None:
        ts = time.time()
    with _scan_heartbeat_lock:
        _last_scan_heartbeat = ts


def _scan_idle_seconds(now: Optional[float] = None) -> float:
    """Return the number of seconds since the last completed scan."""

    with _scan_heartbeat_lock:
        last = _last_scan_heartbeat
    if last <= 0:
        return float("inf")
    if now is None:
        now = time.time()
    return max(0.0, now - last)


def _ensure_periodic_tick(interval: float = 5.0) -> None:
    """Start a lightweight periodic tick that keeps the scan loop warm."""

    global _scan_tick_thread
    if _scan_tick_thread is not None and _scan_tick_thread.is_alive():
        return

    def _runner() -> None:
        while True:
            time.sleep(interval)
            try:
                _trigger_scan("periodic")
            except Exception:
                logger.debug("Periodic scan tick failed", exc_info=True)

    _scan_tick_thread = threading.Thread(
        target=_runner, name="scan-periodic-tick", daemon=True
    )
    _scan_tick_thread.start()


async def _rtsc_diag_task(sc: Any) -> None:
    """Periodically log lightweight diagnostics for the RTSC instance."""

    interval = max(1, int(os.getenv("RTSC_DIAG_INTERVAL", "60")))
    while True:
        try:
            bars_len = None
            try:
                bars_len = sc.bars_len("BTCUSDT")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("RTSC_DIAG: bars_len failed: %s", exc, exc_info=True)

            symbols_attr = getattr(sc, "symbols", None)
            if callable(symbols_attr):
                try:
                    symbols_value = symbols_attr()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("RTSC_DIAG: symbols() failed: %s", exc, exc_info=True)
                    symbols_value = None
            else:
                symbols_value = symbols_attr

            if symbols_value is None:
                symbols_value = getattr(sc, "_symbols", [])

            try:
                sym_count = len(symbols_value)
            except Exception:
                sym_count = -1

            logger.info(
                "RTSC_DIAG: BTCUSDT bars_len=%s symbols=%s",
                bars_len,
                sym_count,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("RTSC_DIAG error: %s", exc)
        await asyncio.sleep(interval)


def _schedule_rtsc_diag(sc: Any) -> None:
    """Schedule the RTSC diagnostic task on an available asyncio loop."""

    if not sc or getattr(sc, "_rtsc_diag_started", False):
        return

    setattr(sc, "_rtsc_diag_started", True)

    def _submit(loop: asyncio.AbstractEventLoop) -> None:
        try:
            loop.create_task(_rtsc_diag_task(sc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("RTSC_DIAG: failed to create task: %s", exc, exc_info=True)

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        _submit(running_loop)
        return

    worker_loop = getattr(sc, "_worker_loop", None)
    if worker_loop and worker_loop.is_running():
        worker_loop.call_soon_threadsafe(lambda: asyncio.create_task(_rtsc_diag_task(sc)))
        return

    bg_loop = getattr(sc, "_loop", None)
    if bg_loop and bg_loop.is_running():
        bg_loop.call_soon_threadsafe(lambda: asyncio.create_task(_rtsc_diag_task(sc)))
        return

    ready_event = getattr(sc, "_worker_loop_ready", None)
    if ready_event and hasattr(ready_event, "wait"):
        def _wait_and_schedule() -> None:
            if ready_event.wait(timeout=5.0):
                loop = getattr(sc, "_worker_loop", None)
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(_rtsc_diag_task(sc))
                    )

        threading.Thread(target=_wait_and_schedule, daemon=True, name="rtsc-diag").start()
        return

    logger.debug("RTSC_DIAG: no available event loop to schedule diagnostic task")

# --- WS bootstrap (force start when USE_WS_PRICES=true) ---
_ws_bridge: Optional[WSPriceBridge] = None
try:
    if use_ws_prices:
        ws_symbols: list[str] = []
        for cand in (
            locals().get("top_symbols"),
            locals().get("tracked_symbols"),
            locals().get("scan_symbols"),
            locals().get("symbols"),
        ):
            if cand:
                ws_symbols = sorted({str(sym).upper() for sym in cand if sym})
                break

        if not ws_symbols:
            ws_symbols = ["BTCUSDT", "ETHUSDT"]

        if "_ws_bridge" not in globals() or _ws_bridge is None:
            _ws_bridge = WSPriceBridge(symbols=ws_symbols)
            try:
                from realtime_signal_cache import get_active_cache

                sc = get_active_cache()

                def _on_kline(msg: dict[str, Any]) -> None:
                    try:
                        symbol = str(msg.get("s") or msg.get("symbol") or "").upper()
                        if sc and hasattr(sc, "schedule_refresh"):
                            sc.schedule_refresh(symbol)
                    except Exception:
                        pass

                if hasattr(_ws_bridge, "register_kline_handler"):
                    _ws_bridge.register_kline_handler(_on_kline)
                elif hasattr(_ws_bridge, "register_callback"):

                    def _bridge_callback(symbol: str, event_type: str, payload: dict[str, Any]) -> None:
                        if event_type != "kline":
                            return
                        enriched = dict(payload)
                        if "s" not in enriched:
                            enriched["s"] = symbol
                        _on_kline(enriched)

                    _ws_bridge.register_callback(_bridge_callback)
            except Exception:
                pass

            starter = getattr(_ws_bridge, "enable_streams", None)
            if callable(starter):
                starter()
            else:
                start_method = getattr(_ws_bridge, "start", None)
                if callable(start_method):
                    start_method()
            logger.warning(
                "WS bootstrap: enabled combined stream for %d symbols.", len(ws_symbols)
            )
        else:
            logger.info("WS bootstrap: bridge already exists; skipping.")
except Exception as e:
    logger.exception("WS bootstrap failed: %s", e)
# --- end WS bootstrap ---
from fetch_news import (
    fetch_news,  # noqa: F401
    run_news_fetcher,  # noqa: F401
    run_news_fetcher_async,  # noqa: F401
    analyze_news_with_llm_async,
    get_news_cache,
    trigger_news_refresh,
)
from news_monitor import (
    get_news_monitor,
    start_background_news_monitor,
)
from trade_utils import simulate_slippage, estimate_commission  # noqa: F401
from trade_utils import (
    get_top_symbols,
    get_price_data_async,
    get_market_session,
    calculate_indicators,
    compute_performance_metrics,
    summarise_technical_score,
    get_order_book,
    load_symbol_scores,
    save_symbol_scores,
    SYMBOL_SCORES_FILE,
)
from trade_manager import (
    manage_trades,
    create_new_trade,
    process_live_kline,
    process_live_ticker,
    process_book_ticker,
    process_user_stream_event,
)  # trade logic
from trade_storage import (
    load_active_trades,
    save_active_trades,
    ACTIVE_TRADES_FILE,
    TRADE_HISTORY_FILE,
    log_trade_result,
)
from notifier import send_email, log_rejection, REJECTED_TRADES_FILE
from trade_logger import TRADE_LEARNING_LOG_FILE
from brain import (
    prepare_trade_decision,
    finalize_trade_decision,
)
from sentiment import get_macro_sentiment
from macro_data import get_btc_dominance_cached, get_fear_greed_cached
from orderflow import detect_aggression
from volume_profile import (
    VolumeProfileResult,
    compute_trend_leg_volume_profile,
    compute_reversion_leg_volume_profile,
)
from diversify import select_diversified_signals
from groq_llm import async_batch_llm_judgment
from groq_client import get_groq_client
from ml_model import predict_success_probability
from sequence_model import (
    SEQ_PKL,
    predict_next_return,
    schedule_sequence_model_training,
)
from drawdown_guard import is_trading_blocked
import numpy as np

from trade_constants import ATR_STOP_MULTIPLIER, TP_ATR_MULTIPLIERS
from rl_policy import RLPositionSizer
from trade_utils import get_rl_state, set_symbol_tiers
from microstructure import plan_execution
from volatility_regime import atr_percentile
from cache_evaluator_adapter import evaluator_for_cache
from realtime_signal_cache import RealTimeSignalCache, set_active_cache, get_active_cache
from ws_user_bridge import UserDataStreamBridge
from auction_state import get_auction_state
from alternative_data import get_alternative_data
from risk_veto import minutes_until_next_event, evaluate_risk_veto
from state_manager import CentralState
from worker_pools import ScheduledTask, WorkerPools
from market_stream import BinanceEventStream
from observability import log_event, record_metric


def warm_up_groq_client() -> bool:
    """Initialise the shared Groq client for downstream modules."""

    return get_groq_client() is not None


def run_pretrade_risk_check(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Evaluate deterministic risk veto rules before placing a trade."""

    return evaluate_risk_veto(payload)


def generate_signal_explainer(payload: Mapping[str, Any]) -> str:
    """Return a concise textual summary describing why a signal was accepted."""

    symbol = str(payload.get("symbol", "Unknown")).upper()
    pattern = payload.get("pattern") or "n/a"
    score = payload.get("score")
    confidence = payload.get("confidence")
    macro_bias = payload.get("macro_bias")
    orderflow = payload.get("orderflow")
    volume_profile = payload.get("volume_profile")
    context = payload.get("context")

    parts = [f"{symbol}: pattern {pattern}"]
    if score is not None:
        parts.append(f"score {score}")
    if confidence is not None:
        parts.append(f"confidence {confidence}")
    if macro_bias:
        parts.append(f"macro {macro_bias}")
    if orderflow:
        parts.append(f"order flow {orderflow}")
    if volume_profile:
        parts.append(f"volume {volume_profile}")
    if context:
        parts.append(f"context: {context}")

    return "; ".join(str(item) for item in parts if item).strip()


def dispatch_schedule_refresh(cache, symbol: str) -> None:
    # Always enqueue; the cache handles loop readiness/queueing internally
    try:
        cache.enqueue_refresh(symbol)
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[AGENT] enqueue_refresh failed for {symbol}: {e}"
        )


BREAKOUT_PATTERNS = {
    "triangle_wedge",
    "flag",
    "cup_handle",
    "double_bottom",
}


def handle_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions with stack traces."""
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def _auto_tune_signal_cache_params(
    symbol_target: int,
    refresh_interval: float,
    stale_multiplier: float,
    max_concurrency: int,
    scan_interval: float,
) -> tuple[float, int, float, float]:
    """Return adjusted stale multiplier, concurrency and scan interval.

    The tunings ensure the refresh workers can cycle through the tracked
    symbols without signals expiring prematurely. They also gently throttle the
    scan cadence when the cache would otherwise still be updating.
    """

    safe_refresh = max(0.5, float(refresh_interval))
    base_stale = max(1.0, safe_refresh * max(1.0, float(stale_multiplier)))
    configured_conc = max(1, int(max_concurrency))
    symbol_target = max(0, int(symbol_target))

    if symbol_target <= 1:
        tuned_scan = max(scan_interval, safe_refresh)
        return base_stale / safe_refresh, configured_conc, tuned_scan, safe_refresh

    desired_conc = max(4, math.ceil(symbol_target / 4))
    desired_conc = min(32, desired_conc)
    tuned_conc = max(configured_conc, desired_conc)

    estimated_cycle = safe_refresh * max(1, math.ceil(symbol_target / tuned_conc))
    min_stale_window = estimated_cycle * 2.5
    tuned_stale = max(base_stale, min_stale_window)

    tuned_scan = float(scan_interval)
    if tuned_scan < estimated_cycle * 0.75:
        tuned_scan = max(tuned_scan, estimated_cycle * 0.9)
    tuned_scan = max(tuned_scan, safe_refresh)

    return tuned_stale / safe_refresh, tuned_conc, tuned_scan, estimated_cycle


# Maximum concurrent open trades (strictly limited to one active position)
# (re-)declare constants for static type-checkers after helper definition
MAX_ACTIVE_TRADES = 1
SCAN_INTERVAL = float(os.getenv("SCAN_INTERVAL", "3"))
SIGNAL_REFRESH_INTERVAL = runtime_settings.refresh_interval
SIGNAL_STALE_MULT = float(os.getenv("SIGNAL_STALE_AFTER", "3"))
SIGNAL_STALE_AFTER = SIGNAL_REFRESH_INTERVAL * SIGNAL_STALE_MULT
MAX_CONCURRENT_FETCHES = int(os.getenv("MAX_CONCURRENT_FETCHES", "10"))
ENABLE_WS_BRIDGE = runtime_settings.use_ws_prices
ENABLE_USER_STREAM = runtime_settings.use_user_stream
REST_BACKFILL_ENABLED = runtime_settings.rest_backfill_enabled
SIGNAL_CACHE_PRIME_AFTER = float(os.getenv("SIGNAL_CACHE_PRIME_AFTER", "120"))
SIGNAL_CACHE_PRIME_COOLDOWN = float(os.getenv("SIGNAL_CACHE_PRIME_COOLDOWN", "300"))
SIGNAL_CACHE_PRIME_TIMEOUT = float(os.getenv("SIGNAL_CACHE_PRIME_TIMEOUT", "15"))


def _legacy_streams_enabled() -> bool:
    """Return True when legacy Binance websocket threads are allowed."""

    flag = (os.getenv("DISABLE_LEGACY_BINANCE_WS", "1") or "1").strip().lower()
    return flag not in {"1", "true", "yes", "on"}


def _prepare_signal_cache_params() -> tuple[float, float, int, float, float]:
    """Resolve environment-tunable parameters for the signal cache."""

    refresh_interval = float(
        os.getenv("SIGNAL_REFRESH_INTERVAL", str(SIGNAL_REFRESH_INTERVAL))
    )
    stale_mult = float(os.getenv("SIGNAL_STALE_AFTER", str(SIGNAL_STALE_MULT)))
    max_conc = int(os.getenv("MAX_CONCURRENT_FETCHES", str(MAX_CONCURRENT_FETCHES)))
    scan_interval = float(os.getenv("SCAN_INTERVAL", str(SCAN_INTERVAL)))

    tuned_stale_mult, tuned_max_conc, tuned_scan_interval, estimated_cycle = (
        _auto_tune_signal_cache_params(
            runtime_settings.max_symbols,
            refresh_interval,
            stale_mult,
            max_conc,
            scan_interval,
        )
    )

    return (
        refresh_interval,
        tuned_stale_mult,
        tuned_max_conc,
        tuned_scan_interval,
        estimated_cycle,
    )


# Interval between news fetches (in seconds)
NEWS_INTERVAL = 3600
NEWS_MONITOR_INTERVAL = float(os.getenv("NEWS_MONITOR_INTERVAL", "3600"))
NEWS_ALERT_THRESHOLD = float(os.getenv("NEWS_ALERT_THRESHOLD", "0.6"))
NEWS_HALT_THRESHOLD = float(os.getenv("NEWS_HALT_THRESHOLD", "0.9"))
NEWS_MONITOR_STATE_PATH = os.getenv("NEWS_MONITOR_STATE_PATH", "news_monitor_state.json")
RUN_DASHBOARD = os.getenv("RUN_DASHBOARD", "0") == "1"
USE_RL_POSITION_SIZER = False
rl_sizer = RLPositionSizer() if USE_RL_POSITION_SIZER else None
GROQ_CLIENT_READY = warm_up_groq_client()
if GROQ_CLIENT_READY:
    logger.info("Groq client initialised for LLM workflows")
else:
    logger.info("Groq client unavailable; LLM workflows will be disabled")

# Time-to-live for alternative data fetches
ALT_DATA_REFRESH_INTERVAL = float(os.getenv("ALT_DATA_REFRESH_INTERVAL", "300"))

# Explicit USD bounds for trade sizing (confidence-weighted between 400-500 USD)
MIN_TRADE_USD = 400.0
MAX_TRADE_USD = 500.0
VOLATILITY_SPIKE_THRESHOLD = float(os.getenv("VOLATILITY_SPIKE_THRESHOLD", "0.9"))

DEFAULT_MACRO_SENTIMENT = {
    "bias": "neutral",
    "confidence": 5.0,
    "summary": "Macro context unavailable.",
}
DEFAULT_DECISION_SENTIMENT = {
    "bias": DEFAULT_MACRO_SENTIMENT["bias"],
    "score": DEFAULT_MACRO_SENTIMENT["confidence"],
    "confidence": DEFAULT_MACRO_SENTIMENT["confidence"],
    "summary": DEFAULT_MACRO_SENTIMENT["summary"],
}
MACRO_CONTEXT_STALE_AFTER = float(os.getenv("MACRO_CONTEXT_STALE_AFTER", "900"))

logger.info(
    "Paths: LOG_FILE=%s TRADE_HISTORY=%s ACTIVE_TRADES=%s REJECTED_TRADES=%s LEARNING_LOG=%s",
    LOG_FILE,
    TRADE_HISTORY_FILE,
    ACTIVE_TRADES_FILE,
    REJECTED_TRADES_FILE,
    TRADE_LEARNING_LOG_FILE,
)

logger.info("News halt mode = %s", os.getenv("NEWS_HALT_MODE", "soft"))


def _default_macro_payload() -> Dict[str, Any]:
    payload = {
        "btc_dominance": None,
        "fear_greed": None,
        "sentiment": dict(DEFAULT_MACRO_SENTIMENT),
        "stale": True,
    }
    return payload


def _sanitize_float(value: Any, previous: Any) -> Tuple[Optional[float], bool]:
    for candidate_value, fallback_flag in ((value, False), (previous, True)):
        if candidate_value is None:
            continue
        try:
            candidate = float(candidate_value)
        except (TypeError, ValueError):
            logger.debug(
                "Invalid macro float candidate: %s", candidate_value, exc_info=True
            )
            continue
        if math.isfinite(candidate):
            return candidate, fallback_flag
        logger.debug(
            "Macro float candidate not finite: %s", candidate_value, exc_info=True
        )

    return None, True


def _sanitize_int(value: Any, previous: Any) -> Tuple[Optional[int], bool]:
    for candidate_value, fallback_flag in ((value, False), (previous, True)):
        if candidate_value is None:
            continue
        try:
            candidate = int(float(candidate_value))
        except (TypeError, ValueError):
            logger.debug(
                "Invalid macro int candidate: %s", candidate_value, exc_info=True
            )
            continue
        if 0 <= candidate <= 100:
            return candidate, fallback_flag
        logger.debug(
            "Macro int candidate out of expected range (0-100): %s", candidate_value
        )
    return None, True


def _sanitize_sentiment(value: Any, previous: Any) -> Tuple[Dict[str, Any], bool]:
    fallback_used = False
    sentiment = dict(DEFAULT_MACRO_SENTIMENT)
    source: Optional[Mapping[str, Any]]
    if isinstance(value, Mapping) and value:
        source = value
    elif isinstance(previous, Mapping) and previous:
        source = previous
        fallback_used = True
    else:
        source = None
        fallback_used = True

    if source is not None:
        bias = source.get("bias")
        if isinstance(bias, str) and bias.strip():
            sentiment["bias"] = bias.strip().lower()
        else:
            fallback_used = True

        confidence = source.get("confidence")
        try:
            conf_value = float(confidence)
        except (TypeError, ValueError):
            fallback_used = True
        else:
            if math.isfinite(conf_value):
                sentiment["confidence"] = conf_value
            else:
                fallback_used = True

        summary = source.get("summary")
        if isinstance(summary, str) and summary.strip():
            sentiment["summary"] = summary.strip()

    return sentiment, fallback_used


def _run_async_task(coro_factory):
    """Safely execute an async coroutine factory from the synchronous agent loop."""

    try:
        return asyncio.run(coro_factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()


def calculate_dynamic_trade_size(confidence: float, ml_prob: float, score: float) -> float:
    """Return the notional size bounded between 400-500 USDT.

    The trade size is primarily determined by the model confidence and
    is constrained so that only a single trade sized between 400 and
    500 USDT can be active at any time.  ``ml_prob`` and ``score`` are
    accepted for signature compatibility and may be incorporated into
    future refinements, but the current policy keeps the mapping
    confidence-driven.
    """

    _ = (ml_prob, score)  # retained for forwards compatibility

    if confidence >= 8.5:
        size = MAX_TRADE_USD
    elif confidence >= 6.5:
        size = (MIN_TRADE_USD + MAX_TRADE_USD) / 2.0
    else:
        size = MIN_TRADE_USD

    return max(MIN_TRADE_USD, min(MAX_TRADE_USD, float(size)))


def macro_filter_decision(
    btc_dom: Optional[float], fear_greed: Optional[int], bias: str, conf: float
):
    """Return macro gating decision.

    Parameters
    ----------
    btc_dom : float | None
        Bitcoin dominance percentage, when available.
    fear_greed : int | None
        Fear & Greed index value, when available.
    bias : str
        Macro sentiment bias.
    conf : float
        Confidence in the sentiment assessment.

    Returns
    -------
    tuple
        ``(skip_all, skip_alt, reasons)`` where ``reasons`` is a list of
        explanatory strings.
    """
    skip_all = False
    skip_alt = False
    reasons: list[str] = []

    logger.info(
        "macro_filter: using BTC_DOM=%s FNG=%s bias=%s conf=%.2f",
        f"{btc_dom:.2f}" if btc_dom is not None else "None",
        fear_greed if fear_greed is not None else "None",
        bias,
        conf,
    )

    if btc_dom in (50, 50.0):
        logger.warning("macro_filter: detected default BTC dominance 50 – unexpected")
    if fear_greed in (50, 50.0):
        logger.warning("macro_filter: detected default FNG=50 – unexpected")

    if fear_greed is None:
        logger.info("macro_filter: FNG unavailable (None) – skipping FNG veto")
    else:
        if fear_greed < 10:
            skip_all = True
            reasons.append("extreme fear (FG < 10)")
        elif bias == "bearish" and conf >= 8.0 and fear_greed < 15:
            skip_all = True
            reasons.append("very bearish sentiment with deep fear")

    alt_risk_score = 0
    if btc_dom is None:
        logger.info("macro_filter: BTC_DOM unavailable (None) – skipping dominance veto")
    elif btc_dom > 60.0:
        alt_risk_score += 1
    if fear_greed is not None and fear_greed < 20:
        alt_risk_score += 1
    if bias == "bearish" and conf >= 6.0:
        alt_risk_score += 1

    if not skip_all and alt_risk_score >= 2:
        skip_alt = True
        if btc_dom > 60.0:
            reasons.append("very high BTC dominance")
        if fear_greed < 20:
            reasons.append("low Fear & Greed")
        if bias == "bearish" and conf >= 6.0:
            reasons.append("bearish sentiment")

    logger.info(
        "macro_filter decision: skip_all=%s skip_alt=%s reasons=%s",
        skip_all,
        skip_alt,
        ", ".join(reasons) if reasons else "none",
    )
    return skip_all, skip_alt, reasons


def dynamic_max_active_trades(
    fear_greed: int | None, bias: str, volatility: float | None
) -> int:
    """Return the hard limit of one active trade regardless of conditions."""

    _ = (fear_greed, bias, volatility)  # inputs retained for compatibility
    return MAX_ACTIVE_TRADES


def auto_run_news() -> None:
    """Background thread that periodically fetches news."""
    while True:
        logger.info("Running scheduled news fetcher...")
        run_news_fetcher()
        jitter = random.uniform(-0.05 * NEWS_INTERVAL, 0.05 * NEWS_INTERVAL)
        time.sleep(max(0.0, NEWS_INTERVAL + jitter))


def run_streamlit() -> None:
    """Launch the Streamlit dashboard using Streamlit's Python API."""
    port = os.environ.get("PORT", "10000")
    import sys
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        "dashboard.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    stcli.main()


def run_agent_loop() -> None:
    """Main loop that scans the market, evaluates signals and opens trades."""
    logger.info("Spot AI Super Agent running in paper trading mode...")
    # start news and dashboard threads
    def _handle_news_alert(alert) -> None:
        try:
            reason = str(getattr(alert, "reason", "")) or "LLM news monitor alert"
            logger.error(
                "LLM news monitor flagged critical news: %s (sensitivity=%.2f, halt=%s)",
                reason,
                float(getattr(alert, "sensitivity", 0.0)),
                bool(getattr(alert, "halt_trading", False)),
            )
        except Exception:
            logger.error("LLM news monitor emitted alert", exc_info=True)

    monitor = start_background_news_monitor(
        interval=NEWS_MONITOR_INTERVAL,
        alert_threshold=NEWS_ALERT_THRESHOLD,
        halt_threshold=NEWS_HALT_THRESHOLD,
        status_path=NEWS_MONITOR_STATE_PATH,
        alert_callback=_handle_news_alert,
    )
    if RUN_DASHBOARD:
        threading.Thread(target=run_streamlit, daemon=True).start()
    # Ensure symbol_scores.json exists
    try:
        if not os.path.exists(SYMBOL_SCORES_FILE):
            save_symbol_scores({})
            logger.info("Initialized empty symbol_scores.json")
    except Exception:
        logger.debug("Failed to initialise symbol_scores store", exc_info=True)
    (
        refresh_interval,
        stale_mult,
        max_conc,
        scan_interval,
        estimated_cycle,
    ) = _prepare_signal_cache_params()

    signal_cache = get_active_cache()
    if signal_cache is None:
        signal_cache = RealTimeSignalCache(
            price_fetcher=get_price_data_async,
            evaluator=evaluator_for_cache,
            refresh_interval=refresh_interval,
            stale_after=refresh_interval * stale_mult,
            max_concurrency=max_conc,
            use_streams=ENABLE_WS_BRIDGE,
        )
        set_active_cache(signal_cache)
        logger.info("RTSC bootstrap: created new global cache instance.")
    else:
        logger.info("RTSC bootstrap: reusing existing global cache instance.")
        signal_cache.use_streams = ENABLE_WS_BRIDGE

    signal_cache.start()
    _mark_scan_heartbeat()
    debounce_overrides = {
        symbol: override.debounce_ms
        for symbol, override in runtime_settings.symbol_overrides.items()
        if override.debounce_ms is not None
    }
    refresh_overrides = {
        symbol: override.refresh_interval
        for symbol, override in runtime_settings.symbol_overrides.items()
        if override.refresh_interval is not None
    }
    signal_cache.configure_runtime(
        default_debounce_ms=runtime_settings.debounce_ms,
        debounce_overrides=debounce_overrides,
        refresh_overrides=refresh_overrides,
        circuit_breaker_threshold=runtime_settings.circuit_breaker_threshold,
        circuit_breaker_window=runtime_settings.circuit_breaker_window,
    )
    if get_active_cache() is not signal_cache:
        set_active_cache(signal_cache)
    logger.info(
        "Signal cache params: interval=%.2fs, stale_after=%.2fs, max_concurrency=%d, scan_interval=%.2fs (cycle≈%.2fs)",
        refresh_interval,
        refresh_interval * stale_mult,
        max_conc,
        scan_interval,
        estimated_cycle,
    )
    boot_syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    try:
        symbols = get_top_symbols(limit=runtime_settings.max_symbols)
    except Exception as warmup_exc:
        logger.error("Initial symbol fetch failed for warm-up: %s", warmup_exc, exc_info=True)
        symbols = []
    symbols = [sym for sym in symbols if isinstance(sym, str) and sym]
    warmup_symbols = sorted({sym.upper() for sym in [*symbols, *boot_syms]})
    REQUIRED_MIN_BARS = int(os.getenv("RTSC_REQUIRED_MIN_BARS", "220"))
    WARMUP_TIMEOUT_SEC = int(os.getenv("RTSC_WARMUP_TIMEOUT_SEC", "120"))
    deadline = time.time() + WARMUP_TIMEOUT_SEC
    pending_backfill: set[str] = set(warmup_symbols)
    while True:
        if pending_backfill:
            for symbol in list(pending_backfill):
                try:
                    success = signal_cache.force_rest_backfill(symbol)
                except Exception as exc:
                    logger.error("Warmup backfill failed for %s: %s", symbol, exc)
                    success = False
                if success:
                    pending_backfill.discard(symbol)
                else:
                    # retry again on the next loop after a short pause
                    time.sleep(0.05)
        missing = {
            s for s in warmup_symbols if signal_cache.bars_len(s) < REQUIRED_MIN_BARS
        }
        if not missing:
            logger.info("RTSC warm-up complete for %d symbols", len(warmup_symbols))
            break
        pending_backfill |= missing
        if time.time() > deadline:
            preview = ",".join(sorted(missing)[:10])
            logger.error("RTSC warm-up timed out; missing=%s", preview or "<all>")
            signal_cache.mark_ready(False)
            for symbol in sorted(missing):
                dispatch_schedule_refresh(signal_cache, symbol)
            break
        time.sleep(0.5)
    for sym in boot_syms:
        dispatch_schedule_refresh(signal_cache, sym)

    time.sleep(1.0)
    signal_cache.flush_pending()

    guard_stop = threading.Event()
    scan_trigger = threading.Event()
    attach_scan_event(scan_trigger)

    async def on_market_event(symbol: str, kind: str) -> None:
        if not symbol:
            return
        try:
            await signal_cache.schedule_refresh(symbol)
        except Exception:
            logger.debug(
                "WS market event refresh failed for %s", symbol, exc_info=True
            )
        if kind == "kline_close":
            try:
                dispatch_schedule_refresh(signal_cache, symbol)
            except Exception:
                logger.debug(
                    "WS market event dispatch failed for %s", symbol, exc_info=True
                )
        try:
            scan_trigger.set()
        except Exception:
            logger.debug(
                "WS market event trigger failed for %s", symbol, exc_info=True
            )
        try:
            await notify_scan(f"{symbol}:{kind}")
        except Exception:
            logger.debug(
                "WS market event notify failed for %s", symbol, exc_info=True
            )

    ws_state_lock = threading.Lock()
    ws_state = {"last_mono": time.monotonic(), "stale": False}
    closed_bar_lock = threading.Lock()
    last_closed_bars: Dict[str, int] = {}

    def _mark_ws_activity() -> None:
        with ws_state_lock:
            ws_state["last_mono"] = time.monotonic()
            ws_state["stale"] = False

    def _note_ws_stale(gap: float) -> None:
        with ws_state_lock:
            ws_state["stale"] = True

    def _should_emit_close(symbol: str, payload: Dict[str, Any]) -> bool:
        close_time_raw = payload.get("T") or payload.get("t")
        try:
            close_time = int(close_time_raw)
        except (TypeError, ValueError):
            close_time = None
        if close_time is None:
            return True
        with closed_bar_lock:
            last = last_closed_bars.get(symbol)
            if last == close_time:
                return False
            last_closed_bars[symbol] = close_time
        return True

    global _ws_bridge
    ws_bridge: Optional[WSPriceBridge]

    if ENABLE_WS_BRIDGE:
        def _handle_ws_kline(symbol: str, frame: str, payload: Dict[str, Any]) -> None:
            try:
                _mark_ws_activity()
                process_live_kline(symbol, frame, payload)
                if payload.get("x") and _should_emit_close(symbol, payload):
                    close_ts = payload.get("T") or payload.get("t")
                    log_event(
                        logger,
                        "bar_close",
                        symbol=symbol,
                        interval=frame,
                        close_time=close_ts,
                        close_price=payload.get("c"),
                    )
                    signal_cache.on_ws_bar_close(symbol, close_ts)
                    kicked = [symbol]
                    dispatch_schedule_refresh(signal_cache, symbol)
                    scan_trigger.set()
                    logging.getLogger(__name__).debug(
                        "[AGENT] kicked %d: %s",
                        len(kicked),
                        kicked,
                    )
            except Exception:
                logger.debug("WS kline handler error for %s", symbol, exc_info=True)

        def _handle_ws_ticker(symbol: str, payload: Dict[str, Any]) -> None:
            try:
                _mark_ws_activity()
                process_live_ticker(symbol, payload)
                dispatch_schedule_refresh(signal_cache, symbol)
                scan_trigger.set()
            except Exception:
                logger.debug("WS ticker handler error for %s", symbol, exc_info=True)

        def _handle_book_ticker(symbol: str, payload: Dict[str, Any]) -> None:
            try:
                _mark_ws_activity()
                process_book_ticker(symbol, payload)
            except Exception:
                logger.debug("WS book ticker handler error for %s", symbol, exc_info=True)

        try:
            existing_bridge = _ws_bridge if isinstance(_ws_bridge, WSPriceBridge) else None
            if existing_bridge is not None:
                try:
                    existing_bridge.stop()
                except Exception:
                    logger.debug("WS bootstrap bridge stop failed", exc_info=True)
                _ws_bridge = None
            book_callback = _handle_book_ticker if runtime_settings.use_ws_book_ticker else None
            ws_bridge = WSPriceBridge(
                symbols=[],
                kline_interval=os.getenv("WS_BRIDGE_INTERVAL", "1m"),
                on_kline=_handle_ws_kline,
                on_ticker=_handle_ws_ticker,
                on_book_ticker=book_callback,
                on_market_event=on_market_event,
                on_stale=_note_ws_stale,
                heartbeat_timeout=runtime_settings.max_ws_gap_before_rest,
                server_time_sync_interval=runtime_settings.server_time_sync_interval,
            )
            _ws_bridge = ws_bridge
            ws_bridge.start()
            try:
                signal_cache.enable_streams(ws_bridge)
                logger.info("RTSC: enable_streams(ws_bridge) wired successfully.")
            except Exception as e:
                logger.exception(f"RTSC wiring failed: {e}")
        except Exception:
            logger.warning("Failed to initialise WebSocket price bridge", exc_info=True)
            ws_bridge = None
            signal_cache.disable_streams()
    else:
        ws_bridge = None
    user_stream_bridge: Optional[UserDataStreamBridge]
    if ENABLE_USER_STREAM and _legacy_streams_enabled():
        def _handle_user_stream(payload: Dict[str, Any]) -> None:
            try:
                process_user_stream_event(payload)
            except Exception:
                logger.debug("User stream handler error", exc_info=True)

        try:
            user_stream_bridge = UserDataStreamBridge(on_event=_handle_user_stream)
            user_stream_bridge.start()
        except Exception:
            logger.warning("Failed to initialise Binance user data stream", exc_info=True)
            user_stream_bridge = None
    else:
        user_stream_bridge = None
        if ENABLE_USER_STREAM:
            logger.info(
                "Binance user data stream disabled (WSPriceBridge is authoritative)."
                " Set DISABLE_LEGACY_BINANCE_WS=0 to re-enable at your own risk."
            )

    state = CentralState()

    def _get_state_section(section: str) -> Dict[str, Any]:
        """Return a section snapshot without assuming ``state`` has the API."""

        getter = getattr(state, "get_section", None)
        if not callable(getter):
            logger.debug(
                "State container missing get_section; returning default for section '%s'",
                section,
            )
            return {"data": None, "timestamp": 0.0}

        try:
            snapshot = getter(section)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "State get_section('%s') failed: %s",
                section,
                exc,
                exc_info=True,
            )
            return {"data": None, "timestamp": 0.0}

        if not isinstance(snapshot, dict):
            return {"data": None, "timestamp": 0.0}

        return {
            "data": snapshot.get("data"),
            "timestamp": float(snapshot.get("timestamp", 0.0)),
        }
    worker_pools = WorkerPools()
    _ensure_periodic_tick()
    scan_lock = threading.Lock()
    manual_cache_primes: dict[str, float] = {}
    guard_interval = float(os.getenv("GUARD_INTERVAL", "0.75"))
    guard_interval = max(0.5, min(1.0, guard_interval))
    watchdog_idle_threshold = float(
        os.getenv(
            "SCAN_WATCHDOG_THRESHOLD",
            str(max(20.0, scan_interval * 4.0)),
        )
    )
    watchdog_idle_threshold = max(0.0, watchdog_idle_threshold)
    watchdog_cooldown = max(guard_interval, scan_interval)
    watchdog_last_trigger = 0.0
    macro_task = ScheduledTask("macro", min_interval=60.0, max_interval=300.0)
    news_task = ScheduledTask("news", min_interval=300.0, max_interval=900.0)
    macro_task.next_run = 0.0
    news_task.next_run = 0.0
    last_rest_backfill = 0.0

    def refresh_macro_state() -> None:
        previous_snapshot = _get_state_section("macro")
        previous_payload = previous_snapshot.get("data")
        if not isinstance(previous_payload, dict):
            previous_payload = {}

        btc_dom_raw = None
        btc_dom_ts = None
        fear_greed_raw = None
        fear_greed_ts = None
        sentiment_raw: Any = None

        try:
            btc_snapshot = get_btc_dominance_cached()
            if btc_snapshot is not None:
                btc_dom_raw = btc_snapshot.value
                btc_dom_ts = btc_snapshot.ts
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("BTC dominance fetch failed: %s", exc, exc_info=True)

        try:
            fg_snapshot = get_fear_greed_cached()
            if fg_snapshot is not None:
                fear_greed_raw = fg_snapshot.value
                fear_greed_ts = fg_snapshot.ts
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Fear & Greed fetch failed: %s", exc, exc_info=True)

        try:
            sentiment_raw = get_macro_sentiment()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Macro sentiment fetch failed: %s", exc, exc_info=True)

        btc_dom, btc_fallback = _sanitize_float(
            btc_dom_raw, previous_payload.get("btc_dominance")
        )
        fear_greed, fg_fallback = _sanitize_int(
            fear_greed_raw, previous_payload.get("fear_greed")
        )
        sentiment, sentiment_fallback = _sanitize_sentiment(
            sentiment_raw, previous_payload.get("sentiment")
        )

        payload = {
            "btc_dominance": btc_dom,
            "fear_greed": fear_greed,
            "sentiment": sentiment,
            "stale": bool(btc_fallback or fg_fallback or sentiment_fallback),
            "btc_timestamp": btc_dom_ts,
            "fear_greed_timestamp": fear_greed_ts,
        }

        state.update_section("macro", payload)
        if payload["stale"]:
            btc_text = f"{btc_dom:.2f}" if btc_dom is not None else "n/a"
            fg_text = f"{fear_greed}" if fear_greed is not None else "n/a"
            logger.debug(
                "Macro refresh using cached values (btc=%s fg=%s bias=%s)",
                btc_text,
                fg_text,
                sentiment.get("bias", "neutral"),
            )
        scan_trigger.set()

    def refresh_news_state() -> None:
        try:
            trigger_news_refresh()
            payload = get_news_cache()
            events = list(payload.get("items", []))
            assessment = None
            next_event = minutes_until_next_event(events)
            if events and payload.get("ok", False) and payload.get("source") not in {"disabled", "neutral"}:
                try:
                    assessment = _run_async_task(lambda: analyze_news_with_llm_async(events))
                except Exception as analysis_exc:
                    logger.debug("LLM news analysis failed: %s", analysis_exc, exc_info=True)
            state.merge_section(
                "news",
                {
                    "events": events,
                    "assessment": assessment,
                    "next_event_minutes": next_event,
                    "source": payload.get("source"),
                    "error": payload.get("error"),
                },
            )
            scan_trigger.set()
        except Exception as exc:
            logger.debug("News refresh task failed: %s", exc, exc_info=True)

    def handle_legacy_market_event(event: dict) -> None:
        try:
            event_type = str(event.get("type", ""))
            symbol = str(event.get("symbol", "")).upper()
            ts = float(event.get("timestamp", time.time()))
            price = event.get("price")
            if event_type in {"trade", "kline", "rest_price"} and symbol:
                try:
                    price_value = float(price) if price is not None else None
                except Exception:
                    price_value = None
                if price_value is not None:
                    state.update_price(symbol, price_value, timestamp=ts)
            if event_type == "kline" and symbol:
                payload = event.get("payload") or {}
                if isinstance(payload, dict):
                    state.update_kline(symbol, payload, timestamp=ts)
            if event_type == "order_update":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    state.append_order_update(payload)
            if event_type in {"trade", "kline", "order_update", "rest_price"}:
                scan_trigger.set()
        except Exception:
            logger.exception("Failed to process market event: %s", event)

    market_stream: Optional[BinanceEventStream]
    if _legacy_streams_enabled():
        market_stream = BinanceEventStream(
            symbols=["BTCUSDT"],
            on_event=handle_legacy_market_event,
            max_queue=runtime_settings.max_queue,
        )
        market_stream.start()
    else:
        market_stream = None
        logger.info(
            "Legacy Binance market stream disabled (WSPriceBridge provides price data)."
            " Set DISABLE_LEGACY_BINANCE_WS=0 to re-enable at your own risk."
        )

    def guard_loop() -> None:
        nonlocal last_rest_backfill
        nonlocal watchdog_last_trigger
        while not guard_stop.is_set():
            now = time.time()
            if macro_task.due(now):
                worker_pools.submit_io(refresh_macro_state)
                macro_task.schedule_next(now)
            if news_task.due(now):
                worker_pools.submit_io(refresh_news_state)
                news_task.schedule_next(now)
            monitor_instance = get_news_monitor()
            if monitor_instance is not None:
                try:
                    monitor_state = monitor_instance.get_latest_assessment()
                    if monitor_state:
                        state.merge_section("news", {"monitor": monitor_state})
                        if monitor_state.get("alert_triggered"):
                            scan_trigger.set()
                except Exception as monitor_exc:
                    logger.debug("News monitor state unavailable: %s", monitor_exc, exc_info=True)
            if market_stream is not None:
                market_stream.ensure_alive()
                if not market_stream.is_connected():
                    tracked = list(state.tracked_symbols())
                    if not tracked:
                        tracked = ["BTCUSDT"]
                    market_stream.poll_rest(tracked)
            if ENABLE_WS_BRIDGE and REST_BACKFILL_ENABLED:
                now_mono = time.monotonic()
                with ws_state_lock:
                    last_mono = ws_state.get("last_mono", now_mono)
                    stale_flag = ws_state["stale"]
                    ws_state["stale"] = False
                gap = now_mono - last_mono
                if gap >= runtime_settings.max_ws_gap_before_rest or stale_flag:
                    if now - last_rest_backfill >= runtime_settings.max_ws_gap_before_rest:
                        tracked_symbols = tuple(signal_cache.symbols())
                        log_event(
                            logger,
                            "ws_gap_fallback",
                            gap_seconds=gap,
                            stale_flag=stale_flag,
                            symbols=len(tracked_symbols),
                        )
                        record_metric(
                            "ws_gap_seconds",
                            gap,
                        )
                        kicked = list(tracked_symbols)
                        for sym in kicked:
                            dispatch_schedule_refresh(signal_cache, sym)
                        logging.getLogger(__name__).debug(
                            "[AGENT] kicked %d: %s",
                            len(kicked),
                            kicked,
                        )
                        try:
                            scan_trigger.set()
                        except Exception:
                            logger.debug(
                                "Failed to set scan trigger after WS fallback",
                                exc_info=True,
                            )
                        try:
                            _trigger_scan("ws_gap_fallback")
                        except Exception:
                            logger.debug(
                                "Failed to notify scan after WS fallback",
                                exc_info=True,
                            )
                        last_rest_backfill = now
            if watchdog_idle_threshold > 0.0:
                idle = _scan_idle_seconds(now)
                if idle > watchdog_idle_threshold and (
                    now - watchdog_last_trigger
                ) >= watchdog_cooldown:
                    watchdog_last_trigger = now
                    logger.warning(
                        "Scan watchdog detected %.1fs of inactivity; forcing scan.",
                        idle,
                    )
                    try:
                        _trigger_scan("watchdog_idle")
                    except Exception:
                        logger.debug("Failed to trigger watchdog scan", exc_info=True)
                    scan_trigger.set()
            guard_stop.wait(guard_interval)

    threading.Thread(target=guard_loop, daemon=True).start()
    worker_pools.submit_io(refresh_macro_state)
    worker_pools.submit_io(refresh_news_state)
    last_scan_time = 0.0
    btc_bars_len_last: Optional[int] = None
    while True:
        triggered = scan_trigger.wait(timeout=guard_interval)
        if not triggered:
            continue
        now = time.time()
        remaining = (last_scan_time + scan_interval) - now
        if remaining > 0:
            # Ensure we don't trigger scans more frequently than configured
            time.sleep(remaining)
            continue
        if not scan_lock.acquire(blocking=False):
            time.sleep(0.01)
            continue
        try:
            now = time.time()
            if (now - last_scan_time) < scan_interval:
                # Another thread completed a scan recently while we were waiting
                continue
            scan_trigger.clear()
            last_scan_time = now
            _mark_scan_heartbeat(now)
            logger.debug("=== Scan @ %s ===", time.strftime('%Y-%m-%d %H:%M:%S'))
            try:
                btc_bars_len = signal_cache.bars_len("BTCUSDT")
            except Exception as cache_exc:
                logger.warning(
                    "Failed to read BTCUSDT cached bars length: %s",
                    cache_exc,
                )
            else:
                if btc_bars_len_last is None:
                    logger.debug("BTCUSDT cached bars: %d", btc_bars_len)
                else:
                    delta = btc_bars_len - btc_bars_len_last
                    logger.debug(
                        "BTCUSDT cached bars: %d (Δ %+d)",
                        btc_bars_len,
                        delta,
                    )
                btc_bars_len_last = btc_bars_len
            # Check drawdown guard
            if is_trading_blocked():
                logger.warning("Drawdown limit reached. Skipping trading for today.")
                continue
            perf = compute_performance_metrics()
            if perf.get("max_drawdown", 0) < -0.25:
                logger.warning("Max drawdown exceeded 25%. Halting trading.")
                continue
            macro_snapshot = _get_state_section("macro")
            macro_payload_raw = macro_snapshot.get("data")
            macro_timestamp = float(macro_snapshot.get("timestamp", 0.0))
            macro_age = time.time() - macro_timestamp if macro_timestamp else float("inf")
            macro_payload = (
                dict(macro_payload_raw) if isinstance(macro_payload_raw, dict) else {}
            )
            macro_missing = not macro_payload
            macro_stale = False
            if macro_payload:
                if bool(macro_payload.get("stale")):
                    macro_stale = True
                elif macro_age > MACRO_CONTEXT_STALE_AFTER:
                    macro_stale = True
            if macro_missing:
                logger.debug("Macro context unavailable; using empty snapshot for gating.")
                macro_payload = _default_macro_payload()
            elif macro_stale:
                logger.debug(
                    "Macro context stale (age=%.0fs). Retaining last known macro values.",
                    macro_age,
                )
                macro_payload["stale"] = True
            btc_d_raw = macro_payload.get("btc_dominance")
            btc_d: Optional[float]
            if btc_d_raw is None:
                btc_d = None
            else:
                try:
                    candidate = float(btc_d_raw)
                    btc_d = candidate if math.isfinite(candidate) else None
                except (TypeError, ValueError):
                    btc_d = None

            fg_raw = macro_payload.get("fear_greed")
            if fg_raw is None:
                fg: Optional[int] = None
            else:
                try:
                    fg = int(float(fg_raw))
                except (TypeError, ValueError):
                    fg = None
            sentiment_payload = macro_payload.get("sentiment") or {}
            if not isinstance(sentiment_payload, dict):
                sentiment_payload = dict(DEFAULT_MACRO_SENTIMENT)
            # Extract sentiment bias and confidence safely.  We intentionally build a
            # dedicated sentiment mapping for downstream consumers so that missing
            # sentiment data never raises NameError inside the trade preparation
            # pipeline (see prepare_trade_decision()).
            try:
                sentiment_confidence = float(
                    sentiment_payload.get("confidence", DEFAULT_MACRO_SENTIMENT["confidence"])
                )
            except Exception:
                sentiment_confidence = DEFAULT_MACRO_SENTIMENT["confidence"]
            sentiment_bias = str(
                sentiment_payload.get("bias", DEFAULT_MACRO_SENTIMENT["bias"])
            ).strip().lower()
            if not sentiment_bias:
                sentiment_bias = DEFAULT_MACRO_SENTIMENT["bias"]
            sentiment_summary = sentiment_payload.get("summary")
            if not isinstance(sentiment_summary, str) or not sentiment_summary.strip():
                sentiment_summary = DEFAULT_MACRO_SENTIMENT["summary"]
            sentiment = dict(DEFAULT_DECISION_SENTIMENT)
            sentiment["bias"] = sentiment_bias
            sentiment["score"] = sentiment_confidence
            sentiment["confidence"] = sentiment_confidence
            sentiment["summary"] = sentiment_summary.strip()
            # Convert BTC dominance and Fear & Greed to numeric values
            try:
                btc_d = float(btc_d)
            except Exception:
                btc_d = 0.0
            try:
                fg = int(fg)
            except Exception:
                fg = 0
            btc_log = f"{btc_d:.2f}%" if btc_d is not None else "n/a"
            fg_log = f"{fg}" if fg is not None else "n/a"
            logger.info(
                "BTC Dominance: %s | Fear & Greed: %s | Sentiment: %s (Confidence: %s)",
                btc_log,
                fg_log,
                sentiment_bias,
                sentiment_confidence,
            )
            skip_all, skip_alt, macro_reasons = macro_filter_decision(
                btc_d, fg, sentiment_bias, sentiment_confidence
            )
            news_snapshot = _get_state_section("news")
            news_data = news_snapshot.get("data") or {}
            monitor_state = news_data.get("monitor") if isinstance(news_data, dict) else None
            if monitor_state:
                reason = str(monitor_state.get("reason", "LLM requested trading halt"))
                if monitor_state.get("halt_trading"):
                    logger.error("Skipping scan due to LLM news halt signal: %s", reason)
                    continue
                if monitor_state.get("warning_only"):
                    logger.warning("LLM news monitor warning: %s", reason)
            if monitor_state and monitor_state.get("alert_triggered"):
                macro_reasons.append("LLM news alert")
            signal_cache.update_context(sentiment_bias=sentiment_bias)
            if skip_all:
                reason_text = " + ".join(macro_reasons) if macro_reasons else "unfavorable conditions"
                logger.warning("Market unfavorable (%s). Skipping scan.", reason_text)
                continue
            btc_vol = float("nan")
            btc_trend_bias = "flat"
            try:
                btc_df = asyncio.run(get_price_data_async("BTCUSDT"))
                if btc_df is not None and not btc_df.empty:
                    btc_vol = atr_percentile(
                        btc_df["high"], btc_df["low"], btc_df["close"]
                    )
                    try:
                        btc_indicators = calculate_indicators(btc_df.tail(200))
                        ema20 = btc_indicators["ema_20"].iloc[-1]
                        ema50 = btc_indicators["ema_50"].iloc[-1]
                        if math.isfinite(float(ema20)) and math.isfinite(float(ema50)):
                            if float(ema20) > float(ema50) * 1.001:
                                btc_trend_bias = "up"
                            elif float(ema20) < float(ema50) * 0.999:
                                btc_trend_bias = "down"
                            else:
                                btc_trend_bias = "flat"
                    except Exception as trend_exc:
                        logger.debug("BTC trend estimation failed: %s", trend_exc, exc_info=True)
                        btc_trend_bias = "flat"
                else:
                    btc_vol = np.nan
            except Exception as e:
                logger.warning("Could not compute BTC volatility: %s", e)
                btc_trend_bias = "flat"
            max_active_trades = dynamic_max_active_trades(
                fg, sentiment_bias, btc_vol
            )
            logger.info(
                "Max active trades dynamically set to %d (BTC vol=%.2f)",
                max_active_trades,
                btc_vol,
            )
            # If we are only skipping altcoins, filter top_symbols down to BTCUSDT
            if skip_alt:
                # We will filter top_symbols later after fetching them
                macro_reason_text = " + ".join(macro_reasons) if macro_reasons else "macro caution"
            else:
                macro_reason_text = ""

            macro_news_assessment = {"safe": True, "reason": "No events analyzed"}
            macro_news_summary = ""
            next_news_minutes = None
            if isinstance(news_data, dict):
                if monitor_state and not monitor_state.get("stale"):
                    macro_news_assessment = monitor_state
                    macro_news_summary = str(monitor_state.get("reason", ""))
                    next_news_minutes = monitor_state.get("next_event_minutes")
                else:
                    stored_assessment = news_data.get("assessment")
                    if stored_assessment:
                        macro_news_assessment = stored_assessment
                        macro_news_summary = str(stored_assessment.get("reason", ""))
                    stored_next = news_data.get("next_event_minutes")
                    if stored_next is not None:
                        next_news_minutes = stored_next
            # Load active trades and ensure only long trades remain (spot mode)
            active_trades_raw = load_active_trades()
            active_trades = []
            for t in active_trades_raw:
                if t.get("direction") == "short":
                    logger.warning(
                        "Removing non-long trade %s from active trades (spot-only mode).",
                        t.get("symbol"),
                    )
                    continue
                active_trades.append(t)
            save_active_trades(active_trades)
            # Get top symbols to scan
            try:
                top_symbols = get_top_symbols(limit=runtime_settings.max_symbols)
            except Exception as e:
                logger.error("Error fetching top symbols: %s", e, exc_info=True)
                top_symbols = []
            if not top_symbols:
                logger.warning(
                    "No symbols fetched from Binance. Check your python-binance installation and network connectivity."
                )
            # Apply macro filtering to symbols: if macro filter indicated to skip altcoins,
            # restrict the universe to BTCUSDT only.  We do this after fetching the symbols
            # to avoid unnecessary API calls during the gating step.
            tier_assignments: dict[str, str] = {}
            if skip_alt:
                # keep BTCUSDT and potentially stablecoins if you wish; here we only keep BTCUSDT
                top_symbols = [sym for sym in top_symbols if sym.upper() == "BTCUSDT"]
                if macro_reason_text:
                    logger.warning("Macro gating (%s). Scanning only BTCUSDT.", macro_reason_text)
            CORE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
            if top_symbols:
                rest = [sym for sym in top_symbols if sym.upper() not in CORE_SYMBOLS]
                tier1_list = rest[:6]
                tier2_list = rest[6:16]
                tier3_list = rest[16:26]
                for sym in tier1_list:
                    tier_assignments[sym.upper()] = "TIER1"
                for sym in tier2_list:
                    tier_assignments[sym.upper()] = "TIER2"
                for sym in tier3_list:
                    tier_assignments[sym.upper()] = "TIER3"
            set_symbol_tiers(tier_assignments)
            symbol_tier_lookup = dict(tier_assignments)
            if top_symbols and market_stream is not None:
                market_stream.set_symbols(top_symbols)
            session = get_market_session()
            if top_symbols:
                logger.info(
                    "Market scan started for %s session covering %d symbols: %s",
                    session,
                    len(top_symbols),
                    ", ".join(top_symbols),
                )
            else:
                logger.info("Market scan started for %s session but no symbols available.", session)
            potential_trades: list[dict] = []
            symbol_scores: dict[str, dict[str, float | None]] = {}
            audit_rows: Dict[str, Dict[str, Any]] = {}

            def _audit_update(symbol_key: str, **updates: Any) -> None:
                if not symbol_key:
                    return
                row = audit_rows.setdefault(symbol_key, {"symbol": symbol_key})
                row.update(updates)

            def _audit_finalize(symbol_key: str, **updates: Any) -> None:
                if not symbol_key:
                    return
                row = audit_rows.pop(symbol_key, None)
                if row is None:
                    row = {"symbol": symbol_key}
                row.update(updates)
                row.setdefault("ts", datetime.now(timezone.utc).isoformat())
                log_signal_audit(row)

            symbols_to_fetch = [
                sym for sym in top_symbols if not any(t.get("symbol") == sym for t in active_trades)
            ]
            if symbols_to_fetch:
                logger.info(
                    "Evaluating %d symbols without active positions: %s",
                    len(symbols_to_fetch),
                    ", ".join(symbols_to_fetch),
                )
            else:
                logger.info("All top symbols already have active positions; skipping fresh evaluations.")
            if ws_bridge is not None:
                ws_symbols: Set[str] = {sym.upper() for sym in top_symbols}
                for trade in active_trades:
                    sym = trade.get("symbol")
                    if isinstance(sym, str) and sym:
                        ws_symbols.add(str(sym).upper())
                ws_bridge.update_symbols(sorted(ws_symbols))
            signal_cache.update_universe(symbols_to_fetch)
            signal_cache.start()
            signal_cache.flush_pending()
            if ws_bridge is None:
                kick = 6
                kicked = list(symbols_to_fetch[:kick])
                for symbol in kicked:
                    dispatch_schedule_refresh(signal_cache, symbol)
                logging.getLogger(__name__).debug(
                    "[AGENT] kicked %d: %s",
                    len(kicked),
                    kicked,
                )
            if signal_cache.circuit_breaker_active():
                logger.warning(
                    "Signal evaluator circuit breaker active; skipping trade evaluation this cycle."
                )
                continue
            cache_miss_symbols: list[str] = []
            for symbol in symbols_to_fetch:
                symbol_key = str(symbol or "").upper()
                breakdown_data: Optional[Dict[str, Any]] = None
                signal_snapshot: Mapping[str, Any] = {}
                custom_skip_reason_key: Optional[str] = None
                custom_skip_reason_text: Optional[str] = None
                metrics_logged = False
                evaluation_started = False
                last_decision_type: Optional[str] = None
                try:
                    cached_signal = signal_cache.get(symbol)
                    if cached_signal is None:
                        cache_miss_symbols.append(symbol)
                        logger.debug("No fresh cache entry for %s yet; skipping this cycle.", symbol)
                        continue
                    price_data = cached_signal.price_data
                    if price_data is None or price_data.empty or len(price_data) < 40:
                        logger.warning("Skipping %s due to insufficient data.", symbol)
                        continue
                    tier = symbol_tier_lookup.get(symbol_key)

                    score = cached_signal.score
                    raw_score = float(score)
                    direction = cached_signal.direction
                    position_size = cached_signal.position_size
                    pattern_name = cached_signal.pattern
                    signal_snapshot = price_data.attrs.get("signal_features", {}) or {}
                    setup_type = signal_snapshot.get("setup_type")
                    record_signal_evaluated()
                    evaluation_started = True
                    breakdown_data = ensure_breakdown_fields(
                        price_data.attrs.get("decision_breakdown"),
                        required_fields=("symbol",),
                    )
                    breakdown_data["symbol"] = symbol_key
                    price_data.attrs["decision_breakdown"] = breakdown_data
                    breakdown_data["setup_type"] = breakdown_data.get("setup_type") or setup_type
                    breakdown_data["norm_score"] = raw_score
                    breakdown_data["pos_size"] = position_size
                    breakdown_data["volume_ok_for_size"] = position_size > 0
                    if breakdown_data.get("direction_raw") is None:
                        breakdown_data["direction_raw"] = direction
                    breakdown_data["direction_final"] = direction
                    try:
                        auction_state = get_auction_state(price_data)
                    except Exception as exc:
                        logger.debug(
                            "Failed to compute auction state for %s: %s",
                            symbol,
                            exc,
                            exc_info=True,
                        )
                        auction_state = "unknown"
                    volume_meta = price_data.attrs.get("volume_context") or {}
                    audit_payload = {
                        "ts": signal_snapshot.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol_key,
                        "tier": tier,
                        "session": session,
                        "auction_state": signal_snapshot.get("auction_state") or auction_state,
                        "macro_bias": sentiment_bias,
                        "fear_greed": fg,
                        "btc_dominance": btc_d,
                        "news_severity": price_data.attrs.get("news_severity", 0),
                        "is_core_symbol": symbol_key in CORE_SYMBOLS,
                        "volume_gate_reason": volume_meta.get("volume_gate_reason"),
                        "macro_skip_all": skip_all,
                        "macro_skip_alt": skip_alt,
                        "selected_for_candidate_list": False,
                        "brain_veto": False,
                        "ml_veto": False,
                        "risk_veto": False,
                        "final_trade_taken": False,
                    }
                    audit_payload.update(signal_snapshot)
                    _audit_update(symbol_key, **audit_payload)
                    _audit_update(
                        symbol_key,
                        size_bucket=position_size,
                        volume_ok_from_size=bool(position_size > 0),
                        base_direction_from_signal=direction,
                        final_direction_after_force=direction,
                        forced_long_applied=bool(breakdown_data.get("forced_long_applied")),
                        auction_breakout_veto=not breakdown_data.get("auction_guard_pass", True),
                    )
                    alt_features: dict[str, object] = {}
                    alt_adjustment = 0.0
                    try:
                        alt_bundle = get_alternative_data(
                            symbol, ttl=ALT_DATA_REFRESH_INTERVAL
                        )
                    except Exception as alt_exc:
                        logger.warning(
                            "Alternative data fetch failed for %s: %s",
                            symbol,
                            alt_exc,
                        )
                        alt_bundle = None
                    if alt_bundle is not None:
                        direction_for_alt = direction or "long"
                        alt_adjustment = alt_bundle.score_adjustment(direction_for_alt)
                        score += alt_adjustment
                        alt_features = alt_bundle.to_features(direction_for_alt)
                        enriched_snapshot = dict(signal_snapshot)
                        enriched_snapshot.update(
                            {
                                "social_sentiment_bias": alt_features.get(
                                    "social_bias", "neutral"
                                ),
                                "social_sentiment_score": alt_features.get("social_score"),
                                "social_sentiment_confidence": alt_features.get(
                                    "social_confidence"
                                ),
                                "social_posts_analyzed": alt_features.get("social_posts"),
                                "onchain_score": alt_features.get("onchain_score"),
                                "onchain_net_exchange_flow": alt_features.get(
                                    "onchain_net_flow"
                                ),
                                "onchain_whale_ratio": alt_features.get(
                                    "onchain_whale_ratio"
                                ),
                                "alternative_score_adjustment": alt_features.get(
                                    "score_adjustment"
                                ),
                            }
                        )
                        signal_snapshot = enriched_snapshot
                        price_data.attrs["signal_features"] = enriched_snapshot
                        logger.info(
                            "%s alt-data -> social=%s (score=%.2f conf=%.2f posts=%s) | on-chain=%.2f | adj=%.2f",
                            symbol,
                            alt_features.get("social_bias", "neutral"),
                            float(alt_features.get("social_score") or 0.0),
                            float(alt_features.get("social_confidence") or 0.0),
                            alt_features.get("social_posts"),
                            float(alt_features.get("onchain_score") or 0.0),
                            alt_adjustment,
                        )
                    adjusted_score = float(raw_score + (alt_adjustment or 0.0))
                    try:
                        entry_cutoff = float(signal_snapshot.get("activation_threshold", 0.0))
                        if not math.isfinite(entry_cutoff) or entry_cutoff <= 0.0:
                            raise ValueError
                    except Exception:
                        entry_cutoff = 5.0
                    breakdown_data["entry_cutoff"] = entry_cutoff
                    breakdown_data["alt_adj"] = alt_adjustment
                    breakdown_data["adjusted_score"] = adjusted_score
                    breakdown_data["alt_adj_block"] = adjusted_score < entry_cutoff
                    _audit_update(
                        symbol_key,
                        alt_adjustment=alt_adjustment,
                        adjusted_score=adjusted_score,
                        alt_adj_block=adjusted_score < entry_cutoff,
                    )
                    should_log_decision = raw_score >= entry_cutoff
                    volume_ok = position_size > 0
                    macro_state = macro_news_assessment or {}
                    macro_veto_flag = not bool(macro_state.get("safe", True))
                    news_veto_flag = bool(
                        monitor_state and bool(monitor_state.get("halt_trading"))
                    )
                    cooldown_active = False
                    _audit_update(
                        symbol_key,
                        macro_veto=macro_veto_flag,
                        news_veto=news_veto_flag,
                        cooldown_active=cooldown_active,
                    )
                    breakdown_data["macro_veto"] = macro_veto_flag
                    breakdown_data["news_veto"] = news_veto_flag
                    breakdown_data["cooldown_block"] = cooldown_active

                    decision_logged = False
                    custom_skip_reason_key = breakdown_data.get("primary_skip_reason")
                    custom_skip_reason_text = breakdown_data.get("primary_skip_text")

                    def _assign_skip_reason(
                        key: Optional[str], text: Optional[str], *, overwrite: bool = False
                    ) -> None:
                        nonlocal custom_skip_reason_key, custom_skip_reason_text
                        if not overwrite and custom_skip_reason_key:
                            return
                        custom_skip_reason_key = key
                        custom_skip_reason_text = text

                    def _compute_skip_reason() -> tuple[str, Optional[str]]:
                        nonlocal custom_skip_reason_key, custom_skip_reason_text
                        if custom_skip_reason_key:
                            return (
                                custom_skip_reason_text or custom_skip_reason_key,
                                custom_skip_reason_key,
                            )
                        if adjusted_score < entry_cutoff:
                            custom_skip_reason_text = "alt_adj below cutoff"
                            if alt_adjustment and alt_adjustment < 0:
                                custom_skip_reason_key = "alt_adj_block"
                            else:
                                custom_skip_reason_key = "low_score"
                            return custom_skip_reason_text, custom_skip_reason_key
                        if not volume_ok:
                            custom_skip_reason_key = "pos_size_zero"
                            custom_skip_reason_text = "volume gate failed"
                            return custom_skip_reason_text, custom_skip_reason_key
                        if macro_veto_flag:
                            custom_skip_reason_key = "macro_veto"
                            custom_skip_reason_text = "macro veto"
                            return custom_skip_reason_text, custom_skip_reason_key
                        if news_veto_flag:
                            custom_skip_reason_key = "news_veto"
                            custom_skip_reason_text = "news veto"
                            return custom_skip_reason_text, custom_skip_reason_key
                        if cooldown_active:
                            custom_skip_reason_key = "cooldown_block"
                            custom_skip_reason_text = "cooldown in effect"
                            return custom_skip_reason_text, custom_skip_reason_key
                        custom_skip_reason_key = "other_guard"
                        custom_skip_reason_text = "other guard"
                        return custom_skip_reason_text, custom_skip_reason_key

                    def _log_decision(decision_type: str) -> None:
                        nonlocal decision_logged
                        if decision_logged or not should_log_decision:
                            return
                        if decision_type == "skip":
                            reason_text, _ = _compute_skip_reason()
                            logger.info(
                                "[DECISION] SKIP %s | raw=%.2f alt_adj=%.2f cutoff=%.2f size=%.1f dir=%s | "
                                "volume_ok=%s macro_veto=%s news_veto=%s cooldown=%s reason=%s",
                                symbol,
                                raw_score,
                                adjusted_score,
                                entry_cutoff,
                                position_size,
                                direction,
                                volume_ok,
                                macro_veto_flag,
                                news_veto_flag,
                                cooldown_active,
                                reason_text,
                            )
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else raw_score,
                                reason_text=reason_text,
                            )
                        else:
                            logger.info(
                                "[DECISION] ENTER_TRADE %s | raw=%.2f alt_adj=%.2f cutoff=%.2f size=%.1f dir=%s",
                                symbol,
                                raw_score,
                                adjusted_score,
                                entry_cutoff,
                                position_size,
                                direction,
                            )
                            decision_summary = (
                                f"raw={raw_score:.2f} alt_adj={adjusted_score:.2f} cutoff={entry_cutoff:.2f}"
                                f" size={position_size:.1f} dir={direction}"
                            )
                            log_simple_decision_metric(
                                symbol,
                                action="enter",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else raw_score,
                                reason_text=decision_summary,
                            )
                        decision_logged = True

                    def _emit_metrics(
                        decision_type: str,
                        *,
                        reason_text: Optional[str] = None,
                        reason_key: Optional[str] = None,
                    ) -> None:
                        nonlocal metrics_logged, breakdown_data, last_decision_type
                        nonlocal custom_skip_reason_key, custom_skip_reason_text
                        last_decision_type = decision_type
                        if metrics_logged:
                            return
                        if reason_key:
                            custom_skip_reason_key = reason_key
                        if reason_text:
                            custom_skip_reason_text = reason_text
                        try:
                            structured_breakdown = ensure_breakdown_fields(
                                breakdown_data,
                                required_fields=("symbol",),
                            )
                        except Exception:
                            logger.warning(
                                "Failed to rehydrate decision breakdown for %s", symbol_key,
                                exc_info=True,
                            )
                            structured_breakdown = {
                                "symbol": symbol_key,
                            }
                        breakdown_data = structured_breakdown
                        price_data.attrs["decision_breakdown"] = structured_breakdown
                        structured_breakdown["symbol"] = symbol_key
                        structured_breakdown["decision_type"] = decision_type
                        structured_breakdown["setup_type"] = (
                            structured_breakdown.get("setup_type") or setup_type
                        )
                        structured_breakdown["norm_score"] = raw_score
                        try:
                            dyn_threshold = float(signal_snapshot.get("activation_threshold"))
                        except (TypeError, ValueError):
                            dyn_threshold = float(entry_cutoff)
                        structured_breakdown["dyn_threshold"] = dyn_threshold
                        structured_breakdown["entry_cutoff"] = entry_cutoff
                        structured_breakdown["alt_adj"] = alt_adjustment
                        structured_breakdown["adjusted_score"] = adjusted_score
                        structured_breakdown["pos_size"] = position_size
                        structured_breakdown["volume_ok_for_size"] = position_size > 0
                        structured_breakdown.setdefault("direction_raw", direction)
                        structured_breakdown["direction_final"] = direction
                        structured_breakdown["macro_veto"] = macro_veto_flag
                        structured_breakdown["news_veto"] = news_veto_flag
                        structured_breakdown["cooldown_block"] = cooldown_active
                        structured_breakdown["alt_adj_block"] = adjusted_score < entry_cutoff
                        structured_breakdown.setdefault("profile_veto", False)
                        structured_breakdown.setdefault("sr_guard_pass", True)
                        structured_breakdown.setdefault("auction_guard_pass", True)
                        structured_breakdown.setdefault("spread_gate_pass", True)
                        structured_breakdown.setdefault("obi_gate_pass", True)
                        structured_breakdown["volume_gate_pass"] = bool(
                            structured_breakdown.get("volume_gate_pass", True) and volume_ok
                        )
                        structured_breakdown.setdefault("volume_gate_reason", None)
                        structured_breakdown.setdefault("profile_min_score", structured_breakdown.get("profile_min_score"))
                        structured_breakdown.setdefault("forced_long_applied", (
                            structured_breakdown.get("direction_raw") is None and direction == "long"
                        ))
                        if decision_type == "skip":
                            if reason_text is None or reason_key is None:
                                reason_text, reason_key = _compute_skip_reason()
                            structured_breakdown["primary_skip_reason"] = reason_key
                            structured_breakdown["primary_skip_text"] = reason_text
                            if reason_key:
                                record_skip_reason(reason_key)
                            _audit_finalize(
                                symbol_key,
                                selected_for_candidate_list=False,
                                final_trade_taken=False,
                                final_skip_reason=reason_text or "",
                                brain_veto=False,
                                ml_veto=False,
                                risk_veto=False,
                            )
                        else:
                            record_trade_opened()
                            _audit_update(symbol_key, selected_for_candidate_list=False)
                        try:
                            log_decision_breakdown(
                                symbol_key,
                                signal_snapshot,
                                structured_breakdown,
                            )
                            metrics_logged = True
                        except Exception:
                            logger.warning(
                                "Failed to emit decision metrics for %s (decision=%s)",
                                symbol_key,
                                decision_type,
                                exc_info=True,
                            )
                        finally:
                            maybe_log_summary()

                    pattern_lower = (pattern_name or "").lower()
                    setup_lower = (setup_type or "").lower() if isinstance(setup_type, str) else ""
                    is_breakout_setup = False
                    if setup_lower in {"trend", "breakout"}:
                        is_breakout_setup = True
                    if pattern_lower in BREAKOUT_PATTERNS or "breakout" in pattern_lower:
                        is_breakout_setup = True
                    if auction_state == "balanced" and is_breakout_setup:
                        reason_text = (
                            "Balanced regime detected – breakout setup filtered "
                            f"(pattern={pattern_name or 'none'}, setup={setup_type or 'unknown'})"
                        )
                        logger.info("[SKIP] %s: %s", symbol, reason_text)
                        log_simple_decision_metric(
                            symbol,
                            action="skip",
                            direction=direction,
                            size_value=position_size,
                            score_value=adjusted_score if adjusted_score is not None else score,
                            reason_text=reason_text,
                        )
                        breakdown_data["auction_guard_pass"] = False
                        reason_key = "auction_guard_fail"
                        _assign_skip_reason(reason_key, reason_text, overwrite=True)
                        _log_decision("skip")
                        _emit_metrics("skip", reason_text=reason_text, reason_key=reason_key)
                        continue
                    logger.info(
                        "%s: Score=%.2f (alt_adj=%.2f), Direction=%s, Pattern=%s, PosSize=%s, AuctionState=%s (age=%.2fs)",
                        symbol,
                        score,
                        alt_adjustment,
                        direction,
                        pattern_name,
                        position_size,
                        auction_state,
                        cached_signal.age(),
                    )
                    symbol_scores[symbol] = {
                        "score": score,
                        "direction": direction,
                        "auction_state": auction_state,
                        "alternative_adjustment": alt_adjustment,
                        "onchain_score": alt_features.get("onchain_score"),
                        "social_score": alt_features.get("social_score"),
                    }
                    if direction is None and score >= 4.5:
                        logger.warning(
                            "No clear direction for %s despite score=%.2f. Forcing 'long' direction.",
                            symbol,
                            score,
                        )
                        direction = "long"
                    breakdown_data["direction_final"] = direction
                    if direction != "long" or position_size <= 0:
                        skip_reasons: list[str] = []
                        if direction != "long":
                            if direction is None:
                                skip_reasons.append("no long signal (score below cutoff)")
                                _assign_skip_reason(
                                    "direction_none",
                                    "direction from evaluator was None",
                                )
                            else:
                                skip_reasons.append("direction not long")
                                _assign_skip_reason("direction_mismatch", "direction not long")
                        if position_size <= 0:
                            volume_ok = False
                            breakdown_data["volume_ok_for_size"] = False
                            _assign_skip_reason("pos_size_zero", "zero position (low confidence)")
                            skip_reasons.append("zero position (low confidence)")
                        reason_text = " and ".join(skip_reasons) if skip_reasons else "not eligible"
                        custom_skip_reason_text = reason_text
                        logger.info(
                            "[SKIP] %s: direction=%s, size=%s – %s, Score=%.2f",
                            symbol,
                            direction,
                            position_size,
                            reason_text,
                            score,
                        )
                        log_simple_decision_metric(
                            symbol,
                            action="skip",
                            direction=direction,
                            size_value=position_size,
                            score_value=adjusted_score if adjusted_score is not None else score,
                            reason_text=reason_text,
                        )
                        reason_key = custom_skip_reason_key
                        _log_decision("skip")
                        _emit_metrics(
                            "skip",
                            reason_text=reason_text,
                            reason_key=reason_key,
                        )
                        continue
                    flow_analysis = detect_aggression(
                        price_data,
                        symbol=symbol,
                        live_trades=price_data.attrs.get("live_trades"),
                    )
                    flow_status = getattr(flow_analysis, "state", "neutral")
                    volume_profile_result: Optional[VolumeProfileResult] = None
                    lvn_touch: Optional[float] = None
                    last_close = None
                    last_high = None
                    last_low = None
                    try:
                        last_close = float(price_data["close"].iloc[-1])
                        last_high = float(price_data["high"].iloc[-1])
                        last_low = float(price_data["low"].iloc[-1])
                    except Exception:
                        last_close = None
                    if auction_state == "out_of_balance_trend":
                        volume_profile_result = compute_trend_leg_volume_profile(price_data)
                        if volume_profile_result is None or not volume_profile_result.lvns:
                            reason_key = "trend_lvn_missing"
                            reason_text = "unable to derive impulse-leg LVNs for trend continuation."
                            _assign_skip_reason(reason_key, reason_text, overwrite=True)
                            logger.debug("[SKIP] %s: %s", symbol, reason_text)
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else score,
                                reason_text=reason_text,
                            )
                            _log_decision("skip")
                            _emit_metrics(
                                "skip",
                                reason_text=reason_text,
                                reason_key=reason_key,
                            )
                            continue
                        lvn_touch = volume_profile_result.touched_lvn(
                            close=last_close or 0.0,
                            high=last_high,
                            low=last_low,
                        )
                        if lvn_touch is None:
                            reason_key = "trend_lvn_no_touch"
                            reason_text = (
                                "price not interacting with impulse-leg LVN "
                                f"(auction_state={auction_state})."
                            )
                            _assign_skip_reason(reason_key, reason_text, overwrite=True)
                            logger.debug("[SKIP] %s: %s", symbol, reason_text)
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else score,
                                reason_text=reason_text,
                            )
                            _log_decision("skip")
                            _emit_metrics(
                                "skip",
                                reason_text=reason_text,
                                reason_key=reason_key,
                            )
                            continue
                        if flow_status != "buyers in control":
                            reason_key = "trend_lvn_flow_weak"
                            reason_text = (
                                "LVN retest lacks buyer aggression "
                                f"(order flow state={flow_status})."
                            )
                            _assign_skip_reason(reason_key, reason_text, overwrite=True)
                            logger.info("[SKIP] %s: %s", symbol, reason_text)
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else score,
                                reason_text=reason_text,
                            )
                            _log_decision("skip")
                            _emit_metrics(
                                "skip",
                                reason_text=reason_text,
                                reason_key=reason_key,
                            )
                            continue
                    elif auction_state in {"out_of_balance_revert", "balanced"}:
                        volume_profile_result = compute_reversion_leg_volume_profile(price_data)
                        if volume_profile_result is None or not volume_profile_result.lvns:
                            reason_key = "revert_lvn_missing"
                            reason_text = (
                                "unable to derive reclaim-leg LVNs "
                                f"(auction_state={auction_state})."
                            )
                            _assign_skip_reason(reason_key, reason_text, overwrite=True)
                            logger.debug("[SKIP] %s: %s", symbol, reason_text)
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else score,
                                reason_text=reason_text,
                            )
                            _log_decision("skip")
                            _emit_metrics(
                                "skip",
                                reason_text=reason_text,
                                reason_key=reason_key,
                            )
                            continue
                        lvn_touch = volume_profile_result.touched_lvn(
                            close=last_close or 0.0,
                            high=last_high,
                            low=last_low,
                        )
                        if lvn_touch is None:
                            reason_key = "revert_lvn_no_pullback"
                            reason_text = "reclaim leg not pulling back into an LVN."
                            _assign_skip_reason(reason_key, reason_text, overwrite=True)
                            logger.debug("[SKIP] %s: %s", symbol, reason_text)
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else score,
                                reason_text=reason_text,
                            )
                            _log_decision("skip")
                            _emit_metrics(
                                "skip",
                                reason_text=reason_text,
                                reason_key=reason_key,
                            )
                            continue
                        if flow_status != "buyers in control":
                            reason_key = "revert_lvn_flow_weak"
                            reason_text = (
                                "reclaim LVN lacks buyer aggression "
                                f"(order flow state={flow_status})."
                            )
                            _assign_skip_reason(reason_key, reason_text, overwrite=True)
                            logger.info("[SKIP] %s: %s", symbol, reason_text)
                            log_simple_decision_metric(
                                symbol,
                                action="skip",
                                direction=direction,
                                size_value=position_size,
                                score_value=adjusted_score if adjusted_score is not None else score,
                                reason_text=reason_text,
                            )
                            _log_decision("skip")
                            _emit_metrics(
                                "skip",
                                reason_text=reason_text,
                                reason_key=reason_key,
                            )
                            continue
                    else:
                        if flow_status == "sellers in control":
                            logger.warning(
                                "Bearish order flow detected in %s. Proceeding with caution (penalized score handled in evaluate_signal).",
                                symbol,
                            )
                    _log_decision("enter")
                    _emit_metrics("enter")
                    if (row := audit_rows.get(symbol_key)):
                        log_signal_audit(dict(row))
                    potential_trades.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "direction": "long",
                            "position_size": position_size,
                            "pattern": pattern_name,
                            "price_data": price_data,
                            "orderflow": flow_analysis,
                            "auction_state": auction_state,
                            "setup_type": setup_type,
                            "volume_profile": volume_profile_result,
                            "lvn_level": lvn_touch,
                            "alternative_data": alt_features,
                            "alternative_adjustment": alt_adjustment,
                        }
                    )
                    logger.info(
                        "[Potential Trade] %s | Score=%.2f | Direction=long | Size=%s | AuctionState=%s",
                        symbol,
                        score,
                        position_size,
                        auction_state,
                    )
                except Exception as e:
                    logger.error("Error evaluating %s: %s", symbol, e, exc_info=True)
                    if "price_data" in locals() and hasattr(price_data, "attrs"):
                        breakdown_snapshot = price_data.attrs.get("decision_breakdown")
                        signal_features = price_data.attrs.get("signal_features", {})
                    else:
                        breakdown_snapshot = None
                        signal_features = {}
                    symbol_for_metrics = locals().get("symbol_key", str(symbol).upper())
                    try:
                        structured_breakdown = ensure_breakdown_fields(
                            breakdown_snapshot,
                            required_fields=("symbol",),
                        )
                    except Exception:
                        structured_breakdown = None
                    if structured_breakdown is not None:
                        structured_breakdown.setdefault("symbol", symbol_for_metrics)
                        structured_breakdown.setdefault("decision_type", "error")
                        update_breakdown_reason(
                            structured_breakdown,
                            "evaluation_error",
                            "signal evaluation raised",
                        )
                        record_skip_reason("evaluation_error")
                        try:
                            log_decision_breakdown(
                                symbol_for_metrics,
                                signal_features,
                                structured_breakdown,
                            )
                        except Exception:
                            logger.debug(
                                "Failed to emit decision metrics after exception",
                                exc_info=True,
                            )
                    metrics_logged = True
                    continue
                finally:
                    if (
                        evaluation_started
                        and breakdown_data
                        and not metrics_logged
                    ):
                        try:
                            fallback_decision_type = (
                                last_decision_type
                                or breakdown_data.get("decision_type")
                                or "skip"
                            )
                            _emit_metrics(fallback_decision_type)
                        except Exception:
                            logger.warning(
                                "Final decision metric emission failed for %s", symbol_key,
                                exc_info=True,
                            )
                    maybe_log_summary()
            maybe_log_summary()
            if not symbol_scores:
                if cache_miss_symbols:
                    diagnostics = signal_cache.pending_diagnostics()
                    pending_lookup = {entry["symbol"]: entry for entry in diagnostics}
                    preview_infos = [
                        pending_lookup[symbol]
                        for symbol in cache_miss_symbols
                        if symbol in pending_lookup
                    ]
                    preview_infos = preview_infos[:5]
                    now_ts = time.time()
                    stuck_candidates: list[tuple[str, float, dict[str, object]]] = []

                    def _metric_raw(metric: object) -> Optional[float]:
                        if isinstance(metric, dict):
                            raw_val = metric.get("raw")
                            if isinstance(raw_val, (int, float)):
                                return float(raw_val)
                            return None
                        if isinstance(metric, (int, float)):
                            return float(metric)
                        return None

                    def _metric_display(metric: object) -> Optional[float]:
                        if isinstance(metric, dict):
                            display_val = metric.get("display")
                            if isinstance(display_val, (int, float)):
                                return float(display_val)
                            return None
                        if isinstance(metric, (int, float)):
                            return float(metric)
                        return None

                    if SIGNAL_CACHE_PRIME_AFTER > 0:
                        for info in preview_infos:
                            symbol_key = info["symbol"]
                            wait_metrics = [
                                raw
                                for raw in (
                                    _metric_raw(info.get("waiting_for")),
                                    _metric_raw(info.get("stale_age")),
                                    _metric_raw(info.get("request_wait")),
                                )
                                if raw is not None
                            ]
                            max_wait = max(wait_metrics) if wait_metrics else 0.0
                            last_prime = manual_cache_primes.get(symbol_key, 0.0)
                            if (
                                max_wait >= SIGNAL_CACHE_PRIME_AFTER
                                and now_ts - last_prime >= SIGNAL_CACHE_PRIME_COOLDOWN
                            ):
                                stuck_candidates.append((symbol_key, max_wait, info))
                    for symbol_key, max_wait, info in stuck_candidates:
                        last_error = info.get("last_error")
                        if last_error:
                            logger.warning(
                                "Signal cache forcing manual refresh for %s after %.1fs (last error: %s)",
                                symbol_key,
                                max_wait,
                                last_error,
                            )
                        else:
                            logger.warning(
                                "Signal cache forcing manual refresh for %s after %.1fs without data.",
                                symbol_key,
                                max_wait,
                            )
                        timeout = max(1.0, SIGNAL_CACHE_PRIME_TIMEOUT)
                        # better, non-blocking:
                        # dispatch_schedule_refresh(signal_cache, symbol_key) or signal_cache.enqueue_refresh(symbol_key)
                        # if you need a boolean *now*, you can still:
                        success = signal_cache.force_refresh(symbol_key, timeout=timeout)
                        manual_cache_primes[symbol_key] = now_ts
                        if success:
                            logger.info(
                                "Manual refresh succeeded for %s; cache primed.",
                                symbol_key,
                            )
                        else:
                            logger.error(
                                "Manual refresh failed for %s; will retry after cooldown.",
                                symbol_key,
                            )
                    preview_parts: list[str] = []
                    for info in preview_infos:
                        details: list[str] = []
                        waiting_for_display = _metric_display(info.get("waiting_for"))
                        if waiting_for_display is not None:
                            details.append(f"pending={waiting_for_display:.1f}s")
                        stale_age_display = _metric_display(info.get("stale_age"))
                        if stale_age_display is not None:
                            details.append(f"stale={stale_age_display:.1f}s")
                        request_wait_display = _metric_display(info.get("request_wait"))
                        if request_wait_display is not None:
                            details.append(f"wait={request_wait_display:.1f}s")
                        last_error = info.get("last_error")
                        if last_error:
                            error_age_display = _metric_display(info.get("error_age"))
                            if error_age_display is not None:
                                details.append(
                                    f"error {error_age_display:.1f}s ago: {last_error}"
                                )
                            else:
                                details.append(f"error: {last_error}")
                        descriptor = "; ".join(details)
                        if descriptor:
                            preview_parts.append(f"{info['symbol']} ({descriptor})")
                        else:
                            preview_parts.append(info["symbol"])
                    remaining = len(cache_miss_symbols) - len(preview_infos)
                    if remaining > 0:
                        preview_parts.append(f"+{remaining} more")
                    logger.info(
                        "Signal cache still warming up; %d symbols waiting for fresh data. Pending: %s",
                        len(cache_miss_symbols),
                        "; ".join(preview_parts) if preview_parts else ", ".join(cache_miss_symbols[:5]),
                    )
                else:
                    logger.info("No symbol evaluations completed this cycle.")
            # Sort by score and select diversified signals
            potential_trades.sort(key=lambda x: x['score'], reverse=True)
            allowed_new = max_active_trades - len(active_trades)
            opened_count = 0
            # Pass allowed_new as max_trades to avoid misassigning correlation threshold
            selected = select_diversified_signals(potential_trades, max_trades=allowed_new)
            selected_keys = {c['symbol'].upper() for c in selected}
            for candidate in potential_trades:
                sym_key = str(candidate['symbol']).upper()
                if sym_key in selected_keys:
                    _audit_update(sym_key, selected_for_candidate_list=True)
                else:
                    _audit_finalize(
                        sym_key,
                        selected_for_candidate_list=False,
                        final_trade_taken=False,
                        final_skip_reason="not selected after diversification",
                    )
            pending_trade_contexts: list[dict] = []
            batched_prompts: dict[str, str] = {}

            for trade_candidate in selected:
                symbol = trade_candidate['symbol']
                score = trade_candidate['score']
                position_size = trade_candidate['position_size']
                pattern_name = trade_candidate['pattern']
                price_data = trade_candidate['price_data']
                auction_state = trade_candidate.get("auction_state", "unknown")
                setup_type = trade_candidate.get("setup_type")
                alt_features = trade_candidate.get("alternative_data") or {}
                alt_adjustment = float(trade_candidate.get("alternative_adjustment", 0.0))
                try:
                    indicators_df = calculate_indicators(price_data)
                    indicators = {
                        "rsi": float(indicators_df['rsi'].iloc[-1] if 'rsi' in indicators_df else 50.0),
                        "macd": float(indicators_df['macd'].iloc[-1] if 'macd' in indicators_df else 0.0),
                        "macd_signal": float(
                            indicators_df['macd_signal'].iloc[-1]
                            if 'macd_signal' in indicators_df
                            else math.nan
                        ),
                        "adx": float(indicators_df['adx'].iloc[-1] if 'adx' in indicators_df else 20.0),
                        "di_plus": float(
                            indicators_df['di_plus'].iloc[-1]
                            if 'di_plus' in indicators_df
                            else math.nan
                        ),
                        "di_minus": float(
                            indicators_df['di_minus'].iloc[-1]
                            if 'di_minus' in indicators_df
                            else math.nan
                        ),
                    }
                except Exception:
                    indicators_df = price_data
                    indicators = {
                        "rsi": 50.0,
                        "macd": 0.0,
                        "macd_signal": math.nan,
                        "adx": 20.0,
                        "di_plus": math.nan,
                        "di_minus": math.nan,
                    }
                next_ret = 0.0
                try:
                    if not os.path.exists(SEQ_PKL):
                        if schedule_sequence_model_training(indicators_df):
                            logger.info(
                                "Sequence model retraining kicked off in background."
                            )
                    next_ret = predict_next_return(indicators_df.tail(10))
                except Exception:
                    pass
                indicators['next_return'] = next_ret
                flow_analysis = trade_candidate.get("orderflow")
                if flow_analysis is None:
                    flow_analysis = detect_aggression(
                        price_data,
                        symbol=symbol,
                        live_trades=price_data.attrs.get("live_trades"),
                    )
                volume_profile_result = trade_candidate.get("volume_profile")
                if not isinstance(volume_profile_result, VolumeProfileResult):
                    volume_profile_result = None
                lvn_level = trade_candidate.get("lvn_level")
                of_state = getattr(flow_analysis, "state", "neutral")
                orderflow = (
                    "buyers" if of_state == "buyers in control" else
                    "sellers" if of_state == "sellers in control" else
                    "neutral"
                )
                signal_snapshot = price_data.attrs.get("signal_features", {}) or {}
                if setup_type is None:
                    setup_type = signal_snapshot.get("setup_type")
                flow_features = getattr(flow_analysis, "features", {}) or {}
                order_imb_feature = signal_snapshot.get("order_book_imbalance")
                if order_imb_feature is None:
                    order_imb_feature = flow_features.get("order_book_imbalance")
                if order_imb_feature != order_imb_feature or order_imb_feature is None:
                    order_imb_feature = flow_features.get("trade_imbalance")
                try:
                    order_imb_ratio = float(order_imb_feature)
                except (TypeError, ValueError):
                    order_imb_ratio = 0.0
                if order_imb_ratio != order_imb_ratio:
                    order_imb_ratio = 0.0
                order_imb = float(order_imb_ratio) * 100.0
                orderflow_metadata = {
                    "state": getattr(flow_analysis, "state", "neutral") or "neutral",
                    "features": {},
                }
                for key, value in (flow_features or {}).items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = None
                    if numeric is None or not math.isfinite(numeric):
                        orderflow_metadata["features"][key] = None
                    else:
                        orderflow_metadata["features"][key] = numeric
                macro_ind = (
                    100.0
                    if sentiment_bias == "bullish"
                    else -100.0 if sentiment_bias == "bearish" else 0.0
                )
                try:
                    sym_vol_pct = atr_percentile(
                        price_data["high"], price_data["low"], price_data["close"]
                    )
                    sym_vol = sym_vol_pct * 100.0
                except Exception:
                    sym_vol_pct = float("nan")
                    sym_vol = 0.0
                try:
                    ema20 = indicators_df['ema_20'].iloc[-1]
                    ema50 = indicators_df['ema_50'].iloc[-1]
                    last_close_price = float(price_data['close'].iloc[-1])
                    if last_close_price:
                        htf_trend_pct = ((ema20 - ema50) / last_close_price) * 100.0
                    else:
                        htf_trend_pct = 0.0
                except Exception:
                    htf_trend_pct = 0.0
                micro_feature_payload = {
                    'volatility': sym_vol,
                    'htf_trend': htf_trend_pct,
                    'order_imbalance': order_imb,
                    'macro_indicator': macro_ind,
                    'sent_bias': sentiment_bias,
                    'social_sentiment_score': alt_features.get('social_score'),
                    'social_sentiment_confidence': alt_features.get('social_confidence'),
                    'social_sentiment_bias': alt_features.get('social_bias'),
                    'social_posts_analyzed': alt_features.get('social_posts'),
                    'onchain_score': alt_features.get('onchain_score'),
                    'onchain_net_flow': alt_features.get('onchain_net_flow'),
                    'onchain_whale_ratio': alt_features.get('onchain_whale_ratio'),
                    'alternative_score_adjustment': alt_adjustment,
                    'order_flow_score': signal_snapshot.get('order_flow_score'),
                    'order_flow_flag': signal_snapshot.get('order_flow_flag'),
                    'cvd': signal_snapshot.get('cvd'),
                    'cvd_change': signal_snapshot.get('cvd_change'),
                    'cvd_divergence': signal_snapshot.get('cvd_divergence'),
                    'cvd_absorption': signal_snapshot.get('cvd_absorption'),
                    'cvd_accumulation': signal_snapshot.get('cvd_accumulation'),
                    'taker_buy_ratio': signal_snapshot.get('taker_buy_ratio'),
                    'trade_imbalance': signal_snapshot.get('trade_imbalance'),
                    'aggressive_trade_rate': signal_snapshot.get('aggressive_trade_rate'),
                    'spoofing_intensity': signal_snapshot.get('spoofing_intensity'),
                    'spoofing_alert': signal_snapshot.get('spoofing_alert'),
                    'volume_ratio': signal_snapshot.get('volume_ratio'),
                    'price_change_pct': signal_snapshot.get('price_change_pct'),
                    'spread_bps': signal_snapshot.get('spread_bps'),
                    'auction_state': auction_state,
                    'setup_type': setup_type,
                }
                if volume_profile_result is not None:
                    try:
                        micro_feature_payload['volume_poc'] = float(volume_profile_result.poc)
                    except (TypeError, ValueError):
                        micro_feature_payload['volume_poc'] = None
                    micro_feature_payload['lvn_level'] = (
                        float(lvn_level) if isinstance(lvn_level, (int, float)) else None
                    )
                    micro_feature_payload['volume_profile_leg_type'] = volume_profile_result.leg_type
                else:
                    micro_feature_payload['volume_poc'] = None
                    micro_feature_payload['lvn_level'] = None
                    micro_feature_payload['volume_profile_leg_type'] = None

                try:
                    pre_result, prepared = prepare_trade_decision(
                        symbol=symbol,
                        score=score,
                        direction="long",
                        indicators=indicators,
                        session=session,
                        pattern_name=pattern_name,
                        orderflow=orderflow,
                        sentiment=sentiment,
                        macro_news=macro_news_assessment,
                        volatility=sym_vol_pct,
                        fear_greed=fg,
                        auction_state=auction_state,
                        setup_type=setup_type if isinstance(setup_type, str) else None,
                        news_summary=macro_news_summary,
                    )
                except Exception as e:
                    logger.error("Error preparing trade decision for %s: %s", symbol, e, exc_info=True)
                    pre_result = {
                        "decision": False,
                        "confidence": 0.0,
                        "reason": f"Error in prepare_trade_decision(): {e}",
                    }
                    prepared = None

                context = {
                    "symbol": symbol,
                    "score": score,
                    "position_size": position_size,
                    "pattern_name": pattern_name,
                    "price_data": price_data,
                    "auction_state": auction_state,
                    "setup_type": setup_type if isinstance(setup_type, str) else None,
                    "alt_features": alt_features,
                    "alt_adjustment": alt_adjustment,
                    "indicators": indicators,
                    "indicators_df": indicators_df,
                    "volume_profile_result": volume_profile_result,
                    "lvn_level": lvn_level,
                    "orderflow": orderflow,
                    "orderflow_metadata": orderflow_metadata,
                    "micro_feature_payload": micro_feature_payload,
                    "sym_vol_pct": sym_vol_pct,
                    "sym_vol": sym_vol,
                    "htf_trend_pct": htf_trend_pct,
                    "signal_snapshot": signal_snapshot,
                    "macro_ind": macro_ind,
                    "pre_result": pre_result,
                    "prepared": prepared,
                    "tier": tier,
                    "session": session,
                    "news_severity": price_data.attrs.get("news_severity", 0),
                    "atr_15m_ratio": price_data.attrs.get("atr_15m_ratio"),
                }
                pending_trade_contexts.append(context)
                if prepared is not None:
                    batched_prompts[symbol] = prepared.advisor_prompt

            llm_batch_results: dict[str, str] = {}
            if batched_prompts:
                try:
                    llm_batch_results = _run_async_task(
                        lambda: async_batch_llm_judgment(batched_prompts)
                    )
                except Exception as batch_exc:
                    logger.error("Batch LLM evaluation failed: %s", batch_exc, exc_info=True)
                    llm_batch_results = {
                        symbol: "LLM error: batch request failed"
                        for symbol in batched_prompts
                    }

            # Iterate over prepared contexts and open trades once LLM responses are ready
            for context in pending_trade_contexts:
                if opened_count >= allowed_new:
                    break

                symbol = context["symbol"]
                symbol_key = str(symbol).upper()
                prepared = context["prepared"]
                pre_result = context["pre_result"]

                if prepared is None:
                    if not isinstance(pre_result, dict):
                        logger.error("No decision object generated for %s; skipping", symbol)
                        continue
                    decision_obj = pre_result
                else:
                    response = llm_batch_results.get(
                        symbol, "LLM error: missing batch response"
                    )
                    decision_obj = finalize_trade_decision(prepared, response)

                score = context["score"]
                position_size = context["position_size"]
                pattern_name = context["pattern_name"]
                price_data = context["price_data"]
                auction_state = context["auction_state"]
                setup_type = context["setup_type"]
                alt_features = context["alt_features"]
                indicators = context["indicators"]
                indicators_df = context["indicators_df"]
                volume_profile_result = context["volume_profile_result"]
                lvn_level = context["lvn_level"]
                orderflow = context["orderflow"]
                orderflow_metadata = context["orderflow_metadata"]
                micro_feature_payload = context["micro_feature_payload"]
                sym_vol_pct = context["sym_vol_pct"]
                sym_vol = context["sym_vol"]
                htf_trend_pct = context["htf_trend_pct"]
                signal_snapshot = context["signal_snapshot"]
                macro_ind = context["macro_ind"]
                direction = context.get("direction") or "long"

                ml_prob: Optional[float] = None

                decision = bool(decision_obj.get("decision", False))
                final_conf = float(decision_obj.get("confidence", score))
                narrative = decision_obj.get("narrative", "")
                reason = decision_obj.get("reason", "")
                llm_signal = decision_obj.get("llm_decision")
                llm_approval = decision_obj.get("llm_approval")
                technical_score = decision_obj.get("technical_indicator_score")
                if technical_score is None:
                    technical_score = summarise_technical_score(indicators, "long")

                logger.info(
                    "[Brain] %s -> %s | Confidence: %.2f | Reason: %s",
                    symbol,
                    decision,
                    final_conf,
                    reason,
                )

                if not decision:
                    rejection_reason = reason or "Unknown reason"
                    log_simple_decision_metric(
                        symbol,
                        action="skip",
                        direction=direction,
                        size_value=position_size,
                        score_value=score,
                        reason_text=f"brain veto: {rejection_reason}",
                    )
                    log_rejection(symbol, rejection_reason)
                    _audit_finalize(
                        symbol_key,
                        selected_for_candidate_list=True,
                        brain_veto=True,
                        final_trade_taken=False,
                        final_skip_reason=rejection_reason,
                    )
                    continue

                ml_prob = predict_success_probability(
                    score=score,
                    confidence=final_conf,
                    session=session,
                    btc_d=btc_d if btc_d is not None else float("nan"),
                    fg=float(fg) if fg is not None else float("nan"),
                    sentiment_conf=sentiment_confidence,
                    pattern=pattern_name,
                    llm_approval=bool(llm_approval) if llm_approval is not None else True,
                    llm_confidence=decision_obj.get("llm_confidence", 5.0),
                    micro_features=micro_feature_payload,
                )
                _audit_update(symbol_key, ml_probability=ml_prob)
                if ml_prob < 0.5:
                    logger.info(
                        "ML model predicted low success probability (%.2f) for %s. Skipping trade.",
                        ml_prob,
                        symbol,
                    )
                    ml_reason = f"ML prob {ml_prob:.2f} too low"
                    log_simple_decision_metric(
                        symbol,
                        action="skip",
                        direction=direction,
                        size_value=position_size,
                        score_value=score,
                        reason_text=ml_reason,
                    )
                    log_rejection(symbol, ml_reason)
                    _audit_finalize(
                        symbol_key,
                        selected_for_candidate_list=True,
                        ml_probability=ml_prob,
                        ml_veto=True,
                        final_trade_taken=False,
                        final_skip_reason=ml_reason,
                    )
                    continue

                final_conf = round((final_conf + ml_prob * 10) / 2.0, 2)

                risk_payload = {
                    "symbol": symbol,
                    "direction": "long",
                    "confidence": final_conf,
                    "ml_probability": ml_prob,
                    "volatility": sym_vol_pct,
                    "htf_trend_pct": htf_trend_pct,
                    "orderflow": orderflow,
                    "macro_bias": sentiment_bias,
                    "session": session,
                    "setup_type": setup_type if isinstance(setup_type, str) else None,
                    "max_trades": max_active_trades,
                    "open_positions": len(active_trades),
                    "btc_trend": btc_trend_bias,
                    "time_to_news_minutes": next_news_minutes,
                    "volatility_threshold": VOLATILITY_SPIKE_THRESHOLD,
                    "max_rr": 2.0,
                }
                volume_profile_note = None
                if volume_profile_result is not None:
                    try:
                        volume_profile_note = (
                            f"POC {float(volume_profile_result.poc):.4f} ({volume_profile_result.leg_type})"
                        )
                    except Exception:
                        volume_profile_note = volume_profile_result.leg_type if volume_profile_result else None
                explainer_payload = {
                    "symbol": symbol,
                    "pattern": pattern_name,
                    "score": score,
                    "confidence": final_conf,
                    "macro_bias": sentiment_bias,
                    "orderflow": orderflow,
                    "volume_profile": volume_profile_note,
                    "context": narrative or llm_signal,
                }
                signal_summary = generate_signal_explainer(explainer_payload)
                risk_review = run_pretrade_risk_check(risk_payload)
                if not isinstance(risk_review, dict):
                    risk_review = {
                        "enter": True,
                        "reasons": ["risk check unavailable"],
                        "conflicts": [],
                        "max_rr": 2.0,
                    }
                risk_enter = bool(risk_review.get("enter", True))
                conflicts = [str(item) for item in (risk_review.get("conflicts") or []) if item]
                reasons_list = [str(item) for item in (risk_review.get("reasons") or []) if item]
                decision_label = "approved" if risk_enter else "veto"
                if signal_summary:
                    logger.info(
                        "[Signal Explainer][%s] %s",
                        decision_label,
                        signal_summary.replace("\n", " "),
                    )
                else:
                    logger.info("[Signal Explainer][%s] unavailable", decision_label)
                if not risk_enter:
                    conflict_text = ", ".join(conflicts) if conflicts else ", ".join(reasons_list)
                    logger.info(
                        "Risk vetoed %s | conflicts=%s | reasons=%s | max_rr=%.2f",
                        symbol,
                        ", ".join(conflicts) if conflicts else "none",
                        ", ".join(reasons_list) if reasons_list else "none",
                        float(risk_review.get("max_rr", 0.0)),
                    )
                    risk_reason = f"Risk veto: {conflict_text or 'guardrail failure'}"
                    log_simple_decision_metric(
                        symbol,
                        action="skip",
                        direction=direction,
                        size_value=position_size,
                        score_value=score,
                        reason_text=risk_reason,
                    )
                    log_rejection(symbol, risk_reason)
                    _audit_finalize(
                        symbol_key,
                        selected_for_candidate_list=True,
                        ml_probability=ml_prob,
                        risk_veto=True,
                        final_trade_taken=False,
                        final_skip_reason=risk_reason,
                    )
                    continue
                else:
                    logger.info(
                        "Risk approved %s | reasons=%s | max_rr=%.2f",
                        symbol,
                        ", ".join(reasons_list) if reasons_list else "none",
                        float(risk_review.get("max_rr", 0.0)),
                    )
                if position_size <= 0:
                    reason_text = "computed position size <= 0 after ML veto integration"
                    logger.info("[SKIP] %s: %s", symbol, reason_text)
                    log_simple_decision_metric(
                        symbol,
                        action="skip",
                        direction="long",
                        size_value=position_size,
                        score_value=score,
                        reason_text=reason_text,
                    )
                    _audit_finalize(
                        symbol_key,
                        selected_for_candidate_list=True,
                        ml_probability=ml_prob,
                        final_trade_taken=False,
                        final_skip_reason=reason_text,
                    )
                    continue

                try:
                    raw_entry_price = float(price_data['close'].iloc[-1])
                    entry_price = round(raw_entry_price, 6)
                    micro_plan = None
                    order_book_snapshot = None
                    try:
                        order_book_snapshot = get_order_book(symbol, limit=20)
                    except Exception as exc:
                        logger.debug(
                            "Order book snapshot unavailable for %s: %s",
                            symbol,
                            exc,
                            exc_info=True,
                        )
                    if order_book_snapshot:
                        try:
                            micro_plan = plan_execution(
                                "buy",
                                raw_entry_price,
                                order_book_snapshot,
                                depth=15,
                            )
                            rec_price = micro_plan.get("recommended_price")
                            if rec_price is not None:
                                entry_price = round(float(rec_price), 6)
                        except Exception as exc:
                            logger.debug(
                                "Failed to compute execution plan for %s: %s",
                                symbol,
                                exc,
                                exc_info=True,
                            )
                            micro_plan = None
                    try:
                        atr_val = (
                            indicators_df['atr'].iloc[-1]
                            if 'atr' in indicators_df
                            else None
                        )
                    except Exception:
                        atr_val = None
                    trade_usd = calculate_dynamic_trade_size(final_conf, ml_prob, score)
                    if atr_val is not None and not np.isnan(atr_val) and atr_val > 0:
                        base_atr = float(atr_val)
                    else:
                        base_atr = entry_price * 0.02
                    sl_dist = base_atr * ATR_STOP_MULTIPLIER
                    state = get_rl_state(sym_vol_pct)
                    mult = 1.0
                    if USE_RL_POSITION_SIZER and rl_sizer is not None:
                        mult = rl_sizer.select_multiplier(state)
                    trade_usd *= mult
                    if micro_plan and micro_plan.get("size_multiplier") is not None:
                        try:
                            trade_usd *= float(micro_plan.get("size_multiplier", 1.0))
                        except (TypeError, ValueError):
                            pass
                    trade_usd = max(MIN_TRADE_USD, min(MAX_TRADE_USD, trade_usd))
                    position_size = round(max(trade_usd / entry_price, 0), 6)
                    _audit_update(
                        symbol_key,
                        size_bucket=position_size,
                        volume_ok_from_size=bool(position_size > 0),
                    )
                    volume_profile_summary = None
                    poc_price: Optional[float] = None
                    poc_target_price: Optional[float] = None
                    lvn_entry_value: Optional[float] = None
                    lvn_stop_price: Optional[float] = None
                    if volume_profile_result is not None:
                        volume_profile_summary = volume_profile_result.to_dict()
                        try:
                            poc_price = float(volume_profile_result.poc)
                            if math.isfinite(poc_price):
                                poc_target_price = round(poc_price, 6)
                        except (TypeError, ValueError):
                            poc_price = None
                        try:
                            touched_lvn = volume_profile_result.touched_lvn(
                                close=price_data["close"].iloc[-1],
                                high=price_data["high"].iloc[-1],
                                low=price_data["low"].iloc[-1],
                            )
                        except Exception:
                            touched_lvn = None
                        if touched_lvn is not None and math.isfinite(touched_lvn):
                            lvn_entry_value = round(float(touched_lvn), 6)
                    sl = round(entry_price - sl_dist, 6)
                    tp_candidates: list[Optional[float]] = []
                    for multiplier in TP_ATR_MULTIPLIERS:
                        tp_value = round(entry_price + base_atr * multiplier, 6)
                        tp_candidates.append(tp_value)
                    while len(tp_candidates) < 3:
                        tp_candidates.append(None)
                    tp1, tp2, tp3 = tp_candidates[:3]
                    if volume_profile_result is not None:
                        failed_low = volume_profile_result.metadata.get("failed_low")
                        if failed_low is not None and math.isfinite(failed_low):
                            buffer = max(
                                volume_profile_result.bin_width,
                                entry_price * 0.001,
                            )
                            candidate_sl = float(failed_low) - buffer
                            if candidate_sl > 0:
                                sl = round(candidate_sl, 6)
                                lvn_stop_price = sl
                        if poc_price is not None and math.isfinite(poc_price):
                            if poc_price > entry_price * 1.0005:
                                poc_target_price = round(poc_price, 6)
                    if lvn_entry_value is None and isinstance(lvn_level, (int, float)) and math.isfinite(lvn_level):
                        lvn_entry_value = round(float(lvn_level), 6)
                    if volume_profile_result is not None and lvn_stop_price is None:
                        lvn_stop_price = sl
                    htf_trend = htf_trend_pct
                    risk_amount = position_size * sl_dist
                    strategy_label = "PatternTrade"
                    llm_decision_token = str(llm_signal or "").strip().lower()
                    if llm_approval is True:
                        strategy_label = "PatternTrade+LLM"
                    elif llm_approval is False:
                        strategy_label = "PatternTrade-LLM"
                    elif llm_decision_token in {"approved", "greenlight", "yes"}:
                        strategy_label = "PatternTrade+LLM"
                    elif llm_decision_token in {"vetoed", "rejected", "no"}:
                        strategy_label = "PatternTrade-LLM"

                    new_trade = {
                        "symbol": symbol,
                        "direction": "long",
                        "entry": entry_price,
                        "entry_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "atr_at_entry": round(base_atr, 6),
                        "trailing_active": False,
                        "position_size": position_size,
                        "size": trade_usd,
                        "initial_size": position_size,
                        "risk_amount": risk_amount,
                        "rl_state": state,
                        "rl_multiplier": mult,
                        "leverage": 1,
                        "confidence": final_conf,
                        "score": score,
                        "session": session,
                        "btc_dominance": btc_d,
                        "fear_greed": fg,
                        "sentiment_bias": sentiment_bias,
                        "sentiment_confidence": sentiment_confidence,
                        "sentiment_summary": sentiment.get("summary", ""),
                        "volatility": sym_vol,
                        "htf_trend": htf_trend,
                        "order_imbalance": order_imb,
                        "order_flow_score": signal_snapshot.get("order_flow_score"),
                        "order_flow_flag": signal_snapshot.get("order_flow_flag"),
                        "order_flow_state": signal_snapshot.get("order_flow_state"),
                        "cvd": signal_snapshot.get("cvd"),
                        "cvd_change": signal_snapshot.get("cvd_change"),
                        "cvd_divergence": signal_snapshot.get("cvd_divergence"),
                        "cvd_absorption": signal_snapshot.get("cvd_absorption"),
                        "cvd_accumulation": signal_snapshot.get("cvd_accumulation"),
                        "taker_buy_ratio": signal_snapshot.get("taker_buy_ratio"),
                        "trade_imbalance": signal_snapshot.get("trade_imbalance"),
                        "aggressive_trade_rate": signal_snapshot.get("aggressive_trade_rate"),
                        "spoofing_intensity": signal_snapshot.get("spoofing_intensity"),
                        "spoofing_alert": signal_snapshot.get("spoofing_alert"),
                        "volume_ratio": signal_snapshot.get("volume_ratio"),
                        "price_change_pct": signal_snapshot.get("price_change_pct"),
                        "spread_bps": signal_snapshot.get("spread_bps"),
                        "macro_indicator": macro_ind,
                        "pattern": pattern_name,
                        "strategy": strategy_label,
                        "narrative": narrative,
                        "ml_prob": ml_prob,
                        "llm_decision": llm_signal,
                        "llm_approval": llm_approval,
                        "llm_confidence": decision_obj.get("llm_confidence"),
                        "llm_error": decision_obj.get("llm_error"),
                        "technical_indicator_score": technical_score,
                        "status": {"tp1": False, "tp2": False, "tp3": False, "sl": False},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "news_summary": decision_obj.get("news_summary", ""),
                        "volume_profile": volume_profile_summary,
                        "lvn_entry_level": lvn_entry_value,
                        "lvn_stop": lvn_stop_price,
                        "poc_target": poc_target_price,
                        "auction_state": auction_state,
                        "orderflow_analysis": orderflow_metadata,
                        "take_profit_strategy": "atr_trailing",
                    }
                    if micro_plan:
                        logger.debug("Microstructure plan for %s: %s", symbol, micro_plan)
                        new_trade["microstructure_plan"] = micro_plan
                    logger.info("Narrative:\n%s\n", narrative)
                    logger.info(
                        "Trade Opened %s @ %s | Notional=%s USD | Qty=%s | TP1 %s / TP2 %s / TP3 %s",
                        symbol,
                        entry_price,
                        trade_usd,
                        position_size,
                        tp1,
                        tp2,
                        tp3,
                    )
                    if create_new_trade(
                        new_trade,
                        stop_price=lvn_stop_price,
                        target_price=poc_target_price,
                        auction_state=auction_state,
                        lvn_entry_level=lvn_entry_value,
                        orderflow_analysis=orderflow_metadata,
                    ):
                        active_trades.append(new_trade)
                        save_active_trades(active_trades)
                        try:
                            state.merge_narrative(
                                symbol,
                                {
                                    "last_open_narrative": narrative,
                                    "opened_at": time.time(),
                                },
                            )
                        except Exception:
                            logger.debug("Failed to persist narrative state for %s", symbol, exc_info=True)
                        send_email(
                            f"New Trade Opened: {symbol}",
                            f"{new_trade}\n\n Narrative:\n{narrative}\n\nNews Summary:\n{decision_obj.get('news_summary', '')}",
                        )
                        opened_count += 1
                        log_simple_decision_metric(
                            symbol,
                            action="enter",
                            direction=direction,
                            size_value=position_size,
                            score_value=score,
                            reason_text=(
                                f"trade opened @ {entry_price} | notional={trade_usd} | qty={position_size}"
                            ),
                        )
                        _audit_finalize(
                            symbol_key,
                            selected_for_candidate_list=True,
                            ml_probability=ml_prob,
                            final_trade_taken=True,
                            final_skip_reason="",
                            brain_veto=False,
                            ml_veto=False,
                            risk_veto=False,
                        )
                    else:
                        reason_text = "trade already active"
                        logger.info("Trade for %s already active; skipping new entry", symbol)
                        log_simple_decision_metric(
                            symbol,
                            action="skip",
                            direction=direction,
                            size_value=position_size,
                            score_value=score,
                            reason_text=reason_text,
                        )
                        _audit_finalize(
                            symbol_key,
                            selected_for_candidate_list=True,
                            ml_probability=ml_prob,
                            final_trade_taken=False,
                            final_skip_reason=reason_text,
                        )
                except Exception as e:
                    logger.error("Error opening trade for %s: %s", symbol, e, exc_info=True)
                    _audit_finalize(
                        symbol_key,
                        selected_for_candidate_list=True,
                        ml_probability=ml_prob if ml_prob is not None else "",
                        final_trade_taken=False,
                        final_skip_reason=f"error opening trade: {e}",
                    )
            for remaining_symbol in list(audit_rows.keys()):
                _audit_finalize(
                    remaining_symbol,
                    final_trade_taken=False,
                    final_skip_reason="no decision",
                )
            # Manage existing trades after opening new ones
            try:
                manage_trades()
            except Exception as e:
                logger.error("Error managing trades: %s", e, exc_info=True)
            # Persist symbol scores to disk
            try:
                old_data = load_symbol_scores()
            except Exception as e:
                logger.error("Error reading symbol_scores.json: %s", e, exc_info=True)
                old_data = {}
            try:
                old_data.update(symbol_scores)
                save_symbol_scores(old_data)
                logger.info("Saved symbol scores (persistent memory updated).")
            except Exception as e:
                logger.error("Error saving symbol scores: %s", e, exc_info=True)
        except Exception as e:
            logger.error("Main Loop Error: %s", e, exc_info=True)
            guard_stop.wait(10)
        finally:
            scan_lock.release()

def main() -> None:
    """Program entry point with WebSocket/bootstrap wiring."""

    global _ws_bridge

    import os  # local import to mirror legacy bootstrap expectations
    from ws_price_bridge import WSPriceBridge

    # --- RTSC bootstrap (in-process singleton) ---
    from realtime_signal_cache import RealTimeSignalCache, get_active_cache, set_active_cache

    sc = get_active_cache()
    if not sc:
        logger.info("RTSC bootstrap: creating RealTimeSignalCache()")
        (
            refresh_interval,
            stale_mult,
            max_conc,
            _scan_interval,
            _estimated_cycle,
        ) = _prepare_signal_cache_params()
        sc = RealTimeSignalCache(
            price_fetcher=get_price_data_async,
            evaluator=evaluator_for_cache,
            refresh_interval=refresh_interval,
            stale_after=refresh_interval * stale_mult,
            max_concurrency=max_conc,
            use_streams=ENABLE_WS_BRIDGE,
        )
        set_active_cache(sc)
    else:
        logger.info("RTSC bootstrap: reusing existing cache")

    try:
        _schedule_rtsc_diag(sc)
    except Exception:
        logger.debug("RTSC_DIAG: scheduling failed", exc_info=True)

    if not hasattr(sc, "on_kline") and hasattr(sc, "handle_ws_update"):
        def _cache_on_kline(symbol: str, interval: str, payload: Dict[str, Any]) -> None:
            try:
                sc.handle_ws_update(symbol, "kline", payload)
            except Exception:
                logger.debug("RTSC on_kline adapter failed for %s", symbol, exc_info=True)

        setattr(sc, "on_kline", _cache_on_kline)

    signal_cache = sc

    ws_bridge: Optional[WSPriceBridge] = None
    try:
        if runtime_settings.use_ws_prices:
            ws_symbols: list[str] = []
            try:
                if hasattr(signal_cache, "symbols"):
                    candidates = signal_cache.symbols()
                    ws_symbols = sorted(
                        {str(sym).upper() for sym in candidates if isinstance(sym, str)}
                    )
            except Exception:
                ws_symbols = []
            if not ws_symbols:
                ws_symbols = ["BTCUSDT", "ETHUSDT"]

            if _ws_bridge is not None:
                ws_bridge = _ws_bridge
                if hasattr(ws_bridge, "update_symbols"):
                    try:
                        ws_bridge.update_symbols(ws_symbols)
                    except Exception:
                        logger.debug("WS bootstrap: failed to update symbols", exc_info=True)
            else:
                try:
                    ws_bridge = WSPriceBridge(
                        symbols=ws_symbols,
                        on_kline=getattr(sc, "on_kline", None),
                        on_ticker=getattr(sc, "on_ticker", None),
                        on_book_ticker=getattr(sc, "on_book_ticker", None),
                    )
                except TypeError:
                    ws_bridge = WSPriceBridge(symbols=ws_symbols)
                _ws_bridge = ws_bridge

            if ws_bridge is not None:
                if getattr(ws_bridge, "on_kline", None) is None and hasattr(sc, "on_kline"):
                    try:
                        ws_bridge.on_kline = sc.on_kline  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("WS bootstrap: unable to set on_kline", exc_info=True)
                if getattr(ws_bridge, "on_ticker", None) is None and hasattr(sc, "on_ticker"):
                    try:
                        ws_bridge.on_ticker = getattr(sc, "on_ticker")  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("WS bootstrap: unable to set on_ticker", exc_info=True)
                if getattr(ws_bridge, "on_book_ticker", None) is None and hasattr(sc, "on_book_ticker"):
                    try:
                        ws_bridge.on_book_ticker = getattr(sc, "on_book_ticker")  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("WS bootstrap: unable to set on_book_ticker", exc_info=True)

                if hasattr(ws_bridge, "start"):
                    ws_bridge.start()
                elif hasattr(ws_bridge, "enable_streams"):
                    ws_bridge.enable_streams()

                try:
                    signal_cache.enable_streams(ws_bridge)
                    logger.info("RTSC: enable_streams(ws_bridge) wired successfully.")
                except Exception as e:
                    logger.exception("RTSC wiring failed: %s", e)
    except Exception as e:
        logger.exception("WS bootstrap failed: %s", e)
    
    logger.info("Starting Spot AI Super Agent loop...")
    run_agent_loop()


if __name__ == "__main__":
    main()
