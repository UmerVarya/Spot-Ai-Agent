"""Background Groq LLM monitor for continuously evaluating news severity.

This module runs a lightweight asynchronous loop that periodically refreshes
news events and asks the Groq LLM (or local fallback) to assess their risk.
If the LLM determines that conditions are unsafe, the monitor raises an alert
and optionally requests that trading be halted temporarily.

The monitor is designed to be resilient: it deduplicates consecutive alerts,
falls back to deterministic defaults when the LLM is unavailable and exposes a
thread-safe accessor for the latest assessment so the main agent loop can make
real-time decisions.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, MutableMapping
import hashlib
import inspect

from fetch_news import analyze_news_with_llm_async, run_news_fetcher_async
from log_utils import setup_logger
from risk_veto import minutes_until_next_event

LOGGER = setup_logger(__name__)


NewsFetcher = Callable[[], Awaitable[Iterable[Mapping[str, Any]]]]
NewsAnalyzer = Callable[[Iterable[Mapping[str, Any]]], Awaitable[Mapping[str, Any]]]
AlertCallback = Callable[["NewsAlert"], Any]


async def _default_fetcher() -> Iterable[Mapping[str, Any]]:
    return await run_news_fetcher_async()


async def _default_analyzer(events: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    return await analyze_news_with_llm_async(list(events))


@dataclass(frozen=True)
class NewsAlert:
    """Container describing an alert emitted by :class:`LLMNewsMonitor`."""

    safe: bool
    sensitivity: float
    reason: str
    halt_trading: bool
    triggered_at: float
    events: tuple[Mapping[str, Any], ...]


class LLMNewsMonitor:
    """Continuously evaluate news severity using an LLM."""

    def __init__(
        self,
        *,
        interval: float = 120.0,
        alert_threshold: float = 0.6,
        halt_threshold: float = 0.85,
        stale_after: float | None = None,
        status_path: str | None = None,
        fetcher: NewsFetcher | None = None,
        analyzer: NewsAnalyzer | None = None,
        alert_callback: AlertCallback | None = None,
    ) -> None:
        self.interval = max(10.0, float(interval))
        self.alert_threshold = max(0.0, float(alert_threshold))
        self.halt_threshold = max(self.alert_threshold, float(halt_threshold))
        if stale_after is None:
            self.stale_after = self.interval * 3
        else:
            self.stale_after = max(0.0, float(stale_after))
        self.status_path = Path(status_path) if status_path else None
        self._fetcher = fetcher or _default_fetcher
        self._analyzer = analyzer or _default_analyzer
        self._alert_callback = alert_callback

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_state: MutableMapping[str, Any] = {}
        self._last_alert_fingerprint: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the monitor in a daemon thread if not already running."""

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Signal the background loop to stop."""

        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=5.0)

    def set_alert_callback(self, callback: AlertCallback | None) -> None:
        """Register a callback that receives :class:`NewsAlert` objects."""

        self._alert_callback = callback

    def get_latest_assessment(self) -> Mapping[str, Any] | None:
        """Return the latest LLM assessment in a thread-safe manner."""

        with self._lock:
            if not self._latest_state:
                return None
            state = dict(self._latest_state)
        last_checked = state.get("last_checked")
        state["stale"] = self._is_stale(last_checked)
        return state

    async def evaluate_now(self) -> Mapping[str, Any] | None:
        """Run a single fetch + analysis cycle immediately."""

        return await self._evaluate_cycle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._continuous_loop())
        finally:
            loop.close()

    async def _continuous_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._evaluate_cycle()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.error("News monitor iteration failed: %s", exc, exc_info=True)
            try:
                await asyncio.wait_for(asyncio.sleep(self.interval), timeout=self.interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:  # pragma: no cover - not expected in tests
                break

    async def _evaluate_cycle(self) -> Mapping[str, Any] | None:
        try:
            events_iterable = await self._fetcher()
            events = list(events_iterable)
        except Exception as exc:
            LOGGER.warning("News monitor fetch failed: %s", exc, exc_info=True)
            events = []

        assessment: Mapping[str, Any]
        if not events:
            assessment = {"safe": True, "sensitivity": 0.0, "reason": "No events fetched"}
        else:
            try:
                assessment = await self._analyzer(events)
            except Exception as exc:
                LOGGER.warning("News monitor LLM analysis failed: %s", exc, exc_info=True)
                assessment = {"safe": True, "sensitivity": 0.0, "reason": "LLM unavailable"}

        state = self._compose_state(assessment, events)
        self._persist_state(state)
        await self._maybe_emit_alert(state, events)
        return state

    def _compose_state(
        self,
        assessment: Mapping[str, Any],
        events: Iterable[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        safe = bool(assessment.get("safe", True))
        try:
            sensitivity = float(assessment.get("sensitivity", 0.0))
        except (TypeError, ValueError):
            sensitivity = 0.0
        reason = str(assessment.get("reason", "")) or "No reason provided"
        severity = sensitivity
        if not safe:
            severity = max(severity, 1.0)
        halt_trading = severity >= self.halt_threshold
        alert_triggered = severity >= self.alert_threshold
        next_event_minutes = minutes_until_next_event(events) if events else None
        now_iso = datetime.now(timezone.utc).isoformat()

        # Limit stored events to the most recent handful to keep the payload light.
        trimmed_events = []
        for idx, event in enumerate(events):
            if idx >= 10:
                break
            trimmed_events.append(dict(event))

        state: dict[str, Any] = {
            "safe": safe,
            "sensitivity": round(sensitivity, 3),
            "reason": reason,
            "severity": round(severity, 3),
            "halt_trading": halt_trading,
            "alert_triggered": alert_triggered,
            "next_event_minutes": next_event_minutes,
            "last_checked": now_iso,
            "events": trimmed_events,
        }
        return state

    def _persist_state(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self._latest_state = dict(state)
        if not self.status_path:
            return
        try:
            self.status_path.write_text(json.dumps(state, indent=2))
        except Exception as exc:  # pragma: no cover - filesystem failures
            LOGGER.warning("Failed to persist news monitor state: %s", exc, exc_info=True)

    async def _maybe_emit_alert(
        self,
        state: Mapping[str, Any],
        events: Iterable[Mapping[str, Any]],
    ) -> None:
        if not state.get("alert_triggered"):
            return
        fingerprint = self._fingerprint(state)
        if fingerprint == self._last_alert_fingerprint:
            return
        self._last_alert_fingerprint = fingerprint

        callback = self._alert_callback
        if not callback:
            return

        alert = NewsAlert(
            safe=bool(state.get("safe", True)),
            sensitivity=float(state.get("sensitivity", 0.0)),
            reason=str(state.get("reason", "")),
            halt_trading=bool(state.get("halt_trading", False)),
            triggered_at=time.time(),
            events=tuple(dict(event) for event in events)  # type: ignore[arg-type]
        )
        try:
            result = callback(alert)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("News monitor alert callback failed: %s", exc, exc_info=True)

    def _fingerprint(self, state: Mapping[str, Any]) -> str:
        payload = json.dumps(
            {
                "safe": state.get("safe"),
                "severity": state.get("severity"),
                "reason": state.get("reason"),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _is_stale(self, last_checked: Any) -> bool:
        if not last_checked or self.stale_after <= 0:
            return False
        try:
            timestamp = datetime.fromisoformat(str(last_checked).replace("Z", "+00:00"))
        except ValueError:
            return False
        now = datetime.now(timezone.utc)
        return now - timestamp > timedelta(seconds=self.stale_after)


_MONITOR_LOCK = threading.Lock()
_DEFAULT_MONITOR: LLMNewsMonitor | None = None


def start_background_news_monitor(**kwargs: Any) -> LLMNewsMonitor:
    """Return a shared :class:`LLMNewsMonitor` instance and ensure it is running."""

    global _DEFAULT_MONITOR
    with _MONITOR_LOCK:
        if _DEFAULT_MONITOR is None:
            _DEFAULT_MONITOR = LLMNewsMonitor(**kwargs)
            _DEFAULT_MONITOR.start()
        else:
            # Allow updating the callback dynamically if supplied.
            if "alert_callback" in kwargs and kwargs["alert_callback"] is not None:
                _DEFAULT_MONITOR.set_alert_callback(kwargs["alert_callback"])
    return _DEFAULT_MONITOR


def get_news_monitor() -> LLMNewsMonitor | None:
    """Return the shared news monitor if it has been started."""

    with _MONITOR_LOCK:
        return _DEFAULT_MONITOR
