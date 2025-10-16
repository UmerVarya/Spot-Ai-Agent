"""Lightweight observability helpers for structured logs and metrics.

The trading agent runs in environments where pulling in a full metrics stack
is not always possible.  This module provides tiny utilities that are cheap to
import and good enough for unit tests:

* ``log_event`` emits JSON encoded log lines with a consistent schema so the
  caller's logger configuration can ship them to any sink.
* ``record_metric`` appends gauge/counter style metrics to a CSV file that can
  be scraped or tailed by lightweight dashboards.

Both helpers are intentionally threadsafe and avoid heavy dependencies to
minimise their runtime footprint in latency sensitive paths.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

_OBSERVABILITY_LOGGER = logging.getLogger("observability")


def log_event(logger: Optional[logging.Logger], event: str, **fields: Any) -> None:
    """Emit a structured JSON log entry.

    Parameters
    ----------
    logger:
        Logger instance to use.  When ``None`` the module level observability
        logger is used.
    event:
        Short event identifier.  Stored under the ``event`` key in the emitted
        payload.
    **fields:
        Additional key/value pairs to include in the log entry.  Values must be
        JSON serialisable.
    """

    payload: MutableMapping[str, Any] = {"event": event, "ts": time.time()}
    payload.update(fields)
    target = logger or _OBSERVABILITY_LOGGER
    try:
        target.info(json.dumps(payload, sort_keys=True))
    except TypeError:
        serialisable = {k: _safe_json_value(v) for k, v in payload.items()}
        target.info(json.dumps(serialisable, sort_keys=True))


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


class _CsvMetricsSink:
    """Thread-safe CSV metrics recorder."""

    def __init__(self, path: Optional[str] = None) -> None:
        metrics_path = Path(path or os.getenv("METRICS_PATH", "metrics.csv"))
        self._path = metrics_path
        self._lock = threading.Lock()
        self._initialised = False

    def record(self, metric: str, value: float, *, labels: Optional[Mapping[str, Any]] = None) -> None:
        row = {
            "ts": f"{time.time():.6f}",
            "metric": metric,
            "value": f"{float(value):.6f}",
            "labels": json.dumps(labels or {}, sort_keys=True),
        }
        with self._lock:
            need_header = not self._initialised or not self._path.exists()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("ts", "metric", "value", "labels"))
                if need_header:
                    writer.writeheader()
                    self._initialised = True
                writer.writerow(row)


_metrics_sink = _CsvMetricsSink()


def record_metric(metric: str, value: float, *, labels: Optional[Mapping[str, Any]] = None) -> None:
    """Record a numeric metric to the CSV sink."""

    try:
        _metrics_sink.record(metric, value, labels=labels)
    except Exception:
        _OBSERVABILITY_LOGGER.debug("Failed to record metric %s", metric, exc_info=True)


__all__ = ["log_event", "record_metric"]
