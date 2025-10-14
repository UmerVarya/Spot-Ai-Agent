"""Worker pool utilities for the event-driven trading agent."""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional


CallableType = Callable[..., Any]


class WorkerPools:
    """Container that exposes IO and CPU oriented pools."""

    def __init__(self, *, io_workers: int = 6, cpu_workers: Optional[int] = None) -> None:
        if cpu_workers is None:
            cpu_workers = max(1, (os.cpu_count() or 2) // 2)
        self._io_pool = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="agent-io")
        self._cpu_pool = ProcessPoolExecutor(max_workers=cpu_workers)
        self._shutdown = False
        self._lock = threading.Lock()

    def submit_io(self, fn: CallableType, *args: Any, **kwargs: Any) -> Future:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("WorkerPools has been shut down")
            return self._io_pool.submit(fn, *args, **kwargs)

    def submit_cpu(self, fn: CallableType, *args: Any, **kwargs: Any) -> Future:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("WorkerPools has been shut down")
            return self._cpu_pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._io_pool.shutdown(wait=wait, cancel_futures=True)
            self._cpu_pool.shutdown(wait=wait, cancel_futures=True)


class ScheduledTask:
    """Helper to schedule periodic tasks with jitter."""

    def __init__(
        self,
        name: str,
        *,
        min_interval: float,
        max_interval: float,
        jitter: float = 0.1,
    ) -> None:
        self.name = name
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.jitter = float(jitter)
        self.next_run = 0.0
        self._lock = threading.Lock()

    def due(self, now: Optional[float] = None) -> bool:
        timestamp = float(now if now is not None else time.time())
        with self._lock:
            return timestamp >= self.next_run

    def schedule_next(self, now: Optional[float] = None) -> None:
        timestamp = float(now if now is not None else time.time())
        interval = random.uniform(self.min_interval, self.max_interval)
        if self.jitter:
            jitter = interval * random.uniform(-self.jitter, self.jitter)
            interval = max(0.0, interval + jitter)
        with self._lock:
            self.next_run = timestamp + interval


__all__ = ["WorkerPools", "ScheduledTask"]
