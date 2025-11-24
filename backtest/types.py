from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestProgress:
    phase: str
    current: int = 0
    total: int = 0
    message: str = ""

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        if self.current <= 0:
            return 0.0
        return min(max(self.current / float(self.total), 0.0), 1.0)


ProgressCallback = Callable[[BacktestProgress], None]


def emit_progress(callback: Optional[ProgressCallback], progress: BacktestProgress) -> None:
    if callback is None:
        return
    try:
        callback(progress)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Progress callback failed: %s", exc, exc_info=True)
