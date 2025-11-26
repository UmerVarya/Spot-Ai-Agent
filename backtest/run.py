from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.engine import BacktestConfig, BacktestResult, run_backtest_from_csv_paths
from backtest.data_manager import ensure_ohlcv_csvs
from backtest.types import ProgressCallback


def _coerce_timestamp(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts


def run_backtest(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    data_dir: Path = Path("data"),
) -> BacktestResult:
    """Run a single-symbol backtest using the research engine.

    This mirrors the Streamlit research lab behaviour by downloading/locating
    OHLCV data, building a :class:`BacktestConfig` with default parameters, and
    invoking :func:`run_backtest_from_csv_paths` with optional progress
    reporting.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    start_ts = _coerce_timestamp(start)
    end_ts = _coerce_timestamp(end)

    csv_paths = ensure_ohlcv_csvs(
        [symbol],
        timeframe,
        start_ts.to_pydatetime(),
        end_ts.to_pydatetime(),
        data_dir=data_dir,
    )

    cfg = BacktestConfig(
        start_ts=start_ts,
        end_ts=end_ts,
        is_backtest=True,
    )

    return run_backtest_from_csv_paths(
        csv_paths,
        cfg,
        symbols=[symbol],
        progress_callback=progress_callback,
    )
