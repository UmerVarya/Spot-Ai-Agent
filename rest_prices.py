"""Helpers for deterministic REST-based kline fetching.

These utilities provide a thin wrapper around the Binance REST API that
returns candles as ``pandas`` data frames ready for indicator
calculation.  They are intentionally synchronous and lightweight so they
can be used during RealTimeSignalCache warm-up phases where we only need
short contiguous windows of historical data.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "REQUIRED_MIN_BARS",
    "WARMUP_BARS",
    "rest_backfill_klines",
    "rest_fetch_latest_closed",
]


REQUIRED_MIN_BARS = int(os.getenv("RTSC_REQUIRED_MIN_BARS", "220"))
WARMUP_BARS = int(os.getenv("RTSC_REST_WARMUP_BARS", "300"))

_KLINE_COLUMNS: Sequence[str] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)


def _as_dataframe(raw: Iterable[Sequence[object]]) -> pd.DataFrame:
    """Shape raw kline payloads into a standardised DataFrame."""

    df = pd.DataFrame(list(raw), columns=list(_KLINE_COLUMNS))
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("close_time").reset_index(drop=True)
    df = df.set_index("close_time")
    df.index.name = "close_time"
    return df[numeric_cols]


def rest_backfill_klines(
    binance, symbol: str, interval: str = "1m", bars: int = WARMUP_BARS
) -> pd.DataFrame:
    """Return a contiguous block of historical klines for ``symbol``."""

    raw = binance.get_klines(symbol=symbol, interval=interval, limit=int(bars))
    return _as_dataframe(raw)


def rest_fetch_latest_closed(
    binance, symbol: str, interval: str = "1m"
) -> pd.DataFrame:
    """Return the latest closed candle (and one prior as guard)."""

    raw = binance.get_klines(symbol=symbol, interval=interval, limit=2)
    return _as_dataframe(raw)
