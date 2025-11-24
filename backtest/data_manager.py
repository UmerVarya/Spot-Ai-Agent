from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from log_utils import setup_logger
from symbol_mapper import map_symbol_for_binance
from trade_utils import _get_binance_client
from backtest.data import invalidate_cache_for_paths

logger = setup_logger(__name__)

_INTERVAL_TO_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def _normalize_ts(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _fetch_klines(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = _get_binance_client()
    if client is None:
        raise RuntimeError("Binance client unavailable; cannot fetch historical klines.")

    mapped_symbol = map_symbol_for_binance(symbol.upper())
    start_ts = int(_normalize_ts(start).timestamp() * 1000)
    end_ts = int(_normalize_ts(end).timestamp() * 1000)

    logger.info(
        "Downloading %s OHLCV for %s from %s to %s",
        interval,
        mapped_symbol,
        pd.to_datetime(start_ts, unit="ms", utc=True),
        pd.to_datetime(end_ts, unit="ms", utc=True),
    )

    klines = client.get_historical_klines(
        symbol=mapped_symbol,
        interval=interval,
        start_str=start_ts,
        end_str=end_ts,
    )

    if not klines:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])

    df = pd.DataFrame(
        klines,
        columns=[
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
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.rename(columns={"quote_asset_volume": "quote_volume"}, inplace=True)
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].dropna()


def _load_existing_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])

    df = pd.read_csv(csv_path)
    time_col = "timestamp" if "timestamp" in df.columns else "open_time"
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df.rename(columns={time_col: "timestamp"}, inplace=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp")
    return df


def _merge_and_save(csv_path: Path, *dfs: Iterable[pd.DataFrame]) -> Path:
    combined = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    if combined.empty:
        return csv_path
    combined = combined.drop_duplicates(subset=["timestamp"])
    combined = combined.sort_values("timestamp")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return csv_path


def ensure_ohlcv_csvs(
    symbols: List[str],
    interval: str,
    start: datetime,
    end: datetime,
    data_dir: Path = Path("data"),
) -> List[Path]:
    """
    Ensure OHLCV CSVs exist for the requested symbols and cover ``[start, end]``.
    """

    start = _normalize_ts(start)
    end = _normalize_ts(end)
    if end <= start:
        raise ValueError("End timestamp must be after start timestamp")

    interval_delta = _INTERVAL_TO_DELTA.get(interval)
    csv_paths: List[Path] = []

    for symbol in symbols:
        csv_path = data_dir / f"{symbol.upper()}_{interval}.csv"
        existing = _load_existing_csv(csv_path)
        existing_start = existing["timestamp"].min() if not existing.empty else None
        existing_end = existing["timestamp"].max() if not existing.empty else None

        downloads: List[pd.DataFrame] = []

        if existing_start is None or existing_end is None:
            downloads.append(_fetch_klines(symbol, interval, start, end))
        elif start >= existing_start and end <= existing_end:
            logger.info(
                "Using cached OHLCV for %s %s; covers [%s, %s]",
                symbol.upper(),
                interval,
                existing_start,
                existing_end,
            )
        else:
            if existing_start > start and interval_delta is not None:
                head_end = existing_start - interval_delta
                if head_end >= start:
                    logger.info(
                        "Extending cached OHLCV for %s %s from %s back to %s",
                        symbol.upper(),
                        interval,
                        existing_start,
                        start,
                    )
                    downloads.append(_fetch_klines(symbol, interval, start, head_end))
            if existing_end < end:
                tail_start = existing_end + (interval_delta or pd.Timedelta(seconds=1))
                if tail_start < end:
                    logger.info(
                        "Extending cached OHLCV for %s %s from %s to %s",
                        symbol.upper(),
                        interval,
                        existing_end,
                        end,
                    )
                    downloads.append(_fetch_klines(symbol, interval, tail_start, end))

        if downloads:
            new_rows = sum(len(df) for df in downloads)
            logger.info(
                "Appending %d new candles to %s",
                new_rows,
                csv_path,
            )
            existing = pd.concat([existing] + downloads, ignore_index=True)
            _merge_and_save(csv_path, existing)
            invalidate_cache_for_paths([csv_path])
        else:
            _merge_and_save(csv_path, existing)
        csv_paths.append(csv_path)

    return csv_paths
