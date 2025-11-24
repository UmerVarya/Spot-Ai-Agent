from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

_DATA_CACHE: dict[Tuple, Dict[str, pd.DataFrame]] = {}


def _cache_key(
    paths: Iterable[Path],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    symbol_col_from_name: bool,
) -> Tuple:
    normalized_paths = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns)
        for path in (Path(p) for p in paths)
        if path.is_file()
    )
    start_key = pd.to_datetime(start, utc=True).isoformat() if start is not None else None
    end_key = pd.to_datetime(end, utc=True).isoformat() if end is not None else None
    return normalized_paths, start_key, end_key, bool(symbol_col_from_name)


def _load_single_csv(csv_path: Path, symbol_col_from_name: bool) -> tuple[str | None, pd.DataFrame]:
    if not csv_path.is_file():
        return None, pd.DataFrame()
    symbol = None
    if symbol_col_from_name:
        symbol = csv_path.stem.split("_")[0].upper()
    df = pd.read_csv(csv_path)
    time_col = "open_time" if "open_time" in df.columns else "timestamp"
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df.set_index(time_col, inplace=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.sort_index(inplace=True)
    return symbol, df


def _load_csv_collection(paths: Iterable[Path], symbol_col_from_name: bool = True) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for csv_path in paths:
        symbol, df = _load_single_csv(csv_path, symbol_col_from_name)
        if df.empty:
            continue
        data[symbol or "UNKNOWN"] = df
    return data


def _maybe_slice(data: Dict[str, pd.DataFrame], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Dict[str, pd.DataFrame]:
    if start is None and end is None:
        return data
    start_ts = pd.to_datetime(start, utc=True) if start is not None else None
    end_ts = pd.to_datetime(end, utc=True) if end is not None else None
    sliced: Dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        windowed = df
        if start_ts is not None or end_ts is not None:
            windowed = df.loc[start_ts:end_ts]
        sliced[symbol] = windowed
    return sliced


def load_csv_folder(
    path_pattern: str,
    symbol_col_from_name: bool = True,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV CSV files into a dictionary keyed by symbol.

    The loader is intentionally tolerant of schema differences; if a file
    contains an ``open_time`` column it will be treated as the timestamp,
    otherwise a ``timestamp`` column is expected.
    """

    paths = [Path(fp) for fp in glob.glob(path_pattern)]
    return load_csv_paths(paths, symbol_col_from_name, start=start, end=end, use_cache=use_cache)


def load_csv_paths(
    paths: list[Path],
    symbol_col_from_name: bool = True,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV CSVs from explicit paths with optional caching and slicing."""

    cache_key = _cache_key(paths, start, end, symbol_col_from_name) if use_cache else None
    if cache_key is not None and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    data = _load_csv_collection(paths, symbol_col_from_name)
    windowed = _maybe_slice(data, start, end)
    if cache_key is not None:
        _DATA_CACHE[cache_key] = windowed
    return windowed


def invalidate_cache_for_paths(paths: Iterable[Path]) -> None:
    """Remove cached entries that reference any of the supplied paths."""

    normalized = {str(Path(p).resolve()) for p in paths}
    keys_to_delete = []
    for key in _DATA_CACHE:
        path_specs = key[0] if key else []
        if any(path in normalized for path, _ in path_specs):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        _DATA_CACHE.pop(key, None)
