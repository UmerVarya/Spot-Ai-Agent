from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict

import pandas as pd


def load_csv_folder(path_pattern: str, symbol_col_from_name: bool = True) -> Dict[str, pd.DataFrame]:
    """Load OHLCV CSV files into a dictionary keyed by symbol.

    The loader is intentionally tolerant of schema differences; if a file
    contains an ``open_time`` column it will be treated as the timestamp,
    otherwise a ``timestamp`` column is expected.
    """

    data: Dict[str, pd.DataFrame] = {}
    for fp in glob.glob(path_pattern):
        csv_path = Path(fp)
        if not csv_path.is_file():
            continue
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
        data[symbol or "UNKNOWN"] = df
    return data
