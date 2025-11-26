from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class BacktestFile:
    path: Path
    symbol: str
    timeframe: str
    start: str
    end: str
    kind: str


def get_backtest_dir() -> Path:
    base = os.getenv("BACKTESTS_DIR", "/home/ubuntu/spot_data/backtests")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_backtest_filename(path: Path) -> BacktestFile | None:
    name = path.name
    if not name.lower().endswith(".csv"):
        return None
    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    symbol, timeframe, start, end, kind = parts
    return BacktestFile(path=path, symbol=symbol, timeframe=timeframe, start=start, end=end, kind=kind)


def discover_backtest_files(backtest_dir: Path) -> List[BacktestFile]:
    files: List[BacktestFile] = []
    for path in sorted(backtest_dir.glob("*.csv")):
        try:
            parsed = _parse_backtest_filename(path)
        except Exception:
            parsed = None
        if parsed is None:
            continue
        files.append(parsed)

    files.sort(key=lambda f: f.end, reverse=True)
    return files
