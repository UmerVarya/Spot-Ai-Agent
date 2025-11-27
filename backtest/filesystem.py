from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Sequence


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


@dataclass
class BacktestRunMetadata:
    backtest_id: str
    symbols: List[str]
    timeframe: str
    start_date: str
    end_date: str
    params: Dict[str, object]
    label: str | None = None
    random_seed: int | None = None
    status: str = "queued"
    progress: float = 0.0
    current_bar: int = 0
    total_bars: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None
    metrics_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_json_dump(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(_safe_json_dump(payload), encoding="utf-8")
    tmp_path.replace(path)


def write_csv_atomic(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def build_backtest_id(symbols: Sequence[str], timeframe: str, start_date: str, end_date: str) -> str:
    safe_symbols = "-".join(sym.upper() for sym in symbols)
    suffix = str(int(time.time()))
    return f"{safe_symbols}_{timeframe}_{start_date}_{end_date}_{suffix}"


def build_backtest_output_paths(backtest_id: str, backtest_dir: Path | None = None) -> Dict[str, Path]:
    base_dir = backtest_dir or get_backtest_dir()
    prefix = base_dir / backtest_id
    return {
        "trades": prefix.with_name(prefix.name + "_trades.csv"),
        "equity": prefix.with_name(prefix.name + "_equity.csv"),
        "metrics": prefix.with_name(prefix.name + "_metrics.json"),
        "meta": prefix.with_name(prefix.name + "_meta.json"),
    }


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


def _load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_run_metadata(path: Path) -> BacktestRunMetadata | None:
    payload = _load_json(path)
    if not payload:
        return None
    try:
        return BacktestRunMetadata(**payload)  # type: ignore[arg-type]
    except TypeError:
        return None


def _safe_get_metrics(path: Path) -> Dict[str, object]:
    payload = _load_json(path)
    metrics: Dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
        else:
            metrics[key] = value
    return metrics


def discover_backtest_runs(backtest_dir: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for meta_path in backtest_dir.glob("*_meta.json"):
        metadata = load_run_metadata(meta_path)
        if metadata is None:
            continue
        metrics_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_metrics.json"))
        metrics = metadata.metrics_summary or {}
        if metrics_path.exists():
            file_metrics = _safe_get_metrics(metrics_path)
            metrics = {**metrics, **file_metrics}
        row: Dict[str, object] = {
            **metadata.to_dict(),
            "metrics": metrics,
            "trades_path": meta_path.with_name(meta_path.name.replace("_meta.json", "_trades.csv")),
            "equity_path": meta_path.with_name(meta_path.name.replace("_meta.json", "_equity.csv")),
            "metrics_path": metrics_path,
            "meta_path": meta_path,
        }
        runs.append(row)
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return runs
