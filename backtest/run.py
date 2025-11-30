from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from backtest import presets as preset_mod
from backtest.data import load_csv_paths
from backtest.data_manager import ensure_ohlcv_csvs
from backtest.engine import BacktestConfig, BacktestResult, run_backtest_from_csv_paths
from backtest.filesystem import (
    BacktestRunMetadata,
    build_backtest_id,
    build_backtest_output_paths,
    get_backtest_dir,
    write_csv_atomic,
    write_json_atomic,
)
from backtest.types import BacktestProgress, ProgressCallback


def _coerce_timestamp(dt: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts


def _format_date_for_path(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, (pd.DataFrame,)):
        return value.to_dict(orient="records")
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _ensure_trades_with_header(trades: pd.DataFrame) -> pd.DataFrame:
    from trade_schema import TRADE_HISTORY_COLUMNS

    if not trades.empty:
        return trades
    return pd.DataFrame(columns=list(TRADE_HISTORY_COLUMNS))


def _ensure_equity_curve(equity: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if not equity.empty:
        return equity
    start = cfg.start_ts or pd.Timestamp.utcnow()
    end = (cfg.end_ts or start) - pd.Timedelta(days=1)
    timestamps = [start, end] if end > start else [start]
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "equity": [cfg.initial_capital for _ in timestamps],
        }
    )
    base["peak_equity"] = base["equity"]
    base["drawdown_pct"] = 0.0
    base["drawdown"] = 0.0
    return base


def _compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    pnl_col = "net_pnl_quote" if "net_pnl_quote" in trades.columns else "pnl"
    pnl_series = trades.get(pnl_col, pd.Series(dtype=float)).astype(float) if not trades.empty else pd.Series(dtype=float)
    total_trades = int(len(trades))
    winning = int((pnl_series > 0).sum()) if total_trades else 0
    losing = int((pnl_series < 0).sum()) if total_trades else 0
    net_pnl = float(pnl_series.sum()) if total_trades else 0.0
    gross_pnl = net_pnl
    r_multiple = trades.get("r_multiple", pd.Series(dtype=float)).astype(float) if not trades.empty else pd.Series(dtype=float)
    avg_r_multiple = float(np.nanmean(r_multiple)) if len(r_multiple) else 0.0
    expectancy_r = float(r_multiple.mean()) if len(r_multiple) else 0.0
    if not np.isfinite(avg_r_multiple):
        avg_r_multiple = 0.0
    if not np.isfinite(expectancy_r):
        expectancy_r = 0.0

    drawdown_pct = equity.get("drawdown_pct") if not equity.empty else pd.Series(dtype=float)
    dd_min = drawdown_pct.min() if drawdown_pct is not None and len(drawdown_pct) else 0.0
    if not np.isfinite(dd_min):
        dd_min = 0.0
    max_drawdown = float(abs(dd_min))

    equity_series = equity.get("equity", pd.Series(dtype=float)).astype(float) if not equity.empty else pd.Series(dtype=float)
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = float(np.sqrt(252) * returns.mean() / (returns.std(ddof=1) or 1e-9)) if not returns.empty else 0.0
    max_dd_duration = 0
    if not equity.empty and "drawdown_pct" in equity.columns:
        dd = equity["drawdown_pct"].values
        current = 0
        for value in dd:
            if value < 0:
                current += 1
                max_dd_duration = max(max_dd_duration, current)
            else:
                current = 0

    calmar_ratio = 0.0
    if drawdown_pct is not None and len(drawdown_pct):
        total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital if len(equity_series) else 0.0
        denom = abs(dd_min) if abs(dd_min) > 0 else 1e-9
        calmar_ratio = float(total_return / denom)

    winrate = float(winning / total_trades) if total_trades else 0.0

    metrics: dict[str, Any] = {
        "total_trades": float(total_trades),
        "winning_trades": float(winning),
        "losing_trades": float(losing),
        "winrate": winrate,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": float(max_dd_duration),
        "avg_r_multiple": avg_r_multiple,
        "expectancy_r": expectancy_r,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": float(calmar_ratio),
    }

    if not trades.empty:
        pnl_col = "net_pnl_quote" if "net_pnl_quote" in trades.columns else "pnl"
        per_symbol: dict[str, dict[str, float]] = {}

        def _symbol_drawdown(pnl: pd.Series) -> float:
            equity_track = initial_capital + pnl.cumsum()
            if equity_track.empty:
                return 0.0
            peak = equity_track.cummax()
            dd = (equity_track - peak) / peak
            worst = float(dd.min()) if not dd.empty else 0.0
            if not np.isfinite(worst):
                return 0.0
            return abs(worst)

        for symbol, group in trades.groupby("symbol"):
            pnl_series = group[pnl_col].astype(float)
            total = int(len(group))
            wins = int((pnl_series > 0).sum()) if total else 0
            losses = int((pnl_series < 0).sum()) if total else 0
            r_mult = group.get("r_multiple", pd.Series(dtype=float)).astype(float)
            avg_r_sym = float(np.nanmean(r_mult)) if len(r_mult) else 0.0
            if not np.isfinite(avg_r_sym):
                avg_r_sym = 0.0
            per_symbol[symbol] = {
                "total_trades": float(total),
                "winning_trades": float(wins),
                "losing_trades": float(losses),
                "winrate": float(wins / total) if total else 0.0,
                "net_pnl": float(pnl_series.sum()) if total else 0.0,
                "avg_r_multiple": avg_r_sym,
                "max_drawdown": _symbol_drawdown(pnl_series),
            }
        metrics["per_symbol"] = per_symbol

    return metrics


class _MetaTracker:
    def __init__(self, metadata: BacktestRunMetadata, meta_path: Path, update_every: int = 5_000) -> None:
        self.metadata = metadata
        self.meta_path = meta_path
        self.last_written = -1
        self.update_every = update_every

    def write(self) -> None:
        write_json_atomic(self.meta_path, self.metadata.to_dict())

    def start(self, total_bars: int) -> None:
        self.metadata.total_bars = int(total_bars)
        self.metadata.status = "running"
        self.metadata.started_at = datetime.utcnow().isoformat()
        self.write()

    def update_progress(self, progress: BacktestProgress) -> None:
        self.metadata.total_bars = max(int(progress.total or 0), self.metadata.total_bars)
        self.metadata.current_bar = int(progress.current or 0)
        if self.metadata.total_bars:
            self.metadata.progress = min(1.0, self.metadata.current_bar / float(self.metadata.total_bars or 1))
        else:
            self.metadata.progress = 0.0
        if self.metadata.current_bar == 0:
            return
        if self.metadata.current_bar == self.metadata.total_bars or self.metadata.current_bar - self.last_written >= self.update_every:
            self.last_written = self.metadata.current_bar
            self.metadata.status = "running"
            self.write()

    def complete(self, metrics: dict[str, float]) -> None:
        self.metadata.status = "completed"
        self.metadata.progress = 1.0
        self.metadata.finished_at = datetime.utcnow().isoformat()
        self.metadata.metrics_summary = {
            "total_trades": metrics.get("total_trades", 0.0),
            "winrate": metrics.get("winrate", 0.0),
            "net_pnl": metrics.get("net_pnl", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
        }
        self.write()

    def fail(self, message: str) -> None:
        self.metadata.status = "error"
        self.metadata.error_message = message
        self.metadata.finished_at = datetime.utcnow().isoformat()
        if self.metadata.total_bars:
            self.metadata.progress = min(1.0, self.metadata.current_bar / float(self.metadata.total_bars))
        self.write()


def _build_backtest_config(
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    risk: float | None,
    score_threshold: float | None,
    min_prob: float,
    trade_size_usd: float,
    fee_bps: float,
    slippage_bps: float,
    atr_stop_multiplier: float,
    sizing_mode: str,
    exit_mode: str,
    latency_bars: int,
    entry_delay_bars: int,
    initial_capital: float,
    skip_fraction: float,
    random_seed: int | None,
    take_profit_strategy: str | None,
) -> BacktestConfig:
    return BacktestConfig(
        start_ts=start_ts,
        end_ts=end_ts,
        is_backtest=True,
        min_score=score_threshold,
        min_prob=min_prob,
        atr_mult_sl=atr_stop_multiplier,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        latency_bars=latency_bars,
        entry_delay_bars=entry_delay_bars,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk,
        take_profit_strategy=take_profit_strategy,
        skip_fraction=skip_fraction,
        sizing_mode=sizing_mode,
        trade_size_usd=trade_size_usd,
        exit_mode=exit_mode,
        random_seed=random_seed,
    )


def launch_backtest(
    symbols: Sequence[str],
    timeframe: str,
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    risk: float | None = 1.0,
    score_threshold: float | None = None,
    min_prob: float = 0.0,
    trade_size_usd: float = 500.0,
    preset_name: str | None = None,
    *,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
    atr_stop_multiplier: float = 1.5,
    sizing_mode: str = "fixed_notional",
    exit_mode: str = "tp_trailing",
    latency_bars: int = 0,
    entry_delay_bars: int = 0,
    initial_capital: float = 10_000.0,
    skip_fraction: float = 0.0,
    random_seed: int | None = None,
    run_label: str | None = None,
    backtest_id: str | None = None,
    data_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    csv_paths: Iterable[str | Path] | None = None,
    dry_run: bool = False,
    take_profit_strategy: str | None = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Canonical entrypoint for launching a backtest job.

    This helper centralises preset resolution, BacktestConfig construction,
    metadata writing, and engine invocation so all callers share identical
    behaviour.
    """

    start_ts = _coerce_timestamp(start)
    end_ts = _coerce_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError(f"End timestamp {end_ts} must be after start {start_ts}")

    preset_cfg = preset_mod.resolve_preset(preset_name)

    resolved_symbols = [sym.upper() for sym in symbols]
    resolved_out_dir = Path(out_dir) if out_dir else get_backtest_dir()
    resolved_data_dir = Path(data_dir) if data_dir else Path("data")

    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    effective_backtest_id = backtest_id or build_backtest_id(
        resolved_symbols,
        timeframe,
        _format_date_for_path(start_ts),
        _format_date_for_path(end_ts - pd.Timedelta(days=1)),
    )

    paths = build_backtest_output_paths(effective_backtest_id, resolved_out_dir)

    if dry_run:
        return {"backtest_id": effective_backtest_id, "paths": paths, "preset": preset_cfg.name}

    cfg = _build_backtest_config(
        start_ts=start_ts,
        end_ts=end_ts,
        risk=risk,
        score_threshold=score_threshold,
        min_prob=min_prob,
        trade_size_usd=trade_size_usd,
        fee_bps=fee_bps or 0.0,
        slippage_bps=slippage_bps or 0.0,
        atr_stop_multiplier=atr_stop_multiplier,
        sizing_mode=sizing_mode,
        exit_mode=exit_mode,
        latency_bars=latency_bars,
        entry_delay_bars=entry_delay_bars,
        initial_capital=initial_capital,
        skip_fraction=skip_fraction,
        random_seed=random_seed,
        take_profit_strategy=take_profit_strategy,
    )

    params_dict: dict[str, Any] = {
        "risk": risk,
        "score_threshold": score_threshold,
        "min_prob": min_prob,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "atr_stop_multiplier": atr_stop_multiplier,
        "latency_bars": latency_bars,
        "entry_delay_bars": entry_delay_bars,
        "initial_capital": initial_capital,
        "take_profit_strategy": take_profit_strategy,
        "skip_fraction": skip_fraction,
        "sizing_mode": sizing_mode,
        "trade_size_usd": trade_size_usd,
        "exit_mode": exit_mode,
        "random_seed": random_seed,
        "preset": preset_cfg.name,
    }

    metadata = BacktestRunMetadata(
        backtest_id=effective_backtest_id,
        symbols=resolved_symbols,
        timeframe=timeframe,
        start_date=_format_date_for_path(start_ts),
        end_date=_format_date_for_path(end_ts - pd.Timedelta(days=1)),
        params=params_dict,
        label=run_label,
        random_seed=random_seed,
    )
    tracker = _MetaTracker(metadata, paths["meta"])
    tracker.start(0)

    try:
        resolved_csv_paths: list[Path]
        if csv_paths:
            resolved_csv_paths = [Path(p) for p in csv_paths]
        else:
            resolved_csv_paths = ensure_ohlcv_csvs(
                resolved_symbols,
                timeframe,
                start_ts.to_pydatetime(),
                end_ts.to_pydatetime(),
                data_dir=resolved_data_dir,
            )

        data_frames = load_csv_paths(resolved_csv_paths, start=start_ts, end=end_ts)
        total_bars = int(sum(len(df) for df in data_frames.values()))
        tracker.start(total_bars)

        def _progress(progress: BacktestProgress) -> None:
            tracker.update_progress(progress)
            if progress_callback:
                progress_callback(progress)

        result: BacktestResult = run_backtest_from_csv_paths(
            resolved_csv_paths,
            cfg,
            symbols=resolved_symbols,
            progress_callback=_progress,
            preset=preset_cfg,
        )
    except Exception as exc:  # pragma: no cover - surfaced via caller
        tracker.fail(str(exc))
        raise

    trades = _ensure_trades_with_header(result.trades if hasattr(result, "trades") else pd.DataFrame())
    equity_curve = _ensure_equity_curve(result.equity_curve if hasattr(result, "equity_curve") else pd.DataFrame(), cfg)
    metrics = _compute_metrics(trades, equity_curve, cfg.initial_capital)

    write_csv_atomic(paths["trades"], trades)
    write_csv_atomic(paths["equity"], equity_curve)
    write_json_atomic(paths["metrics"], metrics)
    tracker.complete(metrics)

    return {
        "backtest_id": effective_backtest_id,
        "paths": paths,
        "metadata": metadata.to_dict(),
        "metrics": metrics,
        "config": json.loads(json.dumps(cfg.__dict__, default=_json_safe)),
        "preset": preset_cfg.name,
    }


def run_backtest(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    data_dir: Path = Path("data"),
    preset: str | None = None,
) -> BacktestResult:
    """Run a single-symbol backtest using the research engine."""

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
        preset=preset,
    )
