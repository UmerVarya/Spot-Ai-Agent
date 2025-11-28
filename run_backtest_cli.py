"""Headless CLI entrypoint to run Spot-AI backtests.

This script reuses the research backtest engine that powers the Streamlit
Backtest / Research Lab. It is designed for terminal usage (e.g. via tmux)
and will persist results to disk for later inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from backtest.data import load_csv_paths
from backtest.data_manager import ensure_ohlcv_csvs
from backtest.engine import BacktestConfig, run_backtest_from_csv_paths
from backtest.filesystem import (
    BacktestRunMetadata,
    build_backtest_id,
    build_backtest_output_paths,
    get_backtest_dir,
    write_csv_atomic,
    write_json_atomic,
)
from config import DEFAULT_MIN_PROB_FOR_TRADE
from trade_constants import TP1_TRAILING_ONLY_STRATEGY
from trade_schema import TRADE_HISTORY_COLUMNS
from backtest.types import BacktestProgress


BACKTEST_DIR = get_backtest_dir()


def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts


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


def _format_date_for_path(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _print_progress(progress: BacktestProgress) -> None:
    label = progress.phase.capitalize()
    if progress.message:
        label = f"{label}: {progress.message}"
    percent = progress.percent * 100 if progress.total else 0
    print(f"[{datetime.utcnow().isoformat()}] {label} {percent:5.1f}%", end="\r", file=sys.stderr)


def _ensure_trades_with_header(trades: pd.DataFrame) -> pd.DataFrame:
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


def _compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
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

    metrics: Dict[str, Any] = {
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
        per_symbol: Dict[str, Dict[str, float]] = {}

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

    def complete(self, metrics: Dict[str, float]) -> None:
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


def run_cli(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Spot-AI backtests from the command line.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to backtest, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (e.g. 1m, 5m, 1h)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade as a percent of equity")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Optional extra global score/confidence threshold; when omitted only per-symbol "
            "profile thresholds apply."
        ),
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=DEFAULT_MIN_PROB_FOR_TRADE,
        help=f"Minimum model probability threshold (default {DEFAULT_MIN_PROB_FOR_TRADE})",
    )
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Fee in basis points")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points")
    parser.add_argument("--atr-stop-multiplier", type=float, default=1.5, help="ATR stop multiplier")
    parser.add_argument(
        "--trade-size-usd",
        type=float,
        default=500.0,
        help="Fixed notional size per trade when using fixed_notional sizing",
    )
    parser.add_argument(
        "--sizing-mode",
        choices=["fixed_notional", "risk_pct"],
        default="fixed_notional",
        help="Position sizing mode",
    )
    parser.add_argument(
        "--exit-mode",
        choices=["tp_trailing", "atr_trailing"],
        default="tp_trailing",
        help="Exit behaviour (TP trailing vs ATR trailing)",
    )
    parser.add_argument("--latency-bars", type=int, default=0, help="Execution latency (bars)")
    parser.add_argument("--entry-delay-bars", type=int, default=0, help="Entry delay (bars)")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="Initial capital for backtest")
    parser.add_argument("--take-profit-strategy", default=TP1_TRAILING_ONLY_STRATEGY, help="Take profit strategy identifier")
    parser.add_argument("--skip-fraction", type=float, default=0.0, help="Randomly skip this fraction of signals")
    parser.add_argument("--random-seed", type=int, default=1337, help="Seed for any stochastic components")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing OHLCV CSVs")
    parser.add_argument("--csv-paths", nargs="*", default=None, help="Optional explicit CSV paths (skip download)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BACKTEST_DIR,
        help="Output directory for CSV/JSON results",
    )
    parser.add_argument("--backtest-id", help="Optional explicit backtest identifier for file naming")
    parser.add_argument("--run-label", default=None, help="Optional label or note for this run")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit without running")

    parsed = parser.parse_args(args=args)

    start_ts = _parse_date(parsed.start)
    end_input = _parse_date(parsed.end)
    if end_input <= start_ts:
        print(
            f"End date must be after start date (start: {start_ts}, end: {end_input})",
            file=sys.stderr,
        )
        return 1

    # Include the full end date by adding one day similar to the dashboard behaviour
    end_ts = end_input + pd.Timedelta(days=1)

    cfg = BacktestConfig(
        start_ts=start_ts,
        end_ts=end_ts,
        is_backtest=True,
        min_score=parsed.score_threshold,
        min_prob=parsed.min_prob,
        atr_mult_sl=parsed.atr_stop_multiplier,
        fee_bps=parsed.fee_bps,
        slippage_bps=parsed.slippage_bps,
        latency_bars=parsed.latency_bars,
        entry_delay_bars=parsed.entry_delay_bars,
        initial_capital=parsed.initial_capital,
        risk_per_trade_pct=parsed.risk,
        take_profit_strategy=parsed.take_profit_strategy,
        skip_fraction=parsed.skip_fraction,
        sizing_mode=parsed.sizing_mode,
        trade_size_usd=parsed.trade_size_usd,
        exit_mode=parsed.exit_mode,
        random_seed=parsed.random_seed,
    )

    print("Configured backtest:")
    print(json.dumps(cfg.__dict__, indent=2, default=_json_safe))
    print(f"Symbols: {parsed.symbols}")
    print(f"Timeframe: {parsed.timeframe}")
    print(f"Output directory: {parsed.out_dir}")
    if parsed.dry_run:
        return 0

    out_dir: Path = parsed.out_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        print(
            f"Cannot create output directory {out_dir!s}: {exc}. "
            "Use --out-dir to specify a writable location.",
            file=sys.stderr,
        )
        return 1

    params_dict: Dict[str, Any] = {
        "risk": parsed.risk,
        "score_threshold": parsed.score_threshold,
        "min_prob": parsed.min_prob,
        "fee_bps": parsed.fee_bps,
        "slippage_bps": parsed.slippage_bps,
        "atr_stop_multiplier": parsed.atr_stop_multiplier,
        "latency_bars": parsed.latency_bars,
        "entry_delay_bars": parsed.entry_delay_bars,
        "initial_capital": parsed.initial_capital,
        "take_profit_strategy": parsed.take_profit_strategy,
        "skip_fraction": parsed.skip_fraction,
        "sizing_mode": parsed.sizing_mode,
        "trade_size_usd": parsed.trade_size_usd,
        "exit_mode": parsed.exit_mode,
        "random_seed": parsed.random_seed,
    }

    backtest_id = parsed.backtest_id or build_backtest_id(
        parsed.symbols,
        parsed.timeframe,
        _format_date_for_path(start_ts),
        _format_date_for_path(end_ts - pd.Timedelta(days=1)),
    )
    paths = build_backtest_output_paths(backtest_id, out_dir)
    metadata = BacktestRunMetadata(
        backtest_id=backtest_id,
        symbols=[sym.upper() for sym in parsed.symbols],
        timeframe=parsed.timeframe,
        start_date=_format_date_for_path(start_ts),
        end_date=_format_date_for_path(end_ts - pd.Timedelta(days=1)),
        params=params_dict,
        label=parsed.run_label,
        random_seed=parsed.random_seed,
    )
    tracker = _MetaTracker(metadata, paths["meta"])
    tracker.start(0)

    try:
        if parsed.csv_paths:
            csv_paths = [Path(p) for p in parsed.csv_paths]
        else:
            csv_paths = ensure_ohlcv_csvs(
                parsed.symbols,
                parsed.timeframe,
                start_ts.to_pydatetime(),
                end_ts.to_pydatetime(),
                data_dir=parsed.data_dir,
            )
        data = load_csv_paths(csv_paths, start=start_ts, end=end_ts)
        total_bars = int(sum(len(df) for df in data.values()))
        tracker.start(total_bars)
        print(
            f"Running backtest {backtest_id} for {parsed.symbols} on {parsed.timeframe} "
            f"from {parsed.start} to {parsed.end}..."
        )

        def _progress(progress: BacktestProgress) -> None:
            tracker.update_progress(progress)
            _print_progress(progress)

        result = run_backtest_from_csv_paths(
            csv_paths,
            cfg,
            symbols=parsed.symbols,
            progress_callback=_progress,
        )
    except Exception as exc:  # pragma: no cover - smoke tested via integration
        tracker.fail(str(exc))
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 1

    trades = _ensure_trades_with_header(result.trades if hasattr(result, "trades") else pd.DataFrame())
    equity_curve = _ensure_equity_curve(result.equity_curve if hasattr(result, "equity_curve") else pd.DataFrame(), cfg)
    metrics = _compute_metrics(trades, equity_curve, cfg.initial_capital)

    write_csv_atomic(paths["trades"], trades)
    write_csv_atomic(paths["equity"], equity_curve)
    write_json_atomic(paths["metrics"], metrics)
    tracker.complete(metrics)

    if trades.empty:
        print("Backtest completed with 0 trades. Outputs have been saved for review.")
    else:
        print("\nBacktest complete.")

    print(f"Trades → {paths['trades']}")
    print(f"Equity → {paths['equity']}")
    print(f"Metrics → {paths['metrics']}")
    print(f"Meta → {paths['meta']}")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
