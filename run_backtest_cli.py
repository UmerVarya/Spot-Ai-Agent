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
from typing import Any, Sequence

import pandas as pd

from backtest.analysis import build_equity_curve
from backtest.data_manager import ensure_ohlcv_csvs
from backtest.engine import BacktestConfig, run_backtest_from_csv_paths
from trade_constants import TP1_TRAILING_ONLY_STRATEGY


BACKTEST_DIR = Path.home() / "spot_data" / "backtests"


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


def _write_symbol_outputs(
    symbol: str,
    trades: pd.DataFrame,
    cfg: BacktestConfig,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    out_dir: Path,
) -> dict[str, Path]:
    safe_symbol = symbol.upper()
    base_name = f"{safe_symbol}_{timeframe}_{_format_date_for_path(start_ts)}_{_format_date_for_path(end_ts)}"

    trades_path = out_dir / f"{base_name}_trades.csv"
    trades.to_csv(trades_path, index=False)

    equity_curve = build_equity_curve(trades, cfg.initial_capital)
    equity_path = out_dir / f"{base_name}_equity.csv"
    equity_curve.to_csv(equity_path, index=False)

    return {"trades": trades_path, "equity": equity_path}


def _build_summary(result) -> dict:
    summary: dict[str, Any] = {
        "overall": result.metrics,
        "symbols": {},
    }
    if isinstance(result.by_symbol, pd.DataFrame) and not result.by_symbol.empty:
        for row in result.by_symbol.to_dict(orient="records"):
            symbol = row.get("symbol", "UNKNOWN")
            summary["symbols"][symbol] = {k: v for k, v in row.items() if k != "symbol"}
    return summary


def _print_progress(progress) -> None:
    label = progress.phase.capitalize()
    if progress.message:
        label = f"{label}: {progress.message}"
    percent = progress.percent * 100 if progress.total else 0
    print(f"[{datetime.utcnow().isoformat()}] {label} {percent:5.1f}%", end="\r", file=sys.stderr)


def run_cli(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Spot-AI backtests from the command line.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to backtest, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (e.g. 1m, 5m, 1h)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade as a percent of equity")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="Minimum score threshold")
    parser.add_argument("--min-prob", type=float, default=0.55, help="Minimum model probability threshold")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Fee in basis points")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points")
    parser.add_argument("--atr-stop-multiplier", type=float, default=1.5, help="ATR stop multiplier")
    parser.add_argument("--latency-bars", type=int, default=0, help="Execution latency (bars)")
    parser.add_argument("--entry-delay-bars", type=int, default=0, help="Entry delay (bars)")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="Initial capital for backtest")
    parser.add_argument("--take-profit-strategy", default=TP1_TRAILING_ONLY_STRATEGY, help="Take profit strategy identifier")
    parser.add_argument("--skip-fraction", type=float, default=0.0, help="Randomly skip this fraction of signals")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing OHLCV CSVs")
    parser.add_argument("--csv-paths", nargs="*", default=None, help="Optional explicit CSV paths (skip download)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BACKTEST_DIR,
        help="Output directory for CSV/JSON results",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit without running")

    parsed = parser.parse_args(args=args)

    start_ts = _parse_date(parsed.start)
    # Include the full end date by adding one day similar to the dashboard behaviour
    end_ts = _parse_date(parsed.end) + pd.Timedelta(days=1)

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
        print(f"Running backtest for {parsed.symbols} on {parsed.timeframe} from {parsed.start} to {parsed.end}...")
        result = run_backtest_from_csv_paths(
            csv_paths,
            cfg,
            symbols=parsed.symbols,
            progress_callback=_print_progress,
        )
    except Exception as exc:  # pragma: no cover - smoke tested via integration
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 1

    if result.trades.empty:
        print("No trades generated for the given parameters.")

    written: list[dict[str, Path]] = []
    for sym in parsed.symbols:
        sym_trades = pd.DataFrame()
        if not result.trades.empty and "symbol" in result.trades.columns:
            sym_mask = result.trades["symbol"].str.upper() == sym.upper()
            sym_trades = result.trades[sym_mask]
        if sym_trades.empty:
            continue
        paths = _write_symbol_outputs(sym, sym_trades, cfg, parsed.timeframe, start_ts, end_ts - pd.Timedelta(days=1), out_dir)
        written.append({"symbol": sym, **paths})

    summary = _build_summary(result)
    summary_path = out_dir / f"summary_{parsed.timeframe}_{_format_date_for_path(start_ts)}_{_format_date_for_path(end_ts - pd.Timedelta(days=1))}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_safe))

    print("\nBacktest complete.")
    for item in written:
        print(f"{item['symbol']}: trades → {item['trades']}, equity → {item['equity']}")
    print(f"Summary → {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
