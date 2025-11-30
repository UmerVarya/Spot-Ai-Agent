"""Headless CLI entrypoint to run Spot-AI backtests.

This script reuses the research backtest engine that powers the Streamlit
Backtest / Research Lab. It is designed for terminal usage (e.g. via tmux)
and will persist results to disk for later inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest import presets as preset_mod
from backtest.filesystem import get_backtest_dir
from backtest.run import _coerce_timestamp, _json_safe, launch_backtest
from backtest.types import BacktestProgress
from config import DEFAULT_MIN_PROB_FOR_TRADE
from trade_constants import TP1_TRAILING_ONLY_STRATEGY


BACKTEST_DIR = get_backtest_dir()


def _parse_date(value: str) -> pd.Timestamp:
    return _coerce_timestamp(value)


def _print_progress(progress: BacktestProgress) -> None:
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
        "--preset",
        default=None,
        help=(
            "Performance/observability preset. "
            f"Choices: {', '.join(preset_mod.list_presets())}. "
            f"Default: {preset_mod.DEFAULT_PRESET_NAME}"
        ),
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

    preset_cfg_name = parsed.preset or getattr(preset_mod, "DEFAULT_PRESET_NAME", "Standard research")
    try:
        preset_mod.get_preset(preset_cfg_name)
    except KeyError:
        valid = ", ".join(preset_mod.list_presets())
        raise SystemExit(f"Unknown preset: {preset_cfg_name!r}. Expected one of: {valid}")

    launch_params: dict[str, Any] = {
        "symbols": parsed.symbols,
        "timeframe": parsed.timeframe,
        "start": start_ts,
        "end": end_ts,
        "risk": parsed.risk,
        "score_threshold": parsed.score_threshold,
        "min_prob": parsed.min_prob,
        "trade_size_usd": parsed.trade_size_usd,
        "preset_name": preset_cfg_name,
        "fee_bps": parsed.fee_bps,
        "slippage_bps": parsed.slippage_bps,
        "atr_stop_multiplier": parsed.atr_stop_multiplier,
        "sizing_mode": parsed.sizing_mode,
        "exit_mode": parsed.exit_mode,
        "latency_bars": parsed.latency_bars,
        "entry_delay_bars": parsed.entry_delay_bars,
        "initial_capital": parsed.initial_capital,
        "skip_fraction": parsed.skip_fraction,
        "random_seed": parsed.random_seed,
        "run_label": parsed.run_label,
        "backtest_id": parsed.backtest_id,
        "data_dir": parsed.data_dir,
        "out_dir": parsed.out_dir,
        "csv_paths": parsed.csv_paths,
        "take_profit_strategy": parsed.take_profit_strategy,
        "progress_callback": _print_progress,
    }

    cfg_preview: dict[str, Any] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "is_backtest": True,
        "min_score": parsed.score_threshold,
        "min_prob": parsed.min_prob,
        "atr_mult_sl": parsed.atr_stop_multiplier,
        "fee_bps": parsed.fee_bps,
        "slippage_bps": parsed.slippage_bps,
        "latency_bars": parsed.latency_bars,
        "entry_delay_bars": parsed.entry_delay_bars,
        "initial_capital": parsed.initial_capital,
        "risk_per_trade_pct": parsed.risk,
        "take_profit_strategy": parsed.take_profit_strategy,
        "skip_fraction": parsed.skip_fraction,
        "sizing_mode": parsed.sizing_mode,
        "trade_size_usd": parsed.trade_size_usd,
        "exit_mode": parsed.exit_mode,
        "random_seed": parsed.random_seed,
    }

    print("Configured backtest:")
    print(json.dumps(cfg_preview, indent=2, default=_json_safe))
    print(f"Symbols: {parsed.symbols}")
    print(f"Timeframe: {parsed.timeframe}")
    print(f"Preset: {preset_cfg_name}")
    print(f"Output directory: {parsed.out_dir}")
    if parsed.dry_run:
        launch_backtest(**{**launch_params, "dry_run": True})
        return 0

    try:
        result = launch_backtest(**{**launch_params, "dry_run": False})
    except Exception as exc:  # pragma: no cover - surfaced to CLI user
        print(f"Backtest failed: {exc}", file=sys.stderr)
        return 1

    paths = result["paths"]
    print("\nBacktest complete.")
    print(f"Trades → {paths['trades']}")
    print(f"Equity → {paths['equity']}")
    print(f"Metrics → {paths['metrics']}")
    print(f"Meta → {paths['meta']}")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
