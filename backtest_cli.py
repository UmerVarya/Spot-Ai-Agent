from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from backtest import presets as preset_mod
from backtest.filesystem import build_backtest_id, get_backtest_dir
from backtest.run import launch_backtest
from config import DEFAULT_MIN_PROB_FOR_TRADE


DEFAULT_OUTPUT_DIR = get_backtest_dir()


def _parse_datetime(value: str) -> datetime:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def _parse_symbols(raw_symbols: str | None, legacy_symbol: str | None) -> list[str]:
    symbols: list[str] = []
    if raw_symbols:
        symbols.extend([s.strip().upper() for s in raw_symbols.split(",") if s.strip()])
    if not symbols and legacy_symbol:
        symbols.append(legacy_symbol.strip().upper())
    if not symbols:
        raise argparse.ArgumentTypeError("Provide --symbols (comma-separated) or --symbol.")
    return symbols


def _resolve_date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if args.start and args.end:
        start = _parse_datetime(args.start)
        end_inclusive = _parse_datetime(args.end)
    elif args.months:
        end_inclusive = _parse_datetime(args.end) if args.end else now
        start_ts = pd.Timestamp(end_inclusive) - pd.DateOffset(months=args.months)
        start = start_ts.to_pydatetime()
    else:
        raise ValueError("Provide either --start/--end or --months to define the date range.")

    if start > end_inclusive:
        raise ValueError("Start date must be before end date.")

    end_exclusive = end_inclusive + timedelta(days=1)
    return start, end_exclusive


def _print_summary(
    *,
    backtest_id: str,
    symbols: Iterable[str],
    timeframe: str,
    start: datetime,
    end_exclusive: datetime,
    score_threshold: Optional[float],
    min_prob: float,
    exit_mode: str,
    trade_size_usd: float,
    fee_bps: float,
    slippage_bps: float,
) -> None:
    end_inclusive = end_exclusive - timedelta(days=1)
    print("Starting backtest:")
    print(f"  symbols: {','.join(symbols)}")
    print(f"  timeframe: {timeframe}")
    print(f"  start: {start.date()}")
    print(f"  end: {end_inclusive.date()}")
    threshold_display = score_threshold if score_threshold is not None else "disabled"
    print(f"  score_threshold: {threshold_display}")
    print(f"  min_prob: {min_prob}")
    print(f"  exit_mode: {exit_mode}")
    print(f"  trade_size_usd: {trade_size_usd}")
    print(f"  fee_bps: {fee_bps}")
    print(f"  slippage_bps: {slippage_bps}")
    print(f"  backtest_id: {backtest_id}")


def main(cli_args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Spot-AI research backtests from the CLI.")
    parser.add_argument("--symbols", help="Comma-separated Binance symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--symbol", help="Legacy single symbol flag")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g. 1m)")
    parser.add_argument("--months", type=int, help="Months back from now")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", help="Output directory for artifacts")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Optional extra global score/confidence threshold; by default no extra gate "
            "is applied beyond per-symbol/tier profiles."
        ),
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=DEFAULT_MIN_PROB_FOR_TRADE,
        help=f"Minimum model probability threshold (default {DEFAULT_MIN_PROB_FOR_TRADE})",
    )
    parser.add_argument("--exit-mode", choices=["tp_trailing", "atr_trailing"], default="tp_trailing")
    parser.add_argument("--trade-size-usd", type=float, default=500.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--preset",
        choices=list(preset_mod.list_presets()),
        default=None,
        help=f"Optional performance preset (default {preset_mod.DEFAULT_PRESET_NAME})",
    )

    args = parser.parse_args(cli_args)

    try:
        symbols = _parse_symbols(args.symbols, args.symbol)
        start, end = _resolve_date_range(args)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    backtest_id = build_backtest_id(
        symbols,
        args.timeframe,
        pd.Timestamp(start).strftime("%Y-%m-%d"),
        (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    _print_summary(
        backtest_id=backtest_id,
        symbols=symbols,
        timeframe=args.timeframe,
        start=start,
        end_exclusive=end,
        score_threshold=args.score_threshold,
        min_prob=args.min_prob,
        exit_mode=args.exit_mode,
        trade_size_usd=args.trade_size_usd,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    try:
        launch_backtest(
            symbols=symbols,
            timeframe=args.timeframe,
            start=start,
            end=end,
            score_threshold=args.score_threshold,
            min_prob=args.min_prob,
            trade_size_usd=args.trade_size_usd,
            preset_name=args.preset or preset_mod.DEFAULT_PRESET_NAME,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            sizing_mode="fixed_notional",
            exit_mode=args.exit_mode,
            random_seed=args.random_seed,
            backtest_id=backtest_id,
            out_dir=output_dir,
            data_dir=Path("data"),
        )
    except Exception as exc:  # pragma: no cover - surface error to user
        print(f"Backtest failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
