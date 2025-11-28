from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from backtest.engine import BacktestConfig, BacktestResult, run_backtest_from_csv_paths
from backtest.data_manager import ensure_ohlcv_csvs
from backtest.filesystem import (
    BacktestRunMetadata,
    build_backtest_id,
    build_backtest_output_paths,
    get_backtest_dir,
    write_csv_atomic,
    write_json_atomic,
)
from config import DEFAULT_MIN_PROB_FOR_TRADE
from backtest.types import BacktestProgress
from run_backtest_cli import (
    _compute_metrics,
    _ensure_equity_curve,
    _ensure_trades_with_header,
    _format_date_for_path,
    _MetaTracker,
    _print_progress,
)


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
    score_threshold: float,
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
    print(f"  score_threshold: {score_threshold}")
    print(f"  min_prob: {min_prob}")
    print(f"  exit_mode: {exit_mode}")
    print(f"  trade_size_usd: {trade_size_usd}")
    print(f"  fee_bps: {fee_bps}")
    print(f"  slippage_bps: {slippage_bps}")
    print(f"  backtest_id: {backtest_id}")


def _run_backtest(
    *,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    cfg: BacktestConfig,
    backtest_id: str,
    output_dir: Path,
    data_dir: Path,
) -> BacktestRunMetadata:
    paths = build_backtest_output_paths(backtest_id, output_dir)
    params_dict: dict[str, Any] = {
        "score_threshold": cfg.min_score,
        "min_prob": cfg.min_prob,
        "exit_mode": cfg.exit_mode,
        "trade_size_usd": cfg.trade_size_usd,
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "random_seed": cfg.random_seed,
    }

    metadata = BacktestRunMetadata(
        backtest_id=backtest_id,
        symbols=[sym.upper() for sym in symbols],
        timeframe=timeframe,
        start_date=_format_date_for_path(pd.Timestamp(start)),
        end_date=_format_date_for_path(pd.Timestamp(end) - pd.Timedelta(days=1)),
        params=params_dict,
        random_seed=cfg.random_seed,
    )
    tracker = _MetaTracker(metadata, paths["meta"])
    tracker.start(0)

    try:
        csv_paths = ensure_ohlcv_csvs(symbols, timeframe, start, end, data_dir=data_dir)
    except Exception as exc:  # pragma: no cover - surfaced to user
        tracker.fail(str(exc))
        raise

    try:
        total_bars = 0
        try:
            from backtest.data import load_csv_paths

            data_frames = load_csv_paths(csv_paths, start=pd.Timestamp(start), end=pd.Timestamp(end))
            total_bars = int(sum(len(df) for df in data_frames.values()))
        except Exception:
            data_frames = None
        tracker.start(total_bars)

        def _progress(progress: BacktestProgress) -> None:
            tracker.update_progress(progress)
            _print_progress(progress)

        result: BacktestResult = run_backtest_from_csv_paths(
            csv_paths, cfg, symbols=symbols, progress_callback=_progress
        )
    except Exception as exc:  # pragma: no cover - surfaced to user
        tracker.fail(str(exc))
        raise

    trades = _ensure_trades_with_header(result.trades if hasattr(result, "trades") else pd.DataFrame())
    equity = _ensure_equity_curve(result.equity_curve if hasattr(result, "equity_curve") else pd.DataFrame(), cfg)
    metrics = _compute_metrics(trades, equity, cfg.initial_capital)

    write_csv_atomic(paths["trades"], trades)
    write_csv_atomic(paths["equity"], equity)
    write_json_atomic(paths["metrics"], metrics)
    tracker.complete(metrics)

    return metadata


def main(cli_args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Spot-AI research backtests from the CLI.")
    parser.add_argument("--symbols", help="Comma-separated Binance symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--symbol", help="Legacy single symbol flag")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g. 1m)")
    parser.add_argument("--months", type=int, help="Months back from now")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", help="Output directory for artifacts")
    parser.add_argument("--score-threshold", type=float, default=0.20)
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
        _format_date_for_path(pd.Timestamp(start)),
        _format_date_for_path(pd.Timestamp(end) - pd.Timedelta(days=1)),
    )

    cfg = BacktestConfig(
        start_ts=pd.Timestamp(start),
        end_ts=pd.Timestamp(end),
        is_backtest=True,
        min_score=args.score_threshold,
        min_prob=args.min_prob,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        sizing_mode="fixed_notional",
        trade_size_usd=args.trade_size_usd,
        exit_mode=args.exit_mode,
        random_seed=args.random_seed,
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
        _run_backtest(
            symbols=symbols,
            timeframe=args.timeframe,
            start=start,
            end=end,
            cfg=cfg,
            backtest_id=backtest_id,
            output_dir=output_dir,
            data_dir=Path("data"),
        )
    except Exception as exc:  # pragma: no cover - surface error to user
        print(f"Backtest failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
