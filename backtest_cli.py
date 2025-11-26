from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest import BacktestResult, BacktestProgress, run_backtest
from backtest.types import ProgressCallback

DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/spot_data/backtests")


def _parse_datetime(value: str) -> datetime:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def _resolve_date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if args.start and args.end:
        start = _parse_datetime(args.start)
        end = (_parse_datetime(args.end) + pd.Timedelta(days=1)).to_pydatetime()
    elif args.months:
        end_dt = _parse_datetime(args.end) if args.end else now
        end = (pd.Timestamp(end_dt) + pd.Timedelta(days=1)).to_pydatetime()
        start = (pd.Timestamp(end_dt) - pd.DateOffset(months=args.months)).to_pydatetime()
    else:
        raise ValueError("Provide either --start/--end or --months to define the date range.")

    if start > end:
        raise ValueError("Start date must be before end date.")

    return start, end


def _make_progress_printer() -> ProgressCallback:
    last_percent = -5
    last_phase: Optional[str] = None

    def _printer(progress: BacktestProgress) -> None:
        nonlocal last_percent, last_phase
        percent = int(progress.percent * 100) if progress.total > 0 else 0
        should_print = progress.phase != last_phase or percent - last_percent >= 5 or progress.current >= progress.total
        if should_print:
            last_percent = percent
            last_phase = progress.phase
            print(
                f"[BACKTEST] Progress: {progress.phase} "
                f"{progress.current}/{progress.total} ({percent:.0f}%)",
                flush=True,
            )

    return _printer


def _write_outputs(
    result: BacktestResult,
    symbol: str,
    timeframe: str,
    start: datetime,
    end_inclusive: datetime,
    output_dir: Path,
) -> None:
    start_str = start.date().isoformat()
    end_str = end_inclusive.date().isoformat()
    symbol_upper = symbol.upper()

    trades_path = output_dir / f"{symbol_upper}_{timeframe}_{start_str}_{end_str}_trades.csv"
    equity_path = output_dir / f"{symbol_upper}_{timeframe}_{start_str}_{end_str}_equity.csv"
    metrics_path = output_dir / f"{symbol_upper}_{timeframe}_{start_str}_{end_str}_metrics.csv"

    trades_df = result.trades if hasattr(result, "trades") else result.trades_df  # type: ignore[attr-defined]
    equity_df = result.equity_curve if hasattr(result, "equity_curve") else result.equity_df  # type: ignore[attr-defined]
    metrics_df = pd.DataFrame([getattr(result, "metrics", {}) or {}])

    outputs = [
        (trades_path, trades_df),
        (equity_path, equity_df),
        (metrics_path, metrics_df),
    ]

    tmp_paths = []
    try:
        for path, df in outputs:
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            df.to_csv(tmp_path, index=False)
            tmp_paths.append((tmp_path, path))

        for tmp_path, final_path in tmp_paths:
            tmp_path.replace(final_path)
    except Exception:
        for tmp_path, _ in tmp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    print("[BACKTEST] Done.")
    print(f"[BACKTEST] Trades:  {trades_path}")
    print(f"[BACKTEST] Equity:  {equity_path}")
    print(f"[BACKTEST] Metrics: {metrics_path}")


def main(cli_args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Spot-AI backtest from the CLI.")
    parser.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (default: 1m)")
    parser.add_argument("--months", type=int, default=None, help="Number of months to backtest ending at --end or now")
    parser.add_argument("--start", help="Start date (ISO format)")
    parser.add_argument("--end", help="End date (ISO format, default: now)")
    parser.add_argument("--output-dir", help="Output directory for CSV results", default=str(DEFAULT_OUTPUT_DIR))

    args = parser.parse_args(cli_args)

    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    start, end = _resolve_date_range(args)
    end_inclusive = (pd.Timestamp(end) - pd.Timedelta(days=1)).to_pydatetime()

    symbol = args.symbol.upper()
    timeframe = args.timeframe

    print(f"[BACKTEST] Starting backtest for {symbol} {timeframe} from {start} to {end_inclusive}")
    print(f"[BACKTEST] Results will be saved under: {output_dir}")

    progress_printer = _make_progress_printer()

    try:
        result = run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            output_dir=output_dir,
            progress_callback=progress_printer,
        )
    except Exception as exc:  # pragma: no cover - surface to user
        print(f"[BACKTEST] Failed: {exc}")
        return 1

    try:
        _write_outputs(result, symbol, timeframe, start, end_inclusive, output_dir)
    except Exception as exc:  # pragma: no cover - surface to user
        print(f"[BACKTEST] Failed to write outputs: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
