"""Lightweight utilities for inspecting probability fields in trade logs.

The functions here are intentionally decoupled from the main runtime so they
can be executed ad-hoc when analysing backtest outputs or live trade history
exports. They provide summary statistics for probability/confidence columns to
help reason about calibration and thresholds without altering trading logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def summarise_probability_columns(
    csv_path: str | Path,
    probability_columns: Iterable[str] = (
        "probability",
        "ml_probability",
        "confidence",
        "score",
    ),
) -> pd.DataFrame:
    """Return descriptive stats for available probability-like columns.

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV file containing backtest trades or live trade history.
    probability_columns : iterable of str, optional
        Candidate column names to summarise. Columns missing from the CSV are
        silently skipped so the helper can be reused across different log
        formats.

    Notes
    -----
    This helper is purely diagnostic and is not invoked by the agent or
    backtesting pipeline. It is intended for manual analysis of probability
    calibration and threshold choices.
    """

    path = Path(csv_path)
    df = pd.read_csv(path)
    available = [col for col in probability_columns if col in df.columns]
    if not available:
        return pd.DataFrame()

    summaries: dict[str, pd.Series] = {}
    for col in available:
        series = pd.to_numeric(df[col], errors="coerce")
        summaries[col] = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

    return pd.DataFrame(summaries)


if __name__ == "__main__":  # pragma: no cover - diagnostic entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Summarise probability columns in a trade log")
    parser.add_argument("csv_path", help="Path to a trade history CSV (backtest or live)")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=["probability", "ml_probability", "confidence", "score"],
        help="Candidate column names to summarise",
    )
    args = parser.parse_args()

    result = summarise_probability_columns(args.csv_path, args.columns)
    if result.empty:
        print("No probability-like columns found in provided file.")
    else:
        pd.set_option("display.max_columns", None)
        print(result)
