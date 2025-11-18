#!/usr/bin/env python3
"""Analyze skip decisions exported by ``parse_skips.py``.

This helper expects ``analysis_logs/skip_decisions.csv`` to be populated by
``opt/spot-agent/parse_skips.py``.  It aggregates the cleaned skip reasons and
prints the most common entries to STDOUT so we can quickly spot recurring
issues.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path("analysis_logs/skip_decisions.csv")


def load_skip_decisions(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the exported skip decisions and normalize the reason strings."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Skip decision export not found: {csv_path}. Run parse_skips.py first."
        )

    df = pd.read_csv(csv_path)

    # Ensure 'reason' is always a string before applying .str operations
    df["reason"] = df["reason"].astype(str)

    # Extract main reason text before parentheses
    df["reason_clean"] = (
        df["reason"].str.extract(r"^(.*?)(?:\(|$)", expand=False).str.strip()
    )

    # Replace weird values / trim spaces
    df["reason_clean"] = df["reason_clean"].fillna("Unknown").str.strip()

    return df


def summarize_reasons(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return the ``limit`` most common cleaned reasons."""
    summary = (
        df["reason_clean"].value_counts().reset_index(name="count").rename(
            columns={"index": "reason_clean"}
        )
    )
    return summary.head(limit)


def main() -> None:
    df = load_skip_decisions()
    summary = summarize_reasons(df)

    print("Top skip reasons:\n")
    for _, row in summary.iterrows():
        print(f"{row['reason_clean']}: {int(row['count'])}")


if __name__ == "__main__":
    main()
