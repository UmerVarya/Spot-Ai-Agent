#!/usr/bin/env python3
"""Analyze skip decisions exported by ``parse_skips.py``.

This helper expects ``analysis_logs/skip_decisions_parsed.csv`` to be populated by
``opt/spot-agent/parse_skips.py``.  It aggregates the cleaned skip reasons and
prints the most common entries to STDOUT so we can quickly spot recurring
issues.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path("analysis_logs/skip_decisions_parsed.csv")


def load_skip_decisions(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the exported skip decisions and normalize the raw reason strings."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Skip decision export not found: {csv_path}. Run parse_skips.py first."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"symbol", "direction", "size", "score", "raw_reason"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "Skip decision export is missing expected columns: " + ", ".join(sorted(missing))
        )

    df["raw_reason"] = df["raw_reason"].fillna("").astype(str).str.strip()

    # Ensure numeric fields are numeric so downstream stats (histograms, etc.) work.
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")

    return df


def categorize_reason(raw_reason: str) -> str:
    """Map a raw skip reason string to a coarse bucket."""

    if not isinstance(raw_reason, str) or not raw_reason:
        return "other"

    text = raw_reason.lower()

    if "score below cutoff" in text or ("no long signal" in text) or (
        "score" in text and "cutoff" in text
    ):
        return "score_below_cutoff"

    if "zero position" in text and "low confidence" in text:
        return "low_confidence_possize_zero"

    if "vol gate" in text or "volgate" in text or "volume" in text:
        return "volume_gate"

    if "spread" in text and ("0.1% of price" in text or ">0.1%" in text or "threshold" in text):
        return "spread_gate"

    if "order book imbalance" in text or "orderbook" in text or "obi" in text:
        return "orderbook_imbalance_gate"

    if "macro/news" in text or (
        ("macro" in text or "news" in text) and any(k in text for k in ["veto", "halt", "pause"])
    ):
        return "macro_news_veto"

    if "auction=" in text or "auction state" in text:
        return "auction_state_guard"

    profile_keywords = ["profile", "atr", "dominance", "trend", "min-score", "min score"]
    if any(keyword in text for keyword in profile_keywords):
        return "profile_guard"

    return "other"


def summarize_reason_buckets(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    summary = (
        df["reason_bucket"].value_counts().reset_index(name="count").rename(
            columns={"index": "bucket"}
        )
    )
    summary["pct_of_total"] = (summary["count"] / total * 100).round(1)
    return summary


def main() -> None:
    df = load_skip_decisions()
    df["reason_bucket"] = df["raw_reason"].map(categorize_reason).fillna("other")

    summary = summarize_reason_buckets(df)

    print("Top skip reasons (bucket, count, pct_of_total):")
    for _, row in summary.iterrows():
        bucket = row["bucket"]
        count = int(row["count"])
        pct = row["pct_of_total"]
        print(f"{bucket:<28}{count:>7}  {pct:>5.1f}%")


if __name__ == "__main__":
    main()
