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


def bucket_reason(raw_reason: str) -> str:
    """Map a raw skip reason string into a coarse bucket."""
    if not isinstance(raw_reason, str):
        return "other"

    text = raw_reason.lower()

    # score / confidence related
    if "score below cutoff" in text or "no long signal" in text:
        return "score_below_cutoff"
    if "zero position (low confidence)" in text or "low confidence" in text:
        # still treat as score/pos-size issue
        return "score_below_cutoff"

    # volume gates
    if "vol gate" in text or "volume gate" in text or "floor=" in text:
        return "volume_gate"

    # spread / liquidity
    if "spread" in text and "% of price" in text:
        return "spread_gate"

    # order book imbalance
    if "order book imbalance" in text or "obi" in text:
        return "orderbook_imbalance"

    # macro / news veto
    if "macro veto" in text or "news veto" in text or "news halt" in text:
        return "macro_news_veto"

    # profile / per-symbol min score
    if "profile min score" in text or "below profile minimum" in text:
        return "profile_min_score"

    # cooldown / auction state
    if "cooldown" in text:
        return "cooldown"
    if "auctionstate=balanced" in text or "balanced auction" in text:
        return "auction_state_guard"

    return "other"


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


def summarize_reason_buckets(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    if total == 0:
        return pd.DataFrame(columns=["bucket", "count", "pct_of_total"])

    summary = (
        df.groupby("bucket")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["pct_of_total"] = summary["count"] / float(total)
    return summary


def main() -> None:
    df = load_skip_decisions()
    if df.empty:
        print("No skip decisions found in export.")
        return

    df["bucket"] = df["raw_reason"].apply(bucket_reason)

    summary = summarize_reason_buckets(df)

    print("Top skip reasons (bucket, count, pct_of_total):")
    for _, row in summary.iterrows():
        print(f"{row['bucket']}: {row['count']} ({row['pct_of_total']:.2%})")


if __name__ == "__main__":
    main()
