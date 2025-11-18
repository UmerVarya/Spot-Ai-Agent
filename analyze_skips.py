#!/usr/bin/env python3
"""Analyze parsed skip decisions and bucket them into high-level reasons."""

from __future__ import annotations

import collections
import csv
import os
import sys
from typing import Dict, Iterable, Tuple

# Where parse_skips.py writes the CSV
DEFAULT_CSV_PATH = "analysis_logs/skip_decisions_parsed.csv"
CSV_PATH = os.getenv("SKIP_DECISIONS_CSV", DEFAULT_CSV_PATH)

# High-level buckets + the substrings that should map into them.
# You can safely extend/tune this list as we learn more patterns.
BUCKET_SUBSTRINGS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "low_score",
        (
            "below min_score",
            "below min score",
            "score too low",
            "score <",  # e.g. "score 2.3 < min_score 3.0"
            "insufficient score",
        ),
    ),
    (
        "low_volume",
        (
            "low volume",
            "volume gate",
            "volume below threshold",
            "quote volume",
            "insufficient liquidity",
            "volume <",
            "skipping due to low volume",
        ),
    ),
    (
        "macro_news",
        (
            "macro halt",
            "macro gate",
            "macro bias",
            "news halt",
            "news risk",
            "news veto",
        ),
    ),
    (
        "cooldown",
        (
            "cooldown",
            "in cooldown window",
            "recent exit",
            "cool-down",
        ),
    ),
    (
        "risk_veto",
        (
            "risk veto",
            "llm veto",
            "vetoed by",
            "risk engine veto",
        ),
    ),
    (
        "concurrency",
        (
            "max concurrent",
            "too many open trades",
            "concurrency limit",
            "existing open trade",
        ),
    ),
    (
        "session_filter",
        (
            "session filter",
            "asia session",
            "europe session",
            "us session",
            "session bias",
        ),
    ),
    (
        "symbol_filter",
        (
            "blacklist",
            "stablecoin",
            "skipping stablecoin",
            "symbol filter",
            "excluded symbol",
        ),
    ),
    (
        "data_not_ready",
        (
            "cache not warm",
            "signal cache not ready",
            "insufficient history",
            "no candles",
            "missing cache",
        ),
    ),
    (
        "ws_down",
        (
            "websocket",
            "ws bridge",
            "no live prices",
            "stream down",
        ),
    ),
)


def _pick_reason_column(fieldnames: Iterable[str]) -> str:
    """Try to pick the column that stores the textual reason string."""

    candidates = [
        "decision_reason",
        "raw_reason",
        "reason",
        "reason_text",
    ]
    lower_map = {name.lower(): name for name in fieldnames}

    for wanted in candidates:
        if wanted.lower() in lower_map:
            return lower_map[wanted.lower()]

    for name in fieldnames:
        if "reason" in name.lower():
            return name

    return ""


def _bucket_from_substrings(reason: str) -> str:
    """Bucket a free-form reason string using BUCKET_SUBSTRINGS."""

    if not reason:
        return "unknown"

    r = reason.lower()
    for bucket, needles in BUCKET_SUBSTRINGS:
        if any(n in r for n in needles):
            return bucket

    return "other"


def _bucket_for_row(row: Dict[str, str], reason_col: str, bucket_col: str) -> Tuple[str, str]:
    """Decide the bucket for a row. Returns (bucket, reason_string)."""

    existing_bucket = ""
    if bucket_col:
        existing_bucket = (row.get(bucket_col) or "").strip()

    if existing_bucket and existing_bucket.lower() not in ("", "other", "unknown"):
        return existing_bucket.strip(), row.get(reason_col, "").strip()

    reason = (row.get(reason_col) or "").strip() if reason_col else ""
    bucket = _bucket_from_substrings(reason)
    return bucket, reason


def analyze(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    counts = collections.Counter()
    other_reasons = collections.Counter()
    total = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[ERROR] CSV has no header/fieldnames.", file=sys.stderr)
            sys.exit(1)

        fieldnames = reader.fieldnames
        reason_col = _pick_reason_column(fieldnames)
        bucket_col = "bucket" if "bucket" in fieldnames else ""

        if not reason_col:
            print(
                f"[WARN] Could not find a 'reason' column in {csv_path}. Fields: {fieldnames}",
                file=sys.stderr,
            )

        for row in reader:
            total += 1
            bucket, raw_reason = _bucket_for_row(row, reason_col, bucket_col)
            counts[bucket] += 1
            if bucket == "other":
                other_reasons[raw_reason or "<EMPTY>"] += 1

    if not total:
        print("[INFO] No rows found in skip_decisions CSV.")
        return

    print("Top skip reasons (bucket, count, pct_of_total):")
    for bucket, count in counts.most_common():
        pct = (count / total) * 100.0
        print(f"{bucket:15s} {count:6d} ({pct:6.2f}%)")

    top_bucket, top_count = counts.most_common(1)[0]
    if top_bucket == "other" and top_count >= 0.8 * total:
        print(
            "\n[DEBUG] 'other' is dominating; here are the most common raw "
            "reason strings inside the 'other' bucket:\n"
        )
        for reason, count in other_reasons.most_common(25):
            pct = (count / total) * 100.0
            print(f"- {count:6d} ({pct:6.2f}%): {reason}")


def main() -> None:
    csv_path = CSV_PATH
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    analyze(csv_path)


if __name__ == "__main__":
    main()
