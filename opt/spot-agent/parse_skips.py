#!/usr/bin/env python3
import re
import csv
from pathlib import Path

LOG_PATH = Path("analysis_logs/skip_decisions.log")
# ``analyze_skips.py`` expects this CSV path, so keep it in sync.
OUT_PATH = Path("analysis_logs/skip_decisions.csv")

# Example line (wrapped):
# Nov 18 04:45:11 vultr python[67187]: 2025-11-18 04:45:11,769 __main__ - INFO - [SKIP] ZECUSDT:
#   direction=None, size=0.0 - no long signal (score below cutoff) and zero position (low confidence), Score=0.00

line_re = re.compile(
    r"""
    ^(?P<sys_ts>\w+\s+\d+\s+\d+:\d+:\d+).*?       # systemd timestamp
    \[SKIP\]\s+
    (?P<symbol>[A-Z0-9]+):\s+
    direction=(?P<direction>[^,]+),\s+
    # size separator can be ASCII hyphen or Unicode en dash depending on the
    # logger (agent.py currently emits an en dash).
    size=(?P<size>[\d\.]+)\s*[-–]\s*
    (?P<reason>.*?),\s*
    Score=(?P<score>[\d\.]+)
    """,
    re.VERBOSE,
)

def parse_line(line: str):
    m = line_re.search(line)
    if not m:
        return None
    return {
        "sys_ts": m.group("sys_ts"),
        "symbol": m.group("symbol"),
        "direction": m.group("direction"),
        "size": float(m.group("size")),
        "score": float(m.group("score")),
        "raw_reason": m.group("reason"),
    }


def main():
    rows = []
    with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = parse_line(line)
            if row:
                rows.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sys_ts",
                "symbol",
                "direction",
                "size",
                "score",
                "raw_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} skip decisions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
