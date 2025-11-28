from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Collection, Dict, List, Optional

logger = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"scan_stats_(\d{4}-\d{2}-\d{2})\.json$")


@dataclass
class DailySymbolStats:
    date: date
    symbol: str
    total_scans: int
    candidates: int
    passed_alpha: int
    passed_prob: int
    llm_approved: int
    trades_entered: int
    llm_calls_total: int
    llm_calls_failed: int
    primary_reasons: Dict[str, int]


@dataclass
class DailyAggregate:
    date: date
    total_scans: int
    candidates: int
    passed_alpha: int
    passed_prob: int
    llm_approved: int
    trades_entered: int
    llm_calls_total: int
    llm_calls_failed: int
    primary_reasons: Dict[str, int]


def _parse_date_from_filename(path: Path) -> Optional[date]:
    match = DATE_PATTERN.search(path.name)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def list_available_dates(base_dir: Path) -> List[date]:
    dates: set[date] = set()
    for path in base_dir.glob("scan_stats_*.json"):
        parsed = _parse_date_from_filename(path)
        if parsed:
            dates.add(parsed)
    return sorted(dates)


def load_all_daily_stats(base_dir: Path) -> List[DailySymbolStats]:
    stats: list[DailySymbolStats] = []
    for path in sorted(base_dir.glob("scan_stats_*.json")):
        parsed_date = _parse_date_from_filename(path)
        if not parsed_date:
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception:
            logger.warning("Failed to parse %s", path)
            continue
        symbols_block = raw.get("symbols") or {}
        if not isinstance(symbols_block, dict):
            continue
        for symbol, payload in symbols_block.items():
            if not isinstance(payload, dict):
                continue
            filters = payload.get("filters", {}) or {}
            llm_block = payload.get("llm", {}) or {}
            final_block = payload.get("final", {}) or {}
            reasons = payload.get("primary_no_trade_reasons", {}) or {}
            stats.append(
                DailySymbolStats(
                    date=parsed_date,
                    symbol=str(symbol),
                    total_scans=int(payload.get("total_scans", 0)),
                    candidates=int(payload.get("scans_with_candidate", 0)),
                    passed_alpha=int(filters.get("passed_alpha_score_gate", 0)),
                    passed_prob=int(filters.get("passed_prob_gate", 0)),
                    llm_approved=int(llm_block.get("llm_approved", 0)),
                    trades_entered=int(final_block.get("trades_entered", 0)),
                    llm_calls_total=int(llm_block.get("llm_calls_total", 0)),
                    llm_calls_failed=int(llm_block.get("llm_calls_failed", 0)),
                    primary_reasons={str(k): int(v) for k, v in reasons.items()},
                )
            )
    return sorted(stats, key=lambda s: s.date)


def filter_stats_by_date_and_symbols(
    stats: List[DailySymbolStats],
    start_date: Optional[date],
    end_date: Optional[date],
    symbols: Optional[Collection[str]] = None,
) -> List[DailySymbolStats]:
    symbol_set = {sym.upper() for sym in symbols} if symbols else None
    filtered: list[DailySymbolStats] = []
    for item in stats:
        if start_date and item.date < start_date:
            continue
        if end_date and item.date > end_date:
            continue
        if symbol_set and item.symbol.upper() not in symbol_set:
            continue
        filtered.append(item)
    return filtered


def aggregate_by_day(stats: List[DailySymbolStats]) -> List[DailyAggregate]:
    by_day: dict[date, Dict[str, object]] = {}
    for item in stats:
        bucket = by_day.setdefault(
            item.date,
            {
                "total_scans": 0,
                "candidates": 0,
                "passed_alpha": 0,
                "passed_prob": 0,
                "llm_approved": 0,
                "trades_entered": 0,
                "llm_calls_total": 0,
                "llm_calls_failed": 0,
                "primary_reasons": {},
            },
        )
        bucket["total_scans"] = int(bucket["total_scans"]) + item.total_scans
        bucket["candidates"] = int(bucket["candidates"]) + item.candidates
        bucket["passed_alpha"] = int(bucket["passed_alpha"]) + item.passed_alpha
        bucket["passed_prob"] = int(bucket["passed_prob"]) + item.passed_prob
        bucket["llm_approved"] = int(bucket["llm_approved"]) + item.llm_approved
        bucket["trades_entered"] = int(bucket["trades_entered"]) + item.trades_entered
        bucket["llm_calls_total"] = int(bucket["llm_calls_total"]) + item.llm_calls_total
        bucket["llm_calls_failed"] = int(bucket["llm_calls_failed"]) + item.llm_calls_failed
        merged_reasons: dict[str, int] = bucket["primary_reasons"]  # type: ignore[assignment]
        for key, value in item.primary_reasons.items():
            merged_reasons[key] = merged_reasons.get(key, 0) + int(value)
    aggregates: list[DailyAggregate] = []
    for stats_date, payload in by_day.items():
        aggregates.append(
            DailyAggregate(
                date=stats_date,
                total_scans=int(payload["total_scans"]),
                candidates=int(payload["candidates"]),
                passed_alpha=int(payload["passed_alpha"]),
                passed_prob=int(payload["passed_prob"]),
                llm_approved=int(payload["llm_approved"]),
                trades_entered=int(payload["trades_entered"]),
                llm_calls_total=int(payload["llm_calls_total"]),
                llm_calls_failed=int(payload["llm_calls_failed"]),
                primary_reasons=dict(payload["primary_reasons"]),
            )
        )
    return sorted(aggregates, key=lambda a: a.date)


def to_funnel_dataframe(daily: List[DailyAggregate]):
    import pandas as pd  # local import to avoid heavy global dependency

    data = [
        {
            "date": item.date,
            "scans": item.total_scans,
            "candidates": item.candidates,
            "passed_alpha": item.passed_alpha,
            "passed_prob": item.passed_prob,
            "llm_approved": item.llm_approved,
            "trades_entered": item.trades_entered,
        }
        for item in sorted(daily, key=lambda d: d.date)
    ]
    return pd.DataFrame(data)


def to_reasons_dataframe(daily: List[DailyAggregate]):
    import pandas as pd  # local import to avoid heavy global dependency

    if not daily:
        return pd.DataFrame()
    all_reasons: set[str] = set()
    for item in daily:
        all_reasons.update(item.primary_reasons.keys())
    sorted_days = sorted(daily, key=lambda d: d.date)
    rows: list[Dict[str, object]] = []
    for item in sorted_days:
        row: Dict[str, object] = {"date": item.date}
        for reason in all_reasons:
            row[reason] = item.primary_reasons.get(reason, 0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")
    return df[sorted(all_reasons)] if all_reasons else df


def aggregate_by_symbol(stats: List[DailySymbolStats]):
    import pandas as pd  # local import to avoid heavy global dependency

    if not stats:
        return pd.DataFrame()
    rows = [
        {
            "symbol": item.symbol,
            "total_scans": item.total_scans,
            "candidates": item.candidates,
            "passed_alpha": item.passed_alpha,
            "passed_prob": item.passed_prob,
            "llm_approved": item.llm_approved,
            "trades_entered": item.trades_entered,
            "llm_calls_total": item.llm_calls_total,
            "llm_calls_failed": item.llm_calls_failed,
        }
        for item in stats
    ]
    df = pd.DataFrame(rows)
    grouped = df.groupby("symbol", as_index=False).sum(numeric_only=True)
    grouped["conv_candidates_per_scan"] = grouped.apply(
        lambda row: (row["candidates"] / row["total_scans"]) if row["total_scans"] else 0.0,
        axis=1,
    )
    grouped["conv_trades_per_candidate"] = grouped.apply(
        lambda row: (row["trades_entered"] / row["candidates"]) if row["candidates"] else 0.0,
        axis=1,
    )
    grouped["conv_trades_per_scan"] = grouped.apply(
        lambda row: (row["trades_entered"] / row["total_scans"]) if row["total_scans"] else 0.0,
        axis=1,
    )
    return grouped.sort_values("symbol").reset_index(drop=True)

