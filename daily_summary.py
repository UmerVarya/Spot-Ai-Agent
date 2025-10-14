"""Utilities for generating daily natural-language recaps of trading activity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Mapping, Sequence

import pandas as pd

import config
from groq_client import get_groq_client
from groq_safe import safe_chat_completion
from local_llm import generate_daily_recap as generate_local_daily_recap
from log_utils import setup_logger
from trade_storage import load_trade_history_df

logger = setup_logger(__name__)


@dataclass(frozen=True)
class DailyMetrics:
    trading_day: date
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    net_pnl: float
    avg_pnl: float
    win_rate: float
    total_notional: float
    veto_count: int


def _normalise_day(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse trading day from {value!r}")
    return parsed.date()


def _ensure_dataframe(history: Sequence[Mapping[str, object]] | pd.DataFrame | None) -> pd.DataFrame:
    if history is None:
        return load_trade_history_df()
    if isinstance(history, pd.DataFrame):
        return history.copy()
    return pd.DataFrame(list(history))


def _select_timestamp_column(df: pd.DataFrame) -> str | None:
    for column in ("exit_time", "timestamp", "entry_time"):
        if column in df.columns:
            return column
    return None


def _extract_pnl(trade: Mapping[str, object]) -> float | None:
    pnl_keys = ["pnl", "net_pnl", "pnl_usd", "PnL ($)"]
    for key in pnl_keys:
        value = trade.get(key)
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _extract_notional(trade: Mapping[str, object]) -> float | None:
    for key in ("notional", "notional_usd", "size"):
        value = trade.get(key)
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _is_veto(record: Mapping[str, object]) -> bool:
    decision = str(record.get("llm_decision", "")).strip().lower()
    if decision in {"vetoed", "rejected", "no", "false", "0"}:
        return True
    approval = record.get("llm_approval")
    if isinstance(approval, bool):
        return approval is False
    if isinstance(approval, str):
        token = approval.strip().lower()
        if token in {"false", "0", "no", "vetoed"}:
            return True
    reason = str(record.get("reason", "")).lower()
    if "veto" in reason:
        return True
    return False


def _split_records(records: Iterable[Mapping[str, object]]):
    executed: list[Mapping[str, object]] = []
    vetoed: list[Mapping[str, object]] = []
    for record in records:
        if _is_veto(record):
            vetoed.append(record)
        else:
            executed.append(record)
    return executed, vetoed


def _compute_metrics(day: date, trades: Sequence[Mapping[str, object]], veto_count: int) -> DailyMetrics:
    pnls = [p for trade in trades if (p := _extract_pnl(trade)) is not None]
    notional_values = [n for trade in trades if (n := _extract_notional(trade)) is not None]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    breakeven = len(trades) - wins - losses
    net_pnl = float(sum(pnls)) if pnls else 0.0
    avg_pnl = net_pnl / len(trades) if trades else 0.0
    win_rate = wins / len(trades) if trades else 0.0
    total_notional = float(sum(notional_values)) if notional_values else 0.0
    return DailyMetrics(
        trading_day=day,
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        net_pnl=net_pnl,
        avg_pnl=avg_pnl,
        win_rate=win_rate,
        total_notional=total_notional,
        veto_count=veto_count,
    )


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _format_percentage(value: float) -> str:
    return f"{value:.0%}"


def _candidate_reason(trade: Mapping[str, object]) -> str | None:
    for key in ("narrative", "exit_reason", "reason", "notes", "news_summary"):
        value = trade.get(key)
        if not value or value == "N/A":
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _format_trade_highlight(trade: Mapping[str, object]) -> str:
    symbol = str(trade.get("symbol", "Unknown")).upper()
    direction = str(trade.get("direction", "")).upper()
    outcome = (
        trade.get("outcome_desc")
        or trade.get("Outcome Description")
        or trade.get("outcome")
        or "Outcome n/a"
    )
    pnl_value = _extract_pnl(trade)
    pnl_text = _format_money(pnl_value or 0.0)
    reason = _candidate_reason(trade)
    parts = [f"{symbol} {direction or ''}: {outcome} ({pnl_text})".strip()]
    if reason:
        parts.append(f"Context: {reason}")
    return " | ".join(parts)


def _format_veto_highlight(record: Mapping[str, object]) -> str:
    symbol = str(record.get("symbol", "Unknown")).upper()
    direction = str(record.get("direction", "")).upper()
    reason = _candidate_reason(record) or record.get("llm_error") or record.get("reason")
    if reason:
        reason_text = str(reason).strip()
    else:
        reason_text = "No explicit reason provided"
    return f"{symbol} {direction or ''} vetoed – {reason_text}"


def _top_highlights(trades: Sequence[Mapping[str, object]], limit: int = 5) -> list[str]:
    if not trades:
        return []
    sorted_trades = sorted(
        trades,
        key=lambda t: abs(_extract_pnl(t) or 0.0),
        reverse=True,
    )
    return [_format_trade_highlight(t) for t in sorted_trades[:limit]]


def _build_prompt(metrics: DailyMetrics, highlights: Sequence[str], veto_notes: Sequence[str]) -> str:
    lines = [
        "You are the journal assistant for the Spot crypto trading agent.",
        f"Prepare a concise recap for {metrics.trading_day.isoformat()}.",
        "Summarise the trading session in 4-6 sentences covering what worked,",
        "risk management, and lessons for tomorrow. Tie comments to the provided",
        "statistics and highlights. Mention notable vetoes when present.",
        "\nKey metrics:",
        f"- Total trades: {metrics.total_trades}",
        f"- Wins / Losses / Flat: {metrics.wins} / {metrics.losses} / {metrics.breakeven}",
        f"- Win rate: {_format_percentage(metrics.win_rate)}",
        f"- Net PnL: {_format_money(metrics.net_pnl)}",
        f"- Average PnL per trade: {_format_money(metrics.avg_pnl)}",
    ]
    if metrics.total_notional:
        lines.append(f"- Total notional deployed: {_format_money(metrics.total_notional)}")
    lines.append(f"- LLM vetoes: {metrics.veto_count}")
    if highlights:
        lines.append("\nTrade highlights:")
        lines.extend(f"- {item}" for item in highlights)
    if veto_notes:
        lines.append("\nVeto notes:")
        lines.extend(f"- {item}" for item in veto_notes)
    lines.append("\nWrite the recap in plain English with a confident tone.")
    return "\n".join(lines)


def _fallback_summary(metrics: DailyMetrics, highlights: Sequence[str], veto_notes: Sequence[str]) -> str:
    if metrics.total_trades == 0 and not veto_notes:
        return (
            f"No completed trades were recorded on {metrics.trading_day.isoformat()}."
        )
    sentences = [
        (
            f"{metrics.trading_day.isoformat()}: Executed {metrics.total_trades} trade(s) "
            f"with net PnL {_format_money(metrics.net_pnl)} and a "
            f"{_format_percentage(metrics.win_rate)} win rate."
        )
    ]
    if highlights:
        sentences.append("Key plays included " + "; ".join(highlights[:2]) + ".")
    if metrics.veto_count:
        sentences.append(
            f"The LLM vetoed {metrics.veto_count} setup(s){' including ' + '; '.join(veto_notes) if veto_notes else ''}."
        )
    elif veto_notes:
        sentences.append("Notable risk calls: " + "; ".join(veto_notes) + ".")
    return " ".join(sentences)


def generate_daily_summary(
    trading_day: date | datetime | str | None = None,
    *,
    history: Sequence[Mapping[str, object]] | pd.DataFrame | None = None,
    highlight_limit: int = 5,
) -> str:
    day = _normalise_day(trading_day)
    df = _ensure_dataframe(history)
    if df.empty:
        metrics = DailyMetrics(day, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
        return _fallback_summary(metrics, [], [])
    timestamp_column = _select_timestamp_column(df)
    if timestamp_column is None:
        logger.warning("Daily summary could not find a timestamp column in history")
        metrics = DailyMetrics(day, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
        return _fallback_summary(metrics, [], [])
    timestamps = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
    daily_df = df.loc[timestamps.dt.date == day].copy()
    if daily_df.empty:
        metrics = DailyMetrics(day, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
        return _fallback_summary(metrics, [], [])
    records = daily_df.to_dict(orient="records")
    executed, vetoed = _split_records(records)
    metrics = _compute_metrics(day, executed, len(vetoed))
    highlights = _top_highlights(executed, limit=highlight_limit)
    veto_notes = [_format_veto_highlight(record) for record in vetoed]
    prompt = _build_prompt(metrics, highlights, veto_notes)

    client = get_groq_client()
    if client is not None:
        try:
            response = safe_chat_completion(
                client,
                model=config.get_news_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
            )
            if response and response.choices:
                message = response.choices[0].message.content
                if message:
                    return message.strip()
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Groq daily summary generation failed: %s", exc)

    local_response = generate_local_daily_recap(prompt)
    if local_response:
        return local_response.strip()

    return _fallback_summary(metrics, highlights, veto_notes)
