from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_SCORE_BUCKETS = [4.0, 5.0, 6.0, 7.0, 10.0]


def _percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _win_rate(trades: pd.DataFrame, pnl_col: str = "net_pnl_quote") -> float:
    if trades.empty:
        return 0.0
    pnl = trades[pnl_col].astype(float)
    winners = pnl[pnl > 0]
    return _percent(len(winners), len(pnl))


def _profit_factor(trades: pd.DataFrame, pnl_col: str = "net_pnl_quote") -> float:
    if trades.empty:
        return 0.0
    pnl = trades[pnl_col].astype(float)
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(abs(gains / losses))


def build_equity_curve(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Construct an equity curve with peak and drawdown columns."""

    if trades.empty:
        base = pd.DataFrame(
            {"timestamp": [pd.Timestamp.utcnow()], "equity": [initial_capital]}
        )
        base["peak_equity"] = base["equity"]
        base["drawdown_pct"] = 0.0
        base["drawdown"] = base["drawdown_pct"]
        return base

    pnl = trades.get("net_pnl_quote")
    if pnl is None:
        pnl = trades.get("pnl")
    if pnl is None:
        pnl = trades.get("net_pnl")
    pnl = pnl.astype(float).fillna(0.0)

    timestamps = pd.to_datetime(trades.get("exit_time"))
    equity = initial_capital + pnl.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "equity": equity,
            "peak_equity": peak,
            "drawdown_pct": dd,
            "drawdown": dd,
        }
    )
    return df


def summarise_backtest_metrics(
    equity_curve: pd.DataFrame, trades: pd.DataFrame, initial_capital: float
) -> Dict[str, float]:
    """Compute headline metrics for display."""

    if equity_curve.empty:
        return {k: 0.0 for k in [
            "total_return_pct",
            "annual_return_pct",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown_pct",
            "win_rate",
            "profit_factor",
        ]}

    equity = equity_curve["equity"].astype(float)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    if len(equity_curve) > 1:
        start = equity_curve["timestamp"].iloc[0]
        end = equity_curve["timestamp"].iloc[-1]
        days = max((end - start).days, 1)
    else:
        days = 1
    annual_return = (1 + total_return) ** (365 / days) - 1 if days else total_return

    pnl_col = "net_pnl_quote" if "net_pnl_quote" in trades.columns else "pnl"
    sharpe = equity.pct_change().mean() / (equity.pct_change().std(ddof=1) or 1e-9)
    sharpe *= np.sqrt(252)
    downside = equity.pct_change()
    downside = downside[downside < 0]
    sortino = downside.mean() / (downside.std(ddof=1) or 1e-9)
    sortino *= -np.sqrt(252)
    max_dd = float(equity_curve["drawdown_pct"].min()) if "drawdown_pct" in equity_curve else 0.0
    calmar = total_return / abs(max_dd or 1e-9)
    win_rate = _win_rate(trades, pnl_col=pnl_col)
    profit_factor = _profit_factor(trades, pnl_col=pnl_col)

    return {
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_pct": max_dd * 100,
        "win_rate": win_rate * 100,
        "profit_factor": profit_factor,
    }


def per_symbol_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    pnl_col = "net_pnl_quote" if "net_pnl_quote" in trades.columns else "pnl"
    rows: List[Dict[str, object]] = []
    for symbol, group in trades.groupby("symbol"):
        pnl = group[pnl_col].astype(float).sum()
        r_values = group.get("r_multiple", np.nan)
        avg_r = float(np.nanmean(r_values)) if len(r_values) else np.nan
        avg_holding = float(
            group.get("holding_time_minutes", pd.Series(dtype=float)).astype(float).mean()
        )
        rows.append(
            {
                "symbol": symbol,
                "num_trades": len(group),
                "total_pnl_quote": pnl,
                "avg_r": avg_r,
                "win_rate": _win_rate(group, pnl_col=pnl_col) * 100,
                "sharpe": group[pnl_col].astype(float).mean() / (
                    group[pnl_col].astype(float).std(ddof=1) or 1e-9
                ),
                "max_drawdown_pct": float("nan"),
                "profit_factor": _profit_factor(group, pnl_col=pnl_col),
                "avg_holding_minutes": avg_holding,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("total_pnl_quote", ascending=False)


def score_bucket_metrics(
    trades: pd.DataFrame,
    buckets: Iterable[float] = DEFAULT_SCORE_BUCKETS,
    score_column: str = "score_at_entry",
) -> pd.DataFrame:
    if score_column not in trades.columns or trades.empty:
        return pd.DataFrame()
    scores = trades[score_column].astype(float)
    labels = [f"{buckets[i]}-{buckets[i+1]-0.1}" for i in range(len(buckets) - 2)]
    labels.append(f"{buckets[-2]}+")
    trades = trades.copy()
    trades["score_bucket"] = pd.cut(scores, buckets, labels=labels, include_lowest=True)
    return _bucket_summary(trades, "score_bucket")


def _bucket_summary(trades: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    if bucket_col not in trades.columns:
        return pd.DataFrame()
    pnl_col = "net_pnl_quote" if "net_pnl_quote" in trades.columns else "pnl"
    rows: List[Dict[str, object]] = []
    for bucket, group in trades.groupby(bucket_col):
        pnl = group[pnl_col].astype(float)
        rows.append(
            {
                bucket_col: bucket,
                "num_trades": len(group),
                "avg_r": float(np.nanmean(group.get("r_multiple", np.nan))),
                "total_pnl_quote": float(pnl.sum()),
                "win_rate": _win_rate(group, pnl_col=pnl_col) * 100,
                "profit_factor": _profit_factor(group, pnl_col=pnl_col),
            }
        )
    return pd.DataFrame(rows)


def session_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if "session" not in trades.columns or trades.empty:
        return pd.DataFrame()
    return _bucket_summary(trades, "session")


def outcome_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if "outcome_type" not in trades.columns or trades.empty:
        return pd.DataFrame()
    counts = trades["outcome_type"].value_counts().rename_axis("outcome_type").reset_index(name="count")
    counts["percent"] = counts["count"] / counts["count"].sum() * 100
    return counts


def add_optional_regime(trades: pd.DataFrame, regimes: Optional[pd.Series] = None) -> pd.DataFrame:
    if regimes is None or trades.empty:
        return trades
    trades = trades.copy()
    trades["regime"] = regimes.reindex(trades.index).values
    return trades
